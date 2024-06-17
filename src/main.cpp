#include <cstdlib>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <string>
#include <sstream>
#include <fstream>
#include <numeric>

#include "../lib/param.hpp"
#include "../lib/gbm.hpp"
#include "../lib/ddpg.hpp"

std::ofstream out;
std::default_random_engine seed(std::chrono::system_clock::now().time_since_epoch().count());

std::vector<GBMParam> param;
std::vector<std::vector<double>> path;

std::vector<Memory> memory;

std::vector<double> mean_reward;
std::vector<double> test;
std::vector<std::vector<double>> test_action;
int ntickers;
int ext;
std::vector<std::string> tickers;
std::string mode;
std::string checkpoint;
int decayt;

std::vector<double> sample_state(unsigned int t) {
    std::vector<double> state(ntickers);
    for(unsigned int i = 0; i < ntickers; i++)
        state[i] = (path[i][t] - path[i][t-OBS]) / path[i][t-OBS];
    return state;
}

void write() {
    out.open("./res/path");
    for(unsigned int i = 0; i < ntickers; i++)
        out << tickers[i] << (i != ntickers-1 ? "," : "\n");
    for(unsigned int t = 0; t <= ext; t++) {
        for(unsigned int i = 0; i < ntickers; i++)
            out << path[i][t] << (i != ntickers-1 ? "," : "\n");
    }
    out.close();

    out.open("./res/log");
    out << "mr\n";
    for(unsigned int i = 0; i < ITR; i++)
        out << mean_reward[i] << "\n";
    out.close();

    out.open("./res/test");
    out << "test\n";
    for(double &x: test)
        out << x << "\n";
    out.close();

    out.open("./res/action");
    for(unsigned int i = 0; i < ntickers; i++)
        out << tickers[i] << (i != ntickers-1 ? "," : "\n");
    for(unsigned int t = 0; t < ext-1; t++) {
        for(unsigned int i = 0; i < ntickers; i++)
            out << test_action[i][t] << (i != ntickers-1 ? "," : "\n");
    }
    out.close();

    std::system("./python/plot.py");
}

void clean() {
    std::vector<GBMParam>().swap(param);
    std::vector<std::vector<double>>().swap(path);
    std::vector<Memory>().swap(memory);
    std::vector<double>().swap(mean_reward);
    std::vector<double>().swap(test);
    std::vector<std::vector<double>>().swap(test_action);
}

void boot(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <mode> <ticker1> <ticker2> ... <checkpoint>\n";
        std::exit(EXIT_FAILURE);
    }
    
    mode = argv[1];
    ntickers = argc - 3;
    
    std::cout << "mode: " << mode << "\n";
    std::cout << "ntickers: " << ntickers << ": ";

    std::string openingcmd = "./python/download.py";

    for (int i = 2; i < argc - 1; ++i) {
        openingcmd += " " + std::string(argv[i]);
        tickers.push_back(std::string(argv[i]));
        std::cout << std::string(argv[i]) << ", ";
    }

    std::cout << "\n";
    checkpoint = argv[argc - 1];
    std::cout << "checkpoint: " << checkpoint << "\n";
    std::cout << "downloading.. \n";
    std::system(openingcmd.c_str());
    std::cout << "finished  \n";
}

void readfile(std::vector<std::vector<double>> &dat) {
    std::ifstream file("./data/merge.csv");
    if (!file.is_open()) {
        std::cout << "Error opening file\n";
        return;
    }

    // Skip the header row
    std::string header;
    std::getline(file, header);

    std::string line;
    while (std::getline(file, line)) {
        std::vector<double> row;
        std::istringstream iss(line);
        std::string value;
        while (std::getline(iss, value, ',')) {
            row.push_back(std::stod(value));
        }
        dat.push_back(row);
    }

    std::cout << "transposing data \n";
    std::vector<std::vector<double>> transposed(dat[0].size(), std::vector<double>(dat.size()));
    for (size_t i = 0; i < dat.size(); ++i) {
        for (size_t j = 0; j < dat[i].size(); ++j) {
            transposed[j][i] = dat[i][j];
        }
    }
    dat = transposed;

    std::cout << "path length: " << dat[0].size() << "\n";
}

double decay(double alpha_init, double t, double size, double itr, double k) {
    return alpha_init * std::exp(double(t) / (size * itr) * k);
}

double jun_decay(double alpha_init, double t, double T, double k) {
    return alpha_init * std::exp(t * k / T);
}

int main(int argc, char *argv[])
{
    boot(argc, argv);
    std::cout << std::fixed;
    std::cout.precision(15);  

    readfile(path);
    ext = path[0].size();

    
    Net actor;
    actor.add_layer(ntickers+0, ntickers+ntickers);
    actor.add_layer(ntickers+ntickers, ntickers+ntickers);
    actor.add_layer(ntickers+ntickers, ntickers+ntickers);
    actor.add_layer(ntickers+ntickers, ntickers+0);
    actor.use_softmax();
    actor.init(seed);

    Net critic;
    critic.add_layer(ntickers+ntickers, ntickers+0);
    critic.add_layer(ntickers+0, ntickers+0);
    critic.add_layer(ntickers+0, ntickers+0);
    critic.add_layer(ntickers+0, 1);
    critic.init(seed);

    DDPG ddpg(actor, critic);

    double eps = EPS_INIT;
    double alpha = ALPHA_INIT;
    double k = log10(ALPHA_FINAL) - log10(ALPHA_INIT);
    decayt = 0;

    for(unsigned int itr = 0; itr < ITR; itr++) {
        unsigned int update_count = 0;
        double reward_sum = 0.00, q_sum = 0.00;
        for(unsigned int t = OBS; t < ext; t++) {

            if((itr+1)*t > OBS && eps > EPS_MIN)
                eps += (EPS_MIN - EPS_INIT) / CAPACITY;

            std::vector<double> state = sample_state(t);
            std::vector<double> action = ddpg.epsilon_greedy(state, eps);
            std::vector<double> next_state = sample_state(t+1);

            double reward = 0.00;
            for(unsigned int i = 0; i < ntickers; i++)
                reward += path[i][t+1] * action[i];
            reward = log10(reward);
            reward_sum += reward; 
            
            memory.push_back(Memory(state, action, next_state, reward));

            if(memory.size() == CAPACITY) {
                std::vector<unsigned int> index(CAPACITY, 0);
                std::iota(index.begin(), index.end(), 0);
                std::shuffle(index.begin(), index.end(), seed);
                index.erase(index.begin() + BATCH, index.end());

                // alpha = decay(ALPHA_INIT, decayt, path[0].size(), ITR, K); 
                decayt ++; 
                alpha = jun_decay(ALPHA_INIT, decayt, ITR * ext, k);


                for(unsigned int &k: index)
                    q_sum += ddpg.optimize(memory[k], GAMMA, alpha, LAMBDA);
                update_count += BATCH;

                memory.erase(memory.begin());
                std::vector<unsigned int>().swap(index);
            }
        }

        mean_reward.push_back(reward_sum / (ext-1));

        std::cout << "ITR=" << itr << " ";
        std::cout << "MR=" << mean_reward.back() << " ";
        std::cout << "ALPHA="<<alpha << " decayt="<<decayt << " ";
        std::cout << "Q=" << q_sum / update_count << "\n";
    }

    test_action.resize(ntickers, std::vector<double>(ext-1));

    for(unsigned int t = OBS; t < ext; t++) {
        std::vector<double> state = sample_state(t);
        std::vector<double> action = ddpg.epsilon_greedy(state, 0.00);

        double reward = 0.00;
        for(unsigned int i = 0; i < ntickers; i++) {
            reward += path[i][t+1] * action[i];
            test_action[i][t-1] = action[i];
        }
        test.push_back(reward);

        std::vector<double>().swap(state);
        std::vector<double>().swap(action);
    }

    write();
    clean();
    
    return 0;
}