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
#include <numeric>

#include "../lib/param.hpp"
#include "../lib/gbm.hpp"
#include "../lib/ddpg.hpp"

std::ofstream out;
std::default_random_engine seed(std::chrono::system_clock::now().time_since_epoch().count());

std::vector<std::vector<double>> path;
std::vector<std::vector<double>> test_action;
int ntickers;
int ext;
std::vector<std::string> tickers;
std::string checkpoint;
std::string mode;

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
        std::cerr << "Error opening file\n";
        std::exit(EXIT_FAILURE);
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

    std::cout << "path length or ext: " << dat[0].size() << "\n";
}

int main(int argc, char *argv[]) {
    boot(argc, argv);

    std::cout << std::fixed;
    std::cout.precision(6);

    readfile(path);

    Net actor;
    actor.add_layer(ntickers + 0, ntickers + ntickers);
    actor.add_layer(ntickers + ntickers, ntickers + ntickers);
    actor.add_layer(ntickers + ntickers, ntickers + ntickers);
    actor.add_layer(ntickers + ntickers, ntickers + 0);
    actor.use_softmax();
    actor.init(seed);

    Net critic;
    critic.add_layer(ntickers + ntickers, ntickers + 0);
    critic.add_layer(ntickers + 0, ntickers + 0);
    critic.add_layer(ntickers + 0, ntickers + 0);
    critic.add_layer(ntickers + 0, 1);
    critic.init(seed);

    DDPG ddpg(actor, critic, path, tickers, checkpoint);

    if (mode == "build") ddpg.build();

    return 0;
}
