#include <cstdlib>
#include <vector>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <thread>
#include <algorithm>
#include "../lib/param.hpp"

#include "../lib/ddpg.hpp"

std::vector<double> *Memory::state() { return &s0; }
std::vector<double> *Memory::action() { return &a; }
std::vector<double> *Memory::next_state() { return &s1; }
double Memory::reward() { return r; }

std::vector<double> DDPG::epsilon_greedy(std::vector<double> &state, double eps) {
    double explore = (double)rand() / RAND_MAX;
    return actor->forward(state, explore < eps);
}

void DDPG::optimize_critic(std::vector<double> &state_action, double q, double optimal, std::vector<double> &agrad, std::vector<bool> &flag, double alpha, double lambda) {
    for(int l = critic->num_of_layers() - 1; l >= 0; l--) {
        double part = 0.00, grad = 0.00;
        for(unsigned int n = 0; n < critic->layer(l)->out_features(); n++) {
            if(l == critic->num_of_layers() - 1) part = -2.00 * (optimal - q);
            else part = critic->layer(l)->node(n)->err() * drelu(critic->layer(l)->node(n)->sum());

            double updated_bias = critic->layer(l)->node(n)->bias() - alpha * part;
            critic->layer(l)->node(n)->set_bias(updated_bias);

            for(unsigned int i = 0; i < critic->layer(l)->in_features(); i++) {
                if(l == 0) {
                    grad = part * state_action[i];
                    if(i < agrad.size()) {
                        agrad[i] = part * critic->layer(l)->node(n)->weight(i);
                        flag[i] = true;
                    }
                }
                else {
                    grad = part * critic->layer(l-1)->node(i)->act();
                    critic->layer(l-1)->node(i)->add_err(part * critic->layer(l)->node(n)->weight(i));
                }

                grad += lambda * critic->layer(l)->node(n)->weight(i);

                double updated_weight = critic->layer(l)->node(n)->weight(i) - alpha * grad;
                critic->layer(l)->node(n)->set_weight(i, updated_weight);
            }
        }
    }
}

void DDPG::optimize_actor(std::vector<double> &state, std::vector<double> &action, std::vector<double> &agrad, std::vector<bool> &flag, double alpha, double lambda) {
    for(int l = actor->num_of_layers() - 1; l >= 0; l--) {
        double part = 0.00, grad = 0.00;
        for(unsigned int n = 0; n < actor->layer(l)->out_features(); n++) {
            if(l == actor->num_of_layers() - 1) {
                while(!flag[n]) {}
                part = agrad[n] * action[n] * (1.00 - action[n]);
            }
            else part = actor->layer(l)->node(n)->err() * drelu(actor->layer(l)->node(n)->sum());

            double updated_bias = actor->layer(l)->node(n)->bias() + alpha * part;
            actor->layer(l)->node(n)->set_bias(updated_bias);

            for(unsigned int i = 0; i < actor->layer(l)->in_features(); i++) {
                if(l == 0) grad = part * state[i];
                else {
                    grad = part * actor->layer(l-1)->node(i)->act();
                    actor->layer(l-1)->node(i)->add_err(part * actor->layer(l)->node(n)->weight(i));
                }

                grad += lambda * actor->layer(l)->node(n)->weight(i);

                double updated_weight = actor->layer(l)->node(n)->weight(i) + alpha * grad;
                actor->layer(l)->node(n)->set_weight(i, updated_weight);
            }
        }
    }
}

double DDPG::optimize(Memory &memory, double gamma, double alpha, double lambda) {
    std::vector<double> *state = memory.state();
    std::vector<double> *action = memory.action();

    std::vector<double> state_action;
    state_action.insert(state_action.end(), action->begin(), action->end());
    state_action.insert(state_action.end(), state->begin(), state->end());

    std::vector<double> *next_state = memory.next_state();
    std::vector<double> next_state_action = target_actor.forward(*next_state, false);
    next_state_action.insert(next_state_action.end(), next_state->begin(), next_state->end());

    std::vector<double> q = critic->forward(state_action, false);
    std::vector<double> future = target_critic.forward(next_state_action, false);
    double optimal = memory.reward() + gamma * future[0];

    std::vector<double> agrad(action->size(), 0.00);
    std::vector<bool> flag(action->size(), false);

    std::thread critic_optimizer(&DDPG::optimize_critic, this, std::ref(state_action),
                                 q[0], optimal, std::ref(agrad), std::ref(flag), alpha, lambda);
    std::thread actor_optimizer(&DDPG::optimize_actor, this, std::ref(*state),
                                std::ref(*action), std::ref(agrad), std::ref(flag), alpha, lambda);

    critic_optimizer.join();
    actor_optimizer.join();

    return q[0];
}

void DDPG::sync(double tau) {
    copy(*actor, target_actor, tau);
    copy(*critic, target_critic, tau);
}



void DDPG::save(const std::string &path) {
    std::cout << "SAVING TO " << path << "\n";
    std::string actor_path = path + "_actor";
    std::string critic_path = path + "_critic";
    std::string target_actor_path = path + "_target_actor";
    std::string target_critic_path = path + "_target_critic";

    actor->save(actor_path);
    critic->save(critic_path);
    target_actor.save(target_actor_path);
    target_critic.save(target_critic_path);
}

void DDPG::load(const std::string &path) {
    std::string actor_path = path + "_actor";
    std::string critic_path = path + "_critic";
    std::string target_actor_path = path + "_target_actor";
    std::string target_critic_path = path + "_target_critic";

    actor->load(actor_path);
    critic->load(critic_path);
    target_actor.load(target_actor_path);
    target_critic.load(target_critic_path);
}






void DDPG::write() {
    std::ofstream out;
    std::cout << "WRITING" << "\n";
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

void DDPG::clean() {
    std::vector<std::vector<double>>().swap(path);
    std::vector<Memory>().swap(memory);
    std::vector<double>().swap(mean_reward);
    std::vector<double>().swap(test);
    std::vector<std::vector<double>>().swap(test_action);
}


double decay(double alpha_init, double t, double size, double itr, double k) {return alpha_init * std::exp(double(t) / (size * itr) * k);}

std::vector<double> DDPG::sample_state(unsigned int t) {
    std::vector<double> state(ntickers);
    for(unsigned int i = 0; i < ntickers; i++)
        state[i] = (path[i][t] - path[i][t-OBS]) / path[i][t-OBS];
    return state;
}


void DDPG::build() {
    std::default_random_engine seed(std::chrono::system_clock::now().time_since_epoch().count());
    std::cout << "Building..." << "\n";
    double eps = EPS_INIT;
    double alpha = ALPHA_INIT;

    for(unsigned int itr = 0; itr < ITR; itr++) {
        unsigned int update_count = 0;
        double reward_sum = 0.00, q_sum = 0.00;

        for(unsigned int t = OBS; t < ext; t++) {
            if((itr+1)*t > OBS && eps > EPS_MIN)
                eps += (EPS_MIN - EPS_INIT) / CAPACITY;

            std::vector<double> state = sample_state(t);
            std::vector<double> action = epsilon_greedy(state, eps);
            std::vector<double> next_state = sample_state(t+1);

            double reward = 0.00;
            for(unsigned int i = 0; i < ntickers; i++) {
                reward += path[i][t+1] * action[i];
            }
            reward = log10(reward);
            reward_sum += reward; 

            memory.push_back(Memory(state, action, next_state, reward));

            if(memory.size() == CAPACITY) {
                std::vector<unsigned int> index(CAPACITY, 0);
                std::iota(index.begin(), index.end(), 0);
                std::shuffle(index.begin(), index.end(), seed);
                index.erase(index.begin() + BATCH, index.end());

                alpha = decay(ALPHA_INIT, t, path[0].size(), ITR, K); 

                for(unsigned int &k: index)
                    q_sum += optimize(memory[k], GAMMA, alpha, LAMBDA);
                update_count += BATCH;

                memory.erase(memory.begin());
                std::vector<unsigned int>().swap(index);
            }
        }

        std::cout << reward_sum << "\n";
        mean_reward.push_back(reward_sum / (ext-1));

        std::cout << "ITR=" << itr << " ";
        std::cout << "MR=" << mean_reward.back() << " ";
        std::cout << "Q=" << q_sum / update_count << "\n";
    }

    test_action.resize(ntickers, std::vector<double>(ext-1));

    for(unsigned int t = OBS; t < ext; t++) {
        std::vector<double> state = sample_state(t);
        std::vector<double> action = epsilon_greedy(state, 0.00);

        double reward = 0.00;
        for(unsigned int i = 0; i < ntickers; i++) {
            if (i < path.size() && t + 1 < path[i].size()) {
                reward += path[i][t+1] * action[i];
                test_action[i][t-1] = action[i];
            } else {
                std::cerr << "Out of bounds access in path\n";
                std::exit(EXIT_FAILURE);
            }
        }
        test.push_back(reward);

        std::vector<double>().swap(state);
        std::vector<double>().swap(action);
    }

    write();
    clean();
    save(checkpoint);
}
