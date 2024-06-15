#ifndef __QUANT_HPP_
#define __QUANT_HPP_

#include <cstdlib>
#include <vector>

#include "../lib/net.hpp"

class Memory
{
private:
    std::vector<double> s0;
    std::vector<double> a;
    std::vector<double> s1;
    double r;
public:
    Memory() {}
    Memory(std::vector<double> &current, std::vector<double> &action, std::vector<double> &next, double reward) {
        s0.swap(current);
        a.swap(action);
        s1.swap(next);
        r = reward;
    }
    ~Memory() {
        std::vector<double>().swap(s0);
        std::vector<double>().swap(a);
        std::vector<double>().swap(s1);
    }

    std::vector<double> *state();
    std::vector<double> *action();
    std::vector<double> *next_state();
    double reward();
};

class DDPG
{
private:
    Net *actor, target_actor;
    Net *critic, target_critic;

    std::string checkpoint;

    std::vector<std::vector<double>> path;

    std::vector<Memory> memory;

    std::vector<double> mean_reward;
    std::vector<double> test;
    std::vector<std::vector<double>> test_action;
    int ntickers;
    int ext;
    std::vector<std::string> tickers;


public:
    DDPG() {}
    DDPG(Net &a, Net &c) {
        actor = &a;
        critic = &c;
        copy(*actor, target_actor, 1.00);
        copy(*critic, target_critic, 1.00);
    }
    DDPG(Net &a, Net &c, std::vector<std::vector<double>> &pathIN, std::vector<std::string> &tickersIN, std::string checkpointpath) {
        path.swap(pathIN);  tickers.swap(tickersIN); checkpoint = checkpointpath; ntickers = tickers.size(); ext = path[0].size();
        actor = &a;
        critic = &c;
        copy(*actor, target_actor, 1.00);
        copy(*critic, target_critic, 1.00);
    }
    ~DDPG() {}

    std::vector<double> epsilon_greedy(std::vector<double> &state, double eps);

    void optimize_critic(std::vector<double> &state_action, double q, double optimal, std::vector<double> &agrad, std::vector<bool> &flag, double alpha, double lambda);
    void optimize_actor(std::vector<double> &state, std::vector<double> &action, std::vector<double> &agrad, std::vector<bool> &flag, double alpha, double lambda);
    double optimize(Memory &memory, double gamma, double alpha, double lambda);

    std::vector<double> sample_state(unsigned int t);
    void write();
    void clean();



    void sync(double tau);

    void build();

    void save(const std::string &path);
    void load(const std::string &path);
};

#endif