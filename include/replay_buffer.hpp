#pragma once

#include <vector>
#include <random>

struct Experience {
    std::vector<double> state;
    int action;
    double reward;
    std::vector<double> next_state;
    bool done;
};

class ReplayBuffer {
    public:
        explicit ReplayBuffer(size_t capacity);
        void push(const Experience& experience);
        std::vector<Experience> sample(size_t batch_size);
        size_t size() const;
        bool is_ready(size_t batch_size) const;

    private:
        std::vector<Experience> memory;
        size_t capacity;
        size_t position;
        size_t current_size;
        std::mt19937 gen;
};
