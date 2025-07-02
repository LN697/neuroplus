#pragma once
#include <vector>

class Optimizer {
    public:
        virtual void update(std::vector<double>& weights, const std::vector<double>& gradients) = 0;
        virtual ~Optimizer() = default;
};