#pragma once
#include "layer.hpp"

class Dense : public Layer {
    private:
        std::vector<std::vector<double>> weights;
        std::vector<double> biases;
        std::vector<double> input_cache;

    public:
        Dense(int input_size, int output_size);
        std::vector<double> forward(const std::vector<double>& input) override;
        std::vector<double> backward(const std::vector<double>& grad_output, double learning_rate) override;
};