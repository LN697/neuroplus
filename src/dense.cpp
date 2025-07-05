#include "dense.hpp"
#include "utils.hpp"

Dense::Dense(int input_size, int output_size) {
    weights.resize(output_size, std::vector<double>(input_size));
    biases.resize(output_size);
    for (auto& row : weights) {
        for (double& w : row) {
            w = Utils::random_weight();
        }
    }
    for (double& b : biases) {
        b = Utils::random_weight();
    }
}

std::vector<double> Dense::forward(const std::vector<double>& input) {
    input_cache = input;
    std::vector<double> output(biases);
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[i].size(); ++j) {
            output[i] += weights[i][j] * input[j];
        }
    }

    return output;
}

std::vector<double> Dense::backward(const std::vector<double>& grad_output, double learning_rate) {
    std::vector<double> grad_input(input_cache.size(), 0.0);
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < input_cache.size(); ++j) {
            grad_input[j] += weights[i][j] * grad_output[i];
            weights[i][j] -= learning_rate * grad_output[i] * input_cache[j];
        }
    }
    for (size_t i = 0; i < biases.size(); ++i) {
        biases[i] -= learning_rate * grad_output[i];
    }
    
    return grad_input;
}

std::unique_ptr<Layer> Dense::clone() const {
    return std::make_unique<Dense>(*this);
}