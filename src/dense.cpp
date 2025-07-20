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

void Dense::setOptimizer(std::unique_ptr<Optimizer> optimizer) {
    weight_optimizer = optimizer->clone();
    bias_optimizer = optimizer->clone();
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
        }
    }
    
    if (weight_optimizer && bias_optimizer) {
        std::vector<double> weight_gradients;
        for (size_t i = 0; i < weights.size(); ++i) {
            for (size_t j = 0; j < weights[i].size(); ++j) {
                weight_gradients.push_back(grad_output[i] * input_cache[j]);
            }
        }
        
        std::vector<double> flat_weights;
        for (const auto& row : weights) {
            for (double w : row) {
                flat_weights.push_back(w);
            }
        }
        
        weight_optimizer->update(flat_weights, weight_gradients);
        
        size_t idx = 0;
        for (size_t i = 0; i < weights.size(); ++i) {
            for (size_t j = 0; j < weights[i].size(); ++j) {
                weights[i][j] = flat_weights[idx++];
            }
        }
        
        std::vector<double> bias_gradients = grad_output;
        bias_optimizer->update(biases, bias_gradients);
    } else {
        for (size_t i = 0; i < weights.size(); ++i) {
            for (size_t j = 0; j < input_cache.size(); ++j) {
                weights[i][j] -= learning_rate * grad_output[i] * input_cache[j];
            }
        }
        for (size_t i = 0; i < biases.size(); ++i) {
            biases[i] -= learning_rate * grad_output[i];
        }
    }
    
    return grad_input;
}

std::unique_ptr<Layer> Dense::clone() const {
    auto cloned = std::make_unique<Dense>(input_cache.size(), weights.size());
    cloned->weights = weights;
    cloned->biases = biases;
    cloned->input_cache = input_cache;
    if (weight_optimizer) {
        std::unique_ptr<Optimizer> optimizer = weight_optimizer->clone();
        cloned->setOptimizer(std::move(optimizer));
    }
    
    return cloned;
}