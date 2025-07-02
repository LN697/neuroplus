#include "activation.hpp"

Activation::Activation(std::function<double(double)> act, std::function<double(double)> act_deriv)
    : activation(act), activation_derivative(act_deriv) {}

std::vector<double> Activation::forward(const std::vector<double>& input) {
    input_cache = input;
    std::vector<double> output(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = activation(input[i]);
    }

    return output;
}

std::vector<double> Activation::backward(const std::vector<double>& grad_output, double learning_rate) {
    std::vector<double> grad_input(grad_output.size());
    for (size_t i = 0; i < grad_output.size(); ++i) {
        grad_input[i] = activation_derivative(input_cache[i]) * grad_output[i];
    }

    return grad_input;
}
