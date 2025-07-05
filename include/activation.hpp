#pragma once
#include "layer.hpp"
#include <functional>

class Activation : public Layer {
    private:
        std::function<double(double)> activation;
        std::function<double(double)> activation_derivative;
        std::vector<double> input_cache;

    public:
        Activation(std::function<double(double)> act, std::function<double(double)> act_deriv);
        std::vector<double> forward(const std::vector<double>& input) override;
        std::vector<double> backward(const std::vector<double>& grad_output, double learning_rate) override;
        std::unique_ptr<Layer> clone() const override;
};