#pragma once
#include <vector>
#include <memory>

class Layer {
    public:
        virtual std::vector<double> forward(const std::vector<double>& input) = 0;
        virtual std::vector<double> backward(const std::vector<double>& grad_output, double learning_rate) = 0;

        virtual std::unique_ptr<Layer> clone() const = 0;

        virtual ~Layer() = default;
};