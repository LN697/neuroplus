#pragma once
#include "layer.hpp"
#include "optimizer.hpp"
#include <memory>

class Dense : public Layer {
    private:
        std::vector<std::vector<double>> weights;
        std::vector<double> biases;
        std::vector<double> input_cache;
        std::unique_ptr<Optimizer> weight_optimizer;
        std::unique_ptr<Optimizer> bias_optimizer;

    public:
        Dense(int input_size, int output_size);
        void setOptimizer(std::unique_ptr<Optimizer> optimizer);
        std::vector<double> forward(const std::vector<double>& input) override;
        std::vector<double> backward(const std::vector<double>& grad_output, double learning_rate) override;
        std::unique_ptr<Layer> clone() const override;
};