#pragma once
#include "layer.hpp"
#include "loss.hpp"
#include <vector>
#include <memory>

class NeuralNet {
    private:
        std::vector<std::shared_ptr<Layer>> layers;
        std::shared_ptr<Loss> loss_function;
    
    public:
        void addLayer(std::shared_ptr<Layer> layer);
        void setLoss(std::shared_ptr<Loss> loss);
        std::vector<double> predict(const std::vector<double>& input);
        void train(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& targets, int epochs, double learning_rate);
};