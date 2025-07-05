#include "neuralnet.hpp"
#include <iostream>

void NeuralNet::addLayer(std::shared_ptr<Layer> layer) {
    layers.push_back(layer);
}

void NeuralNet::setLoss(std::shared_ptr<Loss> loss) {
    loss_function = loss;
}

std::vector<double> NeuralNet::predict(const std::vector<double>& input) {
    std::vector<double> output = input;
    for (auto& layer : layers) {
        output = layer->forward(output);
    }

    return output;
}

void NeuralNet::train(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& targets, int epochs, double learning_rate) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            std::vector<double> output = predict(inputs[i]);

            double loss = loss_function->compute(output, targets[i]);
            total_loss += loss;
            
            std::vector<double> grad = loss_function->gradient(output, targets[i]);
            for (int j = layers.size() - 1; j >= 0; --j) {
                grad = layers[j]->backward(grad, learning_rate);
            }
        }
        
        if (epoch % 1000 == 0) {
            std::cout << "Epoch " << epoch << ", Average Loss: " << total_loss / inputs.size() << std::endl;
        }
    }
}

NeuralNet::NeuralNet(const NeuralNet& other) {
    for (const auto& layer : other.layers) {
        layers.push_back(std::shared_ptr<Layer>(layer->clone()));
    }
    if (other.loss_function) {
        loss_function = std::shared_ptr<Loss>(other.loss_function->clone());
    } else {
        loss_function = nullptr;
    }
}

void NeuralNet::save(const std::string& filename) const {
    // Not implemented in this example
}

void NeuralNet::load(const std::string& filename) {
    // Not implemented in this example
}