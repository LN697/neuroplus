#include "neuralnet.hpp"
#include "dense.hpp"
#include "activation.hpp"
#include "optimizer.hpp"
#include "utils.hpp"
#include <iostream>
#include <memory>

int main() {
    Utils::initialize_random_seed();
    
    NeuralNet network;
    
    auto dense1 = std::make_shared<Dense>(2, 5);
    auto activation1 = std::make_shared<Activation>(Utils::relu, Utils::relu_derivative);
    auto dense2 = std::make_shared<Dense>(5, 1);
    auto activation2 = std::make_shared<Activation>(Utils::sigmoid, Utils::sigmoid_derivative);
    
    // Using SGD with momentum
    dense1->setOptimizer(std::make_unique<SGD>(0.01, 0.9));
    dense2->setOptimizer(std::make_unique<SGD>(0.01, 0.9));
    
    // Using Adam optimizer
    // dense1->setOptimizer(std::make_unique<Adam>(0.001));
    // dense2->setOptimizer(std::make_unique<Adam>(0.001));
    
    network.addLayer(dense1);
    network.addLayer(activation1);
    network.addLayer(dense2);
    network.addLayer(activation2);
    
    network.setLoss(std::make_shared<MSELoss>());
    
    std::vector<std::vector<double>> inputs = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}
    };
    std::vector<std::vector<double>> targets = {
        {0}, {1}, {1}, {0}
    };
    
    std::cout << "Training neural network with SGD optimizer..." << std::endl;
    network.train(inputs, targets, 5000, 0.01);
    
    std::cout << "\nTesting results:" << std::endl;
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto output = network.predict(inputs[i]);
        std::cout << "Input: [" << inputs[i][0] << ", " << inputs[i][1] 
                  << "] -> Output: " << output[0] 
                  << " (Target: " << targets[i][0] << ")" << std::endl;
    }
    
    return 0;
}
