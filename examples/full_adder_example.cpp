#include "neuralnet.hpp"
#include "utils.hpp"
#include "dense.hpp"
#include "activation.hpp"
#include "loss.hpp"
#include "optimizer.hpp"

#include <iostream>
#include <memory>

int main() {
    Utils::initialize_random_seed();
    
    NeuralNet net;

    auto dense1 = std::make_shared<Dense>(4, 5);
    auto activation1 = std::make_shared<Activation>(Utils::relu, Utils::relu_derivative);
    auto dense2 = std::make_shared<Dense>(5, 3);
    auto activation2 = std::make_shared<Activation>(Utils::sigmoid, Utils::sigmoid_derivative);
    
    // Using SGD with momentum
    dense1->setOptimizer(std::make_unique<SGD>(0.01, 0.9));
    dense2->setOptimizer(std::make_unique<SGD>(0.01, 0.9));
    
    // Using Adam optimizer
    // dense1->setOptimizer(std::make_unique<Adam>(0.001));
    // dense2->setOptimizer(std::make_unique<Adam>(0.001));
    
    net.addLayer(dense1);
    net.addLayer(activation1);
    net.addLayer(dense2);
    net.addLayer(activation2);
    
    net.setLoss(std::make_shared<MSELoss>());

    // Data for a 2-bit full adder.
    // Input X: [A1, A0, B1, B0]
    // Output Y: [Carry-Out, Sum1, Sum0]

    std::vector<std::vector<double>> X = {
        {0, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}, {0, 0, 1, 1},
        {0, 1, 0, 0}, {0, 1, 0, 1}, {0, 1, 1, 0}, {0, 1, 1, 1},
        {1, 0, 0, 0}, {1, 0, 0, 1}, {1, 0, 1, 0}, {1, 0, 1, 1},
        {1, 1, 0, 0}, {1, 1, 0, 1}, {1, 1, 1, 0}, {1, 1, 1, 1}
    };
    std::vector<std::vector<double>> Y = {
        {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1},
        {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {1, 0, 0},
        {0, 1, 0}, {0, 1, 1}, {1, 0, 0}, {1, 0, 1},
        {0, 1, 1}, {1, 0, 0}, {1, 0, 1}, {1, 1, 0}
    };

    net.train(X, Y, 10000, 0.1);

    // NeuralNet netcpy(net);

    for (const auto& input : X) {
        auto output = net.predict(input);
        std::cout << "Input: 0b" << input[0] << input[1] << " + 0b" << input[2] 
        << input[3] << " => Output: 0b" << static_cast<int>(output[0] + 0.5) << static_cast<int>(output[1] + 0.5) 
        << static_cast<int>(output[2] + 0.5) << std::endl;
    }

    // std::cout << "\nTesting copied network:" << std::endl;
    // for (const auto& input : X) {
    //     auto output = netcpy.predict(input);
    //     std::cout << "Input: 0b" << input[0] << input[1] << " + 0b" << input[2] 
    //     << input[3] << " => Output: 0b" << static_cast<int>(output[0] + 0.5) << static_cast<int>(output[1] + 0.5) 
    //     << static_cast<int>(output[2] + 0.5) << std::endl;
    // }

    return 0;
}