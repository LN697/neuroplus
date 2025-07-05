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

    net.addLayer(std::make_unique<Dense>(2, 10));
    net.addLayer(std::make_unique<Activation>(Utils::relu, Utils::relu_derivative));
    net.addLayer(std::make_unique<Dense>(10, 1));
    net.addLayer(std::make_unique<Activation>(Utils::sigmoid, Utils::sigmoid_derivative));

    net.setLoss(std::make_unique<MSELoss>());

    std::vector<std::vector<double>> X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<double>> Y = {{0}, {1}, {1}, {0}};

    net.train(X, Y, 10000, 0.1);

    NeuralNet netcpy(net);

    for (const auto& input : X) {
        auto output = netcpy.predict(input);
        std::cout << "Input: (" << input[0] << ", " << input[1] << ") => Output: " << output[0] << std::endl;
    }

    return 0;
}