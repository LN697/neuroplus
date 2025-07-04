#include "neuralnet.hpp"
#include "utils.hpp"
#include "dense.hpp"
#include "activation.hpp"
#include "loss.hpp"
#include "optimizer.hpp"

#include <iostream>
#include <memory>

int main() {
    NeuralNet net;
    net.addLayer(std::make_unique<Dense>(2, 10));
    net.addLayer(std::make_unique<Activation>(Utils::tanh, Utils::tanh_derivative));
    net.addLayer(std::make_unique<Dense>(10, 1));
    net.addLayer(std::make_unique<Activation>(Utils::sigmoid, Utils::sigmoid_derivative));

    class MSELoss : public Loss {
        public:
            double compute(const std::vector<double>& predicted, const std::vector<double>& actual) override {
                double sum = 0.0;
                for (size_t i = 0; i < predicted.size(); ++i) {
                    sum += std::pow(predicted[i] - actual[i], 2);
                }

                return sum / predicted.size();
            }

            std::vector<double> gradient(const std::vector<double>& predicted, const std::vector<double>& actual) override {
                std::vector<double> grad(predicted.size());
                for (size_t i = 0; i < predicted.size(); ++i) {
                    grad[i] = 2 * (predicted[i] - actual[i]) / predicted.size();
                }

                return grad;
            }
    };

    net.setLoss(std::make_unique<MSELoss>());

    std::vector<std::vector<double>> X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<double>> Y = {{0}, {1}, {1}, {0}};

    net.train(X, Y, 10000, 0.1);

    for (const auto& input : X) {
        auto output = net.predict(input);
        std::cout << "Input: (" << input[0] << ", " << input[1] << ") => Output: " << output[0] << std::endl;
    }

    return 0;
}