#include "loss.hpp"
#include <cmath>
#include <stdexcept>

double MSELoss::compute(const std::vector<double>& predicted, const std::vector<double>& actual) {
    if (predicted.size() != actual.size()) {
        throw std::invalid_argument("Predicted and actual vectors must have the same size");
    }
    
    double mse = 0.0;
    for (size_t i = 0; i < predicted.size(); ++i) {
        double diff = predicted[i] - actual[i];
        mse += diff * diff;
    }
    
    return mse / predicted.size();
}

std::vector<double> MSELoss::gradient(const std::vector<double>& predicted, const std::vector<double>& actual) {
    if (predicted.size() != actual.size()) {
        throw std::invalid_argument("Predicted and actual vectors must have the same size");
    }
    
    std::vector<double> grad(predicted.size());
    for (size_t i = 0; i < predicted.size(); ++i) {
        grad[i] = 2.0 * (predicted[i] - actual[i]) / predicted.size();
    }
    
    return grad;
}
