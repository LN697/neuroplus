#pragma once
#include <vector>
#include <random>

namespace Utils {
    double sigmoid(double x);
    double sigmoid_derivative(double x);
    void initialize_random_seed();
    double random_weight();
    double relu(double x);
    double relu_derivative(double x);
    double tanh(double x);
    double tanh_derivative(double x);
}