#include "utils.hpp"
#include <cmath>
#include <cstdlib>
#include <ctime>

namespace Utils {
    double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    double sigmoid_derivative(double x) {
        double s = sigmoid(x);
        return s * (1.0 - s);
    }

    void initialize_random_seed() {
        std::srand(static_cast<unsigned>(std::time(nullptr)));
    }

    double random_weight() {
        return ((double) std::rand() / RAND_MAX) * 2.0 - 1.0;
    }

    double relu(double x) {
        return (x > 0) ? x : 0;
    }

    double relu_derivative(double x) {
        return (x > 0) ? 1.0 : 0.0;
    }

    double tanh(double x) {
        return std::tanh(x);
    }

    double tanh_derivative(double x) {
        double t = tanh(x);
        return 1.0 - t * t;
    }
}

