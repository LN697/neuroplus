#include "utils.hpp"
#include <cmath>
#include <cstdlib>

namespace Utils {
    double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    double sigmoid_derivative(double x) {
        double s = sigmoid(x);
        return s * (1.0 - s);
    }

    double random_weight() {
        return ((double) std::rand() / RAND_MAX) * 2.0 - 1.0;
    }
}

