#include "utils.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <numeric>

namespace Utils {
    namespace {
        // Use a thread-local random engine and distribution for better performance and thread safety
        thread_local std::mt19937 gen(std::random_device{}());
        thread_local std::uniform_real_distribution<double> dist(-1.0, 1.0);
    }

    double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    double sigmoid_derivative(double x) {
        double s = sigmoid(x);
        return s * (1.0 - s);
    }

    void initialize_random_seed() {
        // This function is now obsolete but kept for API compatibility.
        // The random number generator is now self-seeding.
    }

    double random_weight() {
        return dist(gen);
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

    double leaky_relu(double x, double alpha) {
        return (x > 0) ? x : alpha * x;
    }

    double leaky_relu_derivative(double x, double alpha) {
        return (x > 0) ? 1.0 : alpha;
    }

    void softmax(std::vector<double>& input) {
        if (input.empty()) {
            return;
        }

        double max_val = *std::max_element(input.begin(), input.end());
        double sum_exp = 0.0;

        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = std::exp(input[i] - max_val);
            sum_exp += input[i];
        }

        if (sum_exp > 0.0) {
            for (size_t i = 0; i < input.size(); ++i) {
                input[i] /= sum_exp;
            }
        }
    }
}

