#pragma once
#include <vector>
#include <random>

namespace Utils {
    /**
     * @brief Computes the sigmoid activation function.
     * @param x The input value.
     * @return The result of the sigmoid function.
     */
    double sigmoid(double x);

    /**
     * @brief Computes the derivative of the sigmoid function.
     * @param x The input value.
     * @return The derivative of the sigmoid function.
     */
    double sigmoid_derivative(double x);

    /**
     * @brief Initializes the random seed.
     * @note This function is now obsolete as the random number generator is self-seeding.
     */
    void initialize_random_seed();

    /**
     * @brief Generates a random weight between -1.0 and 1.0.
     * @return A random double value.
     */
    double random_weight();

    /**
     * @brief Computes the Rectified Linear Unit (ReLU) activation function.
     * @param x The input value.
     * @return The result of the ReLU function.
     */
    double relu(double x);

    /**
     * @brief Computes the derivative of the ReLU function.
     * @param x The input value.
     * @return The derivative of the ReLU function.
     */
    double relu_derivative(double x);

    /**
     * @brief Computes the Leaky ReLU activation function.
     * @param x The input value.
     * @param alpha The slope for negative values.
     * @return The result of the Leaky ReLU function.
     */
    double leaky_relu(double x, double alpha = 0.01);

    /**
     * @brief Computes the derivative of the Leaky ReLU function.
     * @param x The input value.
     * @param alpha The slope for negative values.
     * @return The derivative of the Leaky ReLU function.
     */
    double leaky_relu_derivative(double x, double alpha = 0.01);

    /**
     * @brief Computes the hyperbolic tangent (tanh) activation function.
     * @param x The input value.
     * @return The result of the tanh function.
     */
    double tanh(double x);

    /**
     * @brief Computes the derivative of the tanh function.
     * @param x The input value.
     * @return The derivative of the tanh function.
     */
    double tanh_derivative(double x);

    /**
     * @brief Computes the softmax function for a vector of values.
     * @param input The vector of values to be transformed.
     */
    void softmax(std::vector<double>& input);
}