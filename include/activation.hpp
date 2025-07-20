#pragma once
#include "layer.hpp"
#include <functional>

/**
 * @brief Activation layer that applies a non-linear activation function to inputs.
 * 
 * The Activation class implements various activation functions like sigmoid, ReLU, 
 * leaky ReLU, and tanh to introduce non-linearity into neural networks.
 */
class Activation : public Layer {
    private:
        /** @brief The activation function to be applied to inputs */
        std::function<double(double)> activation;
        
        /** @brief The derivative of the activation function used during backpropagation */
        std::function<double(double)> activation_derivative;
        
        /** @brief Cache of input values for use in backward pass */
        std::vector<double> input_cache;

    public:
        /**
         * @brief Constructs an Activation layer with the specified activation function and its derivative.
         * 
         * @param act The activation function.
         * @param act_deriv The derivative of the activation function.
         */
        Activation(std::function<double(double)> act, std::function<double(double)> act_deriv);
        
        /**
         * @brief Applies the activation function to the input.
         * 
         * @param input The input vector.
         * @return The activated output vector.
         */
        std::vector<double> forward(const std::vector<double>& input) override;
        
        /**
         * @brief Computes gradients during backpropagation.
         * 
         * @param grad_output Gradient from the next layer.
         * @param learning_rate The learning rate for parameter updates.
         * @return The gradient to pass to the previous layer.
         */
        std::vector<double> backward(const std::vector<double>& grad_output, double learning_rate) override;
        
        /**
         * @brief Creates a deep copy of this layer.
         * 
         * @return A unique pointer to a new instance of this layer.
         */
        std::unique_ptr<Layer> clone() const override;
};