#pragma once
#include <vector>
#include <memory>

/**
 * @brief Base abstract class for all neural network layers.
 * 
 * This abstract class defines the interface that all layer implementations must follow.
 * It includes methods for forward propagation, backward propagation, and cloning.
 */
class Layer {
    public:
        /**
         * @brief Performs forward propagation through the layer.
         * 
         * @param input The input vector to the layer.
         * @return The output vector from the layer.
         */
        virtual std::vector<double> forward(const std::vector<double>& input) = 0;
        
        /**
         * @brief Performs backward propagation through the layer.
         * 
         * @param grad_output The gradient from the next layer.
         * @param learning_rate The learning rate for parameter updates.
         * @return The gradient to pass to the previous layer.
         */
        virtual std::vector<double> backward(const std::vector<double>& grad_output, double learning_rate) = 0;

        /**
         * @brief Creates a deep copy of this layer.
         * 
         * @return A unique pointer to a new instance of this layer.
         */
        virtual std::unique_ptr<Layer> clone() const = 0;

        /**
         * @brief Virtual destructor for proper cleanup in derived classes.
         */
        virtual ~Layer() = default;
};