#pragma once
#include "layer.hpp"
#include "optimizer.hpp"
#include <memory>

/**
 * @brief Fully-connected (dense) layer for neural networks.
 * 
 * This layer implements a fully-connected layer where each input is connected 
 * to each output through weights. Each output also has a bias term.
 */
class Dense : public Layer {
    public:
        /** @brief Deleted copy constructor */
        Dense(const Dense&) = delete;
        
        /** @brief Deleted copy assignment operator */
        Dense& operator=(const Dense&) = delete;
        
    private:
        /** @brief The weight matrix (output_size x input_size) */
        std::vector<std::vector<double>> weights;
        
        /** @brief The bias vector (output_size) */
        std::vector<double> biases;
        
        /** @brief Cache of input values for use in backward pass */
        std::vector<double> input_cache;
        
        /** @brief Optimizer for the weights */
        std::unique_ptr<Optimizer> weight_optimizer;
        
        /** @brief Optimizer for the biases */
        std::unique_ptr<Optimizer> bias_optimizer;

    public:
        /**
         * @brief Constructs a Dense layer with specified input and output sizes.
         * 
         * @param input_size The size of the input vector.
         * @param output_size The size of the output vector.
         */
        Dense(int input_size, int output_size);
        
        /**
         * @brief Sets the optimizer for this layer.
         * 
         * @param optimizer The optimizer to use for updating weights and biases.
         */
        void setOptimizer(std::unique_ptr<Optimizer> optimizer);
        
        /**
         * @brief Computes the forward pass through this layer.
         * 
         * @param input The input vector.
         * @return The output vector.
         */
        std::vector<double> forward(const std::vector<double>& input) override;
        
        /**
         * @brief Computes the backward pass through this layer.
         * 
         * @param grad_output The gradient from the next layer.
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