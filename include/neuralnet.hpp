#pragma once
#include "layer.hpp"
#include "loss.hpp"
#include <vector>
#include <memory>

/**
 * @brief Main neural network class that manages layers and training.
 * 
 * This class represents a fully-connected neural network and provides methods
 * for building, training, and using the network for predictions.
 */
class NeuralNet {
    private:
        /** @brief The layers in the network */
        std::vector<std::shared_ptr<Layer>> layers;
        
        /** @brief The loss function used for training */
        std::shared_ptr<Loss> loss_function;
    
    public:
        /**
         * @brief Adds a layer to the network.
         * 
         * Layers are executed in the order they are added.
         * 
         * @param layer The layer to add to the network.
         */
        void addLayer(std::shared_ptr<Layer> layer);
        
        /**
         * @brief Sets the loss function for the network.
         * 
         * @param loss The loss function to use.
         */
        void setLoss(std::shared_ptr<Loss> loss);
        
        /**
         * @brief Makes a prediction using the network.
         * 
         * @param input The input vector.
         * @return The predicted output vector.
         */
        std::vector<double> predict(const std::vector<double>& input);
        
        /**
         * @brief Trains the network on the provided dataset.
         * 
         * @param inputs Vector of input vectors for training.
         * @param targets Vector of target (ground truth) vectors.
         * @param epochs Number of training epochs.
         * @param learning_rate Learning rate for gradient descent.
         */
        void train(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& targets, int epochs, double learning_rate);

        /**
         * @brief Default constructor.
         */
        NeuralNet() = default;
        
        /**
         * @brief Copy constructor for creating a deep copy of the network.
         * 
         * @param other The network to copy.
         */
        NeuralNet(const NeuralNet& other);

        /**
         * @brief Saves the network to a file.
         * 
         * @param filename The path to the file where the network will be saved.
         */
        void save(const std::string& filename) const;
        
        /**
         * @brief Loads the network from a file.
         * 
         * @param filename The path to the file containing the saved network.
         */
        void load(const std::string& filename);

        /**
         * @brief Default destructor.
         */
        ~NeuralNet() = default;
};