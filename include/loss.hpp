#pragma once
#include <vector>
#include <memory>

/**
 * @brief Base abstract class for loss functions.
 * 
 * This abstract class defines the interface that all loss function implementations must follow.
 * It includes methods for computing the loss and its gradient.
 */
class Loss {
    public:
        /**
         * @brief Computes the loss between predicted and actual values.
         * 
         * @param predicted The predicted output from the network.
         * @param actual The target (ground truth) output.
         * @return The scalar loss value.
         */
        virtual double compute(const std::vector<double>& predicted, const std::vector<double>& actual) = 0;
        
        /**
         * @brief Computes the gradient of the loss with respect to predicted values.
         * 
         * @param predicted The predicted output from the network.
         * @param actual The target (ground truth) output.
         * @return The gradient vector.
         */
        virtual std::vector<double> gradient(const std::vector<double>& predicted, const std::vector<double>& actual) = 0;

        /**
         * @brief Creates a deep copy of this loss function.
         * 
         * @return A unique pointer to a new instance of this loss function.
         */
        virtual std::unique_ptr<Loss> clone() const = 0;

        /**
         * @brief Virtual destructor for proper cleanup in derived classes.
         */
        virtual ~Loss() = default;
};

#include <memory>

/**
 * @brief Mean Squared Error (MSE) loss function.
 * 
 * The MSE loss computes the average squared difference between predicted and actual values.
 * It is commonly used for regression problems.
 */
class MSELoss : public Loss {
    public:
        /**
         * @brief Computes the MSE loss.
         * 
         * @param predicted The predicted output from the network.
         * @param actual The target (ground truth) output.
         * @return The MSE loss value.
         */
        double compute(const std::vector<double>& predicted, const std::vector<double>& actual) override;
        
        /**
         * @brief Computes the gradient of the MSE loss.
         * 
         * @param predicted The predicted output from the network.
         * @param actual The target (ground truth) output.
         * @return The gradient vector.
         */
        std::vector<double> gradient(const std::vector<double>& predicted, const std::vector<double>& actual) override;
        
        /**
         * @brief Creates a deep copy of this loss function.
         * 
         * @return A unique pointer to a new instance of this loss function.
         */
        std::unique_ptr<Loss> clone() const override;
};