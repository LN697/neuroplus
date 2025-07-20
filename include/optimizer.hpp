#pragma once
#include <vector>
#include <memory>

/**
 * @brief Base abstract class for optimization algorithms.
 * 
 * This abstract class defines the interface that all optimizer implementations must follow.
 * Optimizers are used to update weights based on gradients during training.
 */
class Optimizer {
    public:
        /**
         * @brief Updates weights based on gradients.
         * 
         * @param weights The weights to update.
         * @param gradients The gradients used for the update.
         */
        virtual void update(std::vector<double>& weights, const std::vector<double>& gradients) = 0;
        
        /**
         * @brief Creates a deep copy of this optimizer.
         * 
         * @return A unique pointer to a new instance of this optimizer.
         */
        virtual std::unique_ptr<Optimizer> clone() const = 0;
        
        /**
         * @brief Virtual destructor for proper cleanup in derived classes.
         */
        virtual ~Optimizer() = default;
};

/**
 * @brief Stochastic Gradient Descent optimizer with momentum.
 * 
 * This optimizer implements the standard SGD algorithm with momentum support.
 */
class SGD : public Optimizer {
    private:
        /** @brief The learning rate */
        double learning_rate;
        
        /** @brief The momentum coefficient */
        double momentum;
        
        /** @brief The velocity vector used for momentum */
        std::vector<double> velocity;
        
    public:
        /**
         * @brief Constructs an SGD optimizer.
         * 
         * @param lr The learning rate.
         * @param mom The momentum coefficient.
         */
        SGD(double lr, double mom = 0.0);
        
        /**
         * @brief Updates weights using SGD with momentum.
         * 
         * @param weights The weights to update.
         * @param gradients The gradients used for the update.
         */
        void update(std::vector<double>& weights, const std::vector<double>& gradients) override;
        
        /**
         * @brief Creates a deep copy of this optimizer.
         * 
         * @return A unique pointer to a new instance of this optimizer.
         */
        std::unique_ptr<Optimizer> clone() const override;
};

/**
 * @brief Adam optimizer.
 * 
 * This optimizer implements the Adam algorithm, which maintains per-parameter
 * learning rates based on first and second moment estimates of the gradients.
 */
class Adam : public Optimizer {
    private:
        /** @brief The learning rate */
        double learning_rate;
        
        /** @brief The exponential decay rate for the first moment estimates */
        double beta1;
        
        /** @brief The exponential decay rate for the second moment estimates */
        double beta2;
        
        /** @brief A small constant for numerical stability */
        double epsilon;
        
        /** @brief First moment vector */
        std::vector<double> m;
        
        /** @brief Second moment vector */
        std::vector<double> v;
        
        /** @brief Timestep counter */
        int t;
        
    public:
        /**
         * @brief Constructs an Adam optimizer.
         * 
         * @param lr The learning rate.
         * @param b1 The beta1 coefficient for first moment estimates.
         * @param b2 The beta2 coefficient for second moment estimates.
         * @param eps The epsilon value for numerical stability.
         */
        Adam(double lr = 0.001, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8);
        
        /**
         * @brief Updates weights using the Adam algorithm.
         * 
         * @param weights The weights to update.
         * @param gradients The gradients used for the update.
         */
        void update(std::vector<double>& weights, const std::vector<double>& gradients) override;
        
        /**
         * @brief Creates a deep copy of this optimizer.
         * 
         * @return A unique pointer to a new instance of this optimizer.
         */
        std::unique_ptr<Optimizer> clone() const override;
};