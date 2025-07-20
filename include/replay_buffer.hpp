#pragma once

#include <vector>
#include <random>

/**
 * @brief Structure representing a single experience for reinforcement learning.
 * 
 * This structure stores the state, action, reward, next state, and done flag
 * that constitute a transition in a reinforcement learning environment.
 */
struct Experience {
    /** @brief The state before taking the action */
    std::vector<double> state;
    
    /** @brief The action taken */
    int action;
    
    /** @brief The reward received after taking the action */
    double reward;
    
    /** @brief The state after taking the action */
    std::vector<double> next_state;
    
    /** @brief Flag indicating if the episode ended after this transition */
    bool done;
};

/**
 * @brief A circular buffer for storing experiences for experience replay.
 * 
 * This class implements a replay buffer that stores experiences and allows
 * random sampling from them for training reinforcement learning agents.
 */
class ReplayBuffer {
    public:
        /**
         * @brief Constructs a replay buffer with the specified capacity.
         * 
         * @param capacity The maximum number of experiences the buffer can hold.
         */
        explicit ReplayBuffer(size_t capacity);
        
        /**
         * @brief Adds an experience to the buffer.
         * 
         * If the buffer is full, the oldest experience will be overwritten.
         * 
         * @param experience The experience to add to the buffer.
         */
        void push(const Experience& experience);
        
        /**
         * @brief Samples a batch of experiences randomly from the buffer.
         * 
         * @param batch_size The number of experiences to sample.
         * @return A vector of sampled experiences.
         */
        std::vector<Experience> sample(size_t batch_size);
        
        /**
         * @brief Gets the current number of experiences in the buffer.
         * 
         * @return The number of experiences currently stored.
         */
        size_t size() const;
        
        /**
         * @brief Checks if the buffer has enough experiences for sampling.
         * 
         * @param batch_size The batch size to check against.
         * @return True if the buffer has at least batch_size experiences, false otherwise.
         */
        bool is_ready(size_t batch_size) const;

    private:
        /** @brief The storage for experiences */
        std::vector<Experience> memory;
        
        /** @brief The maximum capacity of the buffer */
        size_t capacity;
        
        /** @brief The current position in the buffer for adding new experiences */
        size_t position;
        
        /** @brief The current number of experiences in the buffer */
        size_t current_size;
        
        /** @brief Random number generator for sampling */
        std::mt19937 gen;
};
