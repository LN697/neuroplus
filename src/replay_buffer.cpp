#include "replay_buffer.hpp"
#include <stdexcept>
#include <algorithm>

ReplayBuffer::ReplayBuffer(size_t capacity)
    : capacity(capacity), position(0), current_size(0), gen(std::random_device{}()) {
    memory.reserve(capacity);
}

void ReplayBuffer::push(const Experience& experience) {
    if (current_size < capacity) {
        memory.push_back(experience);
        current_size++;
    } else {
        memory[position] = experience;
    }
    position = (position + 1) % capacity;
}

std::vector<Experience> ReplayBuffer::sample(size_t batch_size) {
    if (current_size < batch_size) {
        throw std::runtime_error("Not enough experiences in memory to sample a batch.");
    }

    std::vector<Experience> batch;
    batch.reserve(batch_size);

    std::sample(memory.begin(), memory.begin() + current_size, std::back_inserter(batch),
                batch_size, gen);

    return batch;
}

size_t ReplayBuffer::size() const {
    return current_size;
}

bool ReplayBuffer::is_ready(size_t batch_size) const {
    return current_size >= batch_size;
}
