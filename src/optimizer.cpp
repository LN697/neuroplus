#include "optimizer.hpp"
#include <cmath>
#include <algorithm>

SGD::SGD(double lr, double mom) 
    : learning_rate(lr), momentum(mom) {}

void SGD::update(std::vector<double>& weights, const std::vector<double>& gradients) {
    if (velocity.size() != weights.size()) {
        velocity.resize(weights.size(), 0.0);
    }
    
    for (size_t i = 0; i < weights.size(); ++i) {
        velocity[i] = momentum * velocity[i] - learning_rate * gradients[i];
        weights[i] += velocity[i];
    }
}

std::unique_ptr<Optimizer> SGD::clone() const {
    auto cloned = std::make_unique<SGD>(learning_rate, momentum);
    cloned->velocity = velocity;
    return cloned;
}

Adam::Adam(double lr, double b1, double b2, double eps)
    : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), t(0) {}

void Adam::update(std::vector<double>& weights, const std::vector<double>& gradients) {
    if (m.size() != weights.size()) {
        m.resize(weights.size(), 0.0);
        v.resize(weights.size(), 0.0);
    }
    
    t++;
    
    for (size_t i = 0; i < weights.size(); ++i) {
        m[i] = beta1 * m[i] + (1.0 - beta1) * gradients[i];
        
        v[i] = beta2 * v[i] + (1.0 - beta2) * gradients[i] * gradients[i];
        
        double m_hat = m[i] / (1.0 - std::pow(beta1, t));
        
        double v_hat = v[i] / (1.0 - std::pow(beta2, t));
        
        weights[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
    }
}

std::unique_ptr<Optimizer> Adam::clone() const {
    return std::make_unique<Adam>(*this);
}
