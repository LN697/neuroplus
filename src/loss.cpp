#include "loss.hpp"
#include <cmath>

class MSELoss : public Loss {
    public:
        double compute(const std::vector<double>& predicted, const std::vector<double>& actual) override {
            double sum = 0.0;
            for (size_t i = 0; i < predicted.size(); ++i) {
                sum += std::pow(predicted[i] - actual[i], 2);
            }

            return sum / predicted.size();
        }

        std::vector<double> gradient(const std::vector<double>& predicted, const std::vector<double>& actual) override {
            std::vector<double> grad(predicted.size());
            for (size_t i = 0; i < predicted.size(); ++i) {
                grad[i] = 2 * (predicted[i] - actual[i]) / predicted.size();
            }

            return grad;
        }
};
