#pragma once
#include <vector>

class Loss {
    public:
        virtual double compute(const std::vector<double>& predicted, const std::vector<double>& actual) = 0;
        virtual std::vector<double> gradient(const std::vector<double>& predicted, const std::vector<double>& actual) = 0;
        virtual ~Loss() = default;
};

class MSELoss : public Loss {
    public:
        double compute(const std::vector<double>& predicted, const std::vector<double>& actual) override;
        std::vector<double> gradient(const std::vector<double>& predicted, const std::vector<double>& actual) override;
};