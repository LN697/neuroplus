#pragma once
#include <vector>
#include <memory>

class Loss {
    public:
        virtual double compute(const std::vector<double>& predicted, const std::vector<double>& actual) = 0;
        virtual std::vector<double> gradient(const std::vector<double>& predicted, const std::vector<double>& actual) = 0;

        virtual std::unique_ptr<Loss> clone() const = 0;

        virtual ~Loss() = default;
};

#include <memory>

class MSELoss : public Loss {
    public:
        double compute(const std::vector<double>& predicted, const std::vector<double>& actual) override;
        std::vector<double> gradient(const std::vector<double>& predicted, const std::vector<double>& actual) override;
        std::unique_ptr<Loss> clone() const override;
};