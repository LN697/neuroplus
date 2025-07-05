#pragma once
#include <vector>
#include <memory>

class Optimizer {
    public:
        virtual void update(std::vector<double>& weights, const std::vector<double>& gradients) = 0;
        virtual std::unique_ptr<Optimizer> clone() const = 0;
        virtual ~Optimizer() = default;
};

class SGD : public Optimizer {
    private:
        double learning_rate;
        double momentum;
        std::vector<double> velocity;
        
    public:
        SGD(double lr, double mom = 0.0);
        void update(std::vector<double>& weights, const std::vector<double>& gradients) override;
        std::unique_ptr<Optimizer> clone() const override;
};

class Adam : public Optimizer {
    private:
        double learning_rate;
        double beta1;
        double beta2;
        double epsilon;
        std::vector<double> m;
        std::vector<double> v;
        int t;
        
    public:
        Adam(double lr = 0.001, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8);
        void update(std::vector<double>& weights, const std::vector<double>& gradients) override;
        std::unique_ptr<Optimizer> clone() const override;
};