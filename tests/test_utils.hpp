#pragma once

#include "test_framework.hpp"
#include "../include/utils.hpp"
#include <vector>
#include <cmath>
#include <numeric>

/**
 * @brief Tests for Utils functionality
 * @return TestSuite with the results
 */
TestFramework::TestSuite runUtilsTests() {
    TestFramework::TestSuite suite("Utils");

    // Test sigmoid function
    suite.runTest("Sigmoid Function", []() {
        // Test sigmoid with various inputs
        TestFramework::assertDoubleEqual(0.5, Utils::sigmoid(0.0), 1e-10, "sigmoid(0.0) should be 0.5");
        TestFramework::assertDoubleEqual(0.7310585786300049, Utils::sigmoid(1.0), 1e-10, "sigmoid(1.0) incorrect");
        TestFramework::assertDoubleEqual(0.2689414213699951, Utils::sigmoid(-1.0), 1e-10, "sigmoid(-1.0) incorrect");
        TestFramework::assertDoubleEqual(0.9999546021312976, Utils::sigmoid(10.0), 1e-10, "sigmoid(10.0) incorrect");
        TestFramework::assertDoubleEqual(0.00004539786870243, Utils::sigmoid(-10.0), 1e-10, "sigmoid(-10.0) incorrect");
    });

    // Test sigmoid_derivative function
    suite.runTest("Sigmoid Derivative", []() {
        // Test sigmoid_derivative with various inputs
        TestFramework::assertDoubleEqual(0.25, Utils::sigmoid_derivative(0.0), 1e-10, "sigmoid_derivative(0.0) should be 0.25");
        TestFramework::assertDoubleEqual(0.19661193324148188, Utils::sigmoid_derivative(1.0), 1e-10, "sigmoid_derivative(1.0) incorrect");
        TestFramework::assertDoubleEqual(0.19661193324148188, Utils::sigmoid_derivative(-1.0), 1e-10, "sigmoid_derivative(-1.0) incorrect");
    });

    // Test random_weight function
    suite.runTest("Random Weight", []() {
        // Test random_weight returns values within expected range
        for (int i = 0; i < 1000; ++i) {
            double weight = Utils::random_weight();
            TestFramework::assertTrue(weight >= -1.0 && weight <= 1.0, 
                                     "random_weight() should return a value between -1.0 and 1.0");
        }
        
        // Test that multiple calls generate different values (with high probability)
        double weight1 = Utils::random_weight();
        double weight2 = Utils::random_weight();
        double weight3 = Utils::random_weight();
        bool all_different = (weight1 != weight2) || (weight2 != weight3) || (weight1 != weight3);
        TestFramework::assertTrue(all_different, "Multiple calls to random_weight() should generate different values");
    });

    // Test ReLU function
    suite.runTest("ReLU Function", []() {
        // Test ReLU with various inputs
        TestFramework::assertDoubleEqual(0.0, Utils::relu(0.0), 1e-10, "relu(0.0) should be 0.0");
        TestFramework::assertDoubleEqual(1.0, Utils::relu(1.0), 1e-10, "relu(1.0) should be 1.0");
        TestFramework::assertDoubleEqual(0.0, Utils::relu(-1.0), 1e-10, "relu(-1.0) should be 0.0");
        TestFramework::assertDoubleEqual(10.0, Utils::relu(10.0), 1e-10, "relu(10.0) should be 10.0");
        TestFramework::assertDoubleEqual(0.0, Utils::relu(-10.0), 1e-10, "relu(-10.0) should be 0.0");
    });

    // Test ReLU derivative function
    suite.runTest("ReLU Derivative", []() {
        // Test ReLU derivative with various inputs
        TestFramework::assertDoubleEqual(0.0, Utils::relu_derivative(-1.0), 1e-10, "relu_derivative(-1.0) should be 0.0");
        TestFramework::assertDoubleEqual(1.0, Utils::relu_derivative(1.0), 1e-10, "relu_derivative(1.0) should be 1.0");
        TestFramework::assertDoubleEqual(0.0, Utils::relu_derivative(-0.1), 1e-10, "relu_derivative(-0.1) should be 0.0");
        TestFramework::assertDoubleEqual(1.0, Utils::relu_derivative(0.1), 1e-10, "relu_derivative(0.1) should be 1.0");
    });

    // Test Leaky ReLU function
    suite.runTest("Leaky ReLU Function", []() {
        double alpha = 0.01;
        // Test Leaky ReLU with various inputs
        TestFramework::assertDoubleEqual(0.0, Utils::leaky_relu(0.0, alpha), 1e-10, "leaky_relu(0.0) should be 0.0");
        TestFramework::assertDoubleEqual(1.0, Utils::leaky_relu(1.0, alpha), 1e-10, "leaky_relu(1.0) should be 1.0");
        TestFramework::assertDoubleEqual(-0.01, Utils::leaky_relu(-1.0, alpha), 1e-10, "leaky_relu(-1.0) should be -0.01");
        TestFramework::assertDoubleEqual(10.0, Utils::leaky_relu(10.0, alpha), 1e-10, "leaky_relu(10.0) should be 10.0");
        TestFramework::assertDoubleEqual(-0.1, Utils::leaky_relu(-10.0, alpha), 1e-10, "leaky_relu(-10.0) should be -0.1");
        
        // Test with different alpha
        alpha = 0.2;
        TestFramework::assertDoubleEqual(-0.2, Utils::leaky_relu(-1.0, alpha), 1e-10, "leaky_relu(-1.0, 0.2) should be -0.2");
    });

    // Test Leaky ReLU derivative function
    suite.runTest("Leaky ReLU Derivative", []() {
        double alpha = 0.01;
        // Test Leaky ReLU derivative with various inputs
        TestFramework::assertDoubleEqual(alpha, Utils::leaky_relu_derivative(-1.0, alpha), 1e-10, 
                                        "leaky_relu_derivative(-1.0) should be alpha");
        TestFramework::assertDoubleEqual(1.0, Utils::leaky_relu_derivative(1.0, alpha), 1e-10, 
                                        "leaky_relu_derivative(1.0) should be 1.0");
        
        // Test with different alpha
        alpha = 0.2;
        TestFramework::assertDoubleEqual(alpha, Utils::leaky_relu_derivative(-1.0, alpha), 1e-10, 
                                        "leaky_relu_derivative(-1.0, 0.2) should be 0.2");
    });

    // Test tanh function
    suite.runTest("Tanh Function", []() {
        // Test tanh with various inputs
        TestFramework::assertDoubleEqual(0.0, Utils::tanh(0.0), 1e-10, "tanh(0.0) should be 0.0");
        TestFramework::assertDoubleEqual(std::tanh(1.0), Utils::tanh(1.0), 1e-10, "tanh(1.0) incorrect");
        TestFramework::assertDoubleEqual(std::tanh(-1.0), Utils::tanh(-1.0), 1e-10, "tanh(-1.0) incorrect");
        TestFramework::assertDoubleEqual(std::tanh(10.0), Utils::tanh(10.0), 1e-10, "tanh(10.0) incorrect");
        TestFramework::assertDoubleEqual(std::tanh(-10.0), Utils::tanh(-10.0), 1e-10, "tanh(-10.0) incorrect");
    });

    // Test tanh derivative function
    suite.runTest("Tanh Derivative", []() {
        // Test tanh_derivative with various inputs
        double x = 0.0;
        double t = std::tanh(x);
        TestFramework::assertDoubleEqual(1.0 - t * t, Utils::tanh_derivative(x), 1e-10, 
                                        "tanh_derivative(0.0) incorrect");
        
        x = 1.0;
        t = std::tanh(x);
        TestFramework::assertDoubleEqual(1.0 - t * t, Utils::tanh_derivative(x), 1e-10, 
                                        "tanh_derivative(1.0) incorrect");
        
        x = -1.0;
        t = std::tanh(x);
        TestFramework::assertDoubleEqual(1.0 - t * t, Utils::tanh_derivative(x), 1e-10, 
                                        "tanh_derivative(-1.0) incorrect");
    });

    // Test softmax function
    suite.runTest("Softmax Function", []() {
        // Test with typical input
        std::vector<double> input = {1.0, 2.0, 3.0};
        std::vector<double> expectedOutput = {0.09003057317038046, 0.24472847105479767, 0.6652409557748219};
        
        Utils::softmax(input);
        TestFramework::assertVectorDoubleEqual(expectedOutput, input, 1e-10, "softmax([1.0, 2.0, 3.0]) incorrect");
        
        // Test that softmax output sums to 1.0
        double sum = std::accumulate(input.begin(), input.end(), 0.0);
        TestFramework::assertDoubleEqual(1.0, sum, 1e-10, "softmax output should sum to 1.0");
        
        // Test with large values (numerical stability)
        input = {1000.0, 1000.0, 1000.0};
        expectedOutput = {0.33333333333333337, 0.33333333333333337, 0.33333333333333337};
        Utils::softmax(input);
        TestFramework::assertVectorDoubleEqual(expectedOutput, input, 1e-10, "softmax([1000.0, 1000.0, 1000.0]) incorrect");
        
        // Test with single value
        input = {5.0};
        expectedOutput = {1.0};
        Utils::softmax(input);
        TestFramework::assertVectorDoubleEqual(expectedOutput, input, 1e-10, "softmax([5.0]) incorrect");
        
        // Test with empty vector
        input = {};
        expectedOutput = {};
        Utils::softmax(input);
        TestFramework::assertVectorDoubleEqual(expectedOutput, input, 1e-10, "softmax([]) should return empty vector");
    });

    return suite;
}
