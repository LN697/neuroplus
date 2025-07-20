#pragma once

#include "test_framework.hpp"
#include "../include/activation.hpp"
#include "../include/utils.hpp"
#include <vector>
#include <functional>

/**
 * @brief Tests for Activation layer functionality
 * @return TestSuite with the results
 */
TestFramework::TestSuite runActivationTests() {
    TestFramework::TestSuite suite("Activation");

    // Test Activation constructor and forward/backward with sigmoid
    suite.runTest("Sigmoid Activation Forward", []() {
        // Create a sigmoid activation layer
        Activation sigmoidLayer(Utils::sigmoid, Utils::sigmoid_derivative);
        
        // Test forward pass
        std::vector<double> input = {-2.0, -1.0, 0.0, 1.0, 2.0};
        std::vector<double> output = sigmoidLayer.forward(input);
        
        // Expected sigmoid values
        std::vector<double> expected = {
            1.0 / (1.0 + std::exp(2.0)),
            1.0 / (1.0 + std::exp(1.0)),
            0.5,
            1.0 / (1.0 + std::exp(-1.0)),
            1.0 / (1.0 + std::exp(-2.0))
        };
        
        // Check that output matches expected values
        TestFramework::assertVectorDoubleEqual(expected, output, 1e-10, "Sigmoid forward pass incorrect");
    });

    suite.runTest("Sigmoid Activation Backward", []() {
        // Create a sigmoid activation layer
        Activation sigmoidLayer(Utils::sigmoid, Utils::sigmoid_derivative);
        
        // First do a forward pass to set up the input cache
        std::vector<double> input = {-2.0, -1.0, 0.0, 1.0, 2.0};
        sigmoidLayer.forward(input);
        
        // Test backward pass
        std::vector<double> gradOutput = {0.5, 0.5, 0.5, 0.5, 0.5};
        std::vector<double> gradInput = sigmoidLayer.backward(gradOutput, 0.1); // learning rate doesn't matter for activation
        
        // Expected gradient values (sigmoid_derivative * gradOutput)
        std::vector<double> expected(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            expected[i] = Utils::sigmoid_derivative(input[i]) * gradOutput[i];
        }
        
        // Check that gradient input matches expected values
        TestFramework::assertVectorDoubleEqual(expected, gradInput, 1e-10, "Sigmoid backward pass incorrect");
    });

    // Test Activation with ReLU
    suite.runTest("ReLU Activation Forward", []() {
        // Create a ReLU activation layer
        Activation reluLayer(Utils::relu, Utils::relu_derivative);
        
        // Test forward pass
        std::vector<double> input = {-2.0, -1.0, 0.0, 1.0, 2.0};
        std::vector<double> output = reluLayer.forward(input);
        
        // Expected ReLU values
        std::vector<double> expected = {0.0, 0.0, 0.0, 1.0, 2.0};
        
        // Check that output matches expected values
        TestFramework::assertVectorDoubleEqual(expected, output, 1e-10, "ReLU forward pass incorrect");
    });

    suite.runTest("ReLU Activation Backward", []() {
        // Create a ReLU activation layer
        Activation reluLayer(Utils::relu, Utils::relu_derivative);
        
        // First do a forward pass to set up the input cache
        std::vector<double> input = {-2.0, -1.0, 0.0, 1.0, 2.0};
        reluLayer.forward(input);
        
        // Test backward pass
        std::vector<double> gradOutput = {0.5, 0.5, 0.5, 0.5, 0.5};
        std::vector<double> gradInput = reluLayer.backward(gradOutput, 0.1); // learning rate doesn't matter for activation
        
        // Expected gradient values (relu_derivative * gradOutput)
        std::vector<double> expected = {0.0, 0.0, 0.0, 0.5, 0.5};
        
        // Check that gradient input matches expected values
        TestFramework::assertVectorDoubleEqual(expected, gradInput, 1e-10, "ReLU backward pass incorrect");
    });

    // Test Activation with Leaky ReLU
    suite.runTest("Leaky ReLU Activation", []() {
        // Create a Leaky ReLU activation layer with alpha=0.1
        double alpha = 0.1;
        auto leakyRelu = [alpha](double x) { return Utils::leaky_relu(x, alpha); };
        auto leakyReluDerivative = [alpha](double x) { return Utils::leaky_relu_derivative(x, alpha); };
        Activation leakyReluLayer(leakyRelu, leakyReluDerivative);
        
        // Test forward pass
        std::vector<double> input = {-2.0, -1.0, 0.0, 1.0, 2.0};
        std::vector<double> output = leakyReluLayer.forward(input);
        
        // Expected Leaky ReLU values
        std::vector<double> expected = {-0.2, -0.1, 0.0, 1.0, 2.0};
        
        // Check that output matches expected values
        TestFramework::assertVectorDoubleEqual(expected, output, 1e-10, "Leaky ReLU forward pass incorrect");
        
        // Test backward pass
        std::vector<double> gradOutput = {0.5, 0.5, 0.5, 0.5, 0.5};
        std::vector<double> gradInput = leakyReluLayer.backward(gradOutput, 0.1);
        
        // Expected gradient values
        // For leaky ReLU, the derivative is alpha for x < 0, and 1 for x >= 0
        std::vector<double> expectedGrad = {0.05, 0.05, 0.05, 0.5, 0.5};
        
        // Check that gradient input matches expected values
        TestFramework::assertVectorDoubleEqual(expectedGrad, gradInput, 1e-10, "Leaky ReLU backward pass incorrect");
    });

    // Test Activation with tanh
    suite.runTest("Tanh Activation", []() {
        // Create a tanh activation layer
        Activation tanhLayer(Utils::tanh, Utils::tanh_derivative);
        
        // Test forward pass
        std::vector<double> input = {-2.0, -1.0, 0.0, 1.0, 2.0};
        std::vector<double> output = tanhLayer.forward(input);
        
        // Expected tanh values
        std::vector<double> expected = {
            std::tanh(-2.0),
            std::tanh(-1.0),
            std::tanh(0.0),
            std::tanh(1.0),
            std::tanh(2.0)
        };
        
        // Check that output matches expected values
        TestFramework::assertVectorDoubleEqual(expected, output, 1e-10, "Tanh forward pass incorrect");
        
        // Test backward pass
        std::vector<double> gradOutput = {0.5, 0.5, 0.5, 0.5, 0.5};
        std::vector<double> gradInput = tanhLayer.backward(gradOutput, 0.1);
        
        // Expected gradient values
        std::vector<double> expectedGrad(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            expectedGrad[i] = Utils::tanh_derivative(input[i]) * gradOutput[i];
        }
        
        // Check that gradient input matches expected values
        TestFramework::assertVectorDoubleEqual(expectedGrad, gradInput, 1e-10, "Tanh backward pass incorrect");
    });

    // Test Activation cloning
    suite.runTest("Activation Clone", []() {
        // Create a sigmoid activation layer
        Activation sigmoidLayer(Utils::sigmoid, Utils::sigmoid_derivative);
        
        // Clone the layer
        auto clonedLayer = sigmoidLayer.clone();
        auto* activationLayer = dynamic_cast<Activation*>(clonedLayer.get());
        
        // Verify the clone is not null and is of the correct type
        TestFramework::assertTrue(activationLayer != nullptr, "Clone should be an Activation layer");
        
        // Test that the cloned layer behaves the same as the original
        std::vector<double> input = {-1.0, 0.0, 1.0};
        std::vector<double> output1 = sigmoidLayer.forward(input);
        std::vector<double> output2 = activationLayer->forward(input);
        
        // Check that outputs match
        TestFramework::assertVectorDoubleEqual(output1, output2, 1e-10, "Cloned layer should produce the same output");
    });

    return suite;
}
