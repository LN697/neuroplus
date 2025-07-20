#pragma once

#include "test_framework.hpp"
#include "../include/neuralnet.hpp"
#include "../include/activation.hpp"
#include "../include/dense.hpp"
#include "../include/loss.hpp"
#include "../include/utils.hpp"
#include <vector>
#include <memory>

/**
 * @brief Tests for NeuralNet functionality
 * @return TestSuite with the results
 */
TestFramework::TestSuite runNeuralNetTests() {
    TestFramework::TestSuite suite("NeuralNet");

    // Test neural network construction
    suite.runTest("NeuralNet Construction", []() {
        NeuralNet net;
        
        // Add layers
        net.addLayer(std::make_shared<Dense>(2, 3));
        net.addLayer(std::make_shared<Activation>(Utils::sigmoid, Utils::sigmoid_derivative));
        net.addLayer(std::make_shared<Dense>(3, 1));
        net.addLayer(std::make_shared<Activation>(Utils::sigmoid, Utils::sigmoid_derivative));
        
        // Set loss function
        net.setLoss(std::make_shared<MSELoss>());
        
        // The test passes if we reach this point without exceptions
        TestFramework::assertTrue(true, "Neural network construction should not throw exceptions");
    });

    // Test prediction
    suite.runTest("NeuralNet Prediction", []() {
        NeuralNet net;
        
        // Add layers
        net.addLayer(std::make_shared<Dense>(2, 3));
        net.addLayer(std::make_shared<Activation>(Utils::sigmoid, Utils::sigmoid_derivative));
        net.addLayer(std::make_shared<Dense>(3, 1));
        net.addLayer(std::make_shared<Activation>(Utils::sigmoid, Utils::sigmoid_derivative));
        
        // Make a prediction
        std::vector<double> input = {0.5, 0.5};
        std::vector<double> output = net.predict(input);
        
        // Check output size
        TestFramework::assertEqual(1, (int)output.size(), "Output size should be 1");
        
        // Check that output is between 0 and 1 (sigmoid output range)
        TestFramework::assertTrue(output[0] >= 0.0 && output[0] <= 1.0, 
                                 "Sigmoid output should be between 0 and 1");
    });

    // Test training (minimal training, just to check if it runs without errors)
    suite.runTest("NeuralNet Training", []() {
        NeuralNet net;
        
        // Add layers for XOR problem
        net.addLayer(std::make_shared<Dense>(2, 3));
        net.addLayer(std::make_shared<Activation>(Utils::sigmoid, Utils::sigmoid_derivative));
        net.addLayer(std::make_shared<Dense>(3, 1));
        net.addLayer(std::make_shared<Activation>(Utils::sigmoid, Utils::sigmoid_derivative));
        
        // Set loss function
        net.setLoss(std::make_shared<MSELoss>());
        
        // Training data for XOR
        std::vector<std::vector<double>> inputs = {
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
        };
        
        std::vector<std::vector<double>> targets = {
            {0},
            {1},
            {1},
            {0}
        };
        
        // Train for just a few epochs to test functionality, not performance
        int epochs = 5;
        double learning_rate = 0.1;
        
        // This should not throw exceptions
        net.train(inputs, targets, epochs, learning_rate);
        
        // Verify predictions after training
        // We're not checking for accuracy, just that predictions are in the right range
        for (const auto& input : inputs) {
            std::vector<double> output = net.predict(input);
            TestFramework::assertTrue(output[0] >= 0.0 && output[0] <= 1.0, 
                                     "Predicted output should be between 0 and 1");
        }
    });

    // Test neural network copy constructor
    suite.runTest("NeuralNet Copy Constructor", []() {
        // Create original network
        NeuralNet net1;
        net1.addLayer(std::make_shared<Dense>(2, 3));
        net1.addLayer(std::make_shared<Activation>(Utils::sigmoid, Utils::sigmoid_derivative));
        net1.setLoss(std::make_shared<MSELoss>());
        
        // Create a copy
        NeuralNet net2(net1);
        
        // Test that both networks produce the same output
        std::vector<double> input = {0.5, 0.5};
        std::vector<double> output1 = net1.predict(input);
        std::vector<double> output2 = net2.predict(input);
        
        // Check that outputs match
        TestFramework::assertEqual(output1.size(), output2.size(), 
                                  "Copied network output size should match original");
        TestFramework::assertVectorDoubleEqual(output1, output2, 1e-10, 
                                              "Copied network should produce the same output");
    });

    // Test network with different activation functions
    suite.runTest("NeuralNet With Different Activations", []() {
        // Test with ReLU
        NeuralNet netReLU;
        netReLU.addLayer(std::make_shared<Dense>(2, 3));
        netReLU.addLayer(std::make_shared<Activation>(Utils::relu, Utils::relu_derivative));
        netReLU.addLayer(std::make_shared<Dense>(3, 1));
        netReLU.setLoss(std::make_shared<MSELoss>());
        
        // Test with tanh
        NeuralNet netTanh;
        netTanh.addLayer(std::make_shared<Dense>(2, 3));
        netTanh.addLayer(std::make_shared<Activation>(Utils::tanh, Utils::tanh_derivative));
        netTanh.addLayer(std::make_shared<Dense>(3, 1));
        netTanh.setLoss(std::make_shared<MSELoss>());
        
        // Test with leaky ReLU
        auto leakyRelu = [](double x) { return Utils::leaky_relu(x, 0.1); };
        auto leakyReluDerivative = [](double x) { return Utils::leaky_relu_derivative(x, 0.1); };
        
        NeuralNet netLeakyReLU;
        netLeakyReLU.addLayer(std::make_shared<Dense>(2, 3));
        netLeakyReLU.addLayer(std::make_shared<Activation>(leakyRelu, leakyReluDerivative));
        netLeakyReLU.addLayer(std::make_shared<Dense>(3, 1));
        netLeakyReLU.setLoss(std::make_shared<MSELoss>());
        
        // Make predictions with all networks
        std::vector<double> input = {0.5, 0.5};
        
        std::vector<double> outputReLU = netReLU.predict(input);
        std::vector<double> outputTanh = netTanh.predict(input);
        std::vector<double> outputLeakyReLU = netLeakyReLU.predict(input);
        
        // Check output sizes
        TestFramework::assertEqual(1, (int)outputReLU.size(), "ReLU network output size should be 1");
        TestFramework::assertEqual(1, (int)outputTanh.size(), "Tanh network output size should be 1");
        TestFramework::assertEqual(1, (int)outputLeakyReLU.size(), "Leaky ReLU network output size should be 1");
        
        // Check that outputs are finite
        TestFramework::assertTrue(std::isfinite(outputReLU[0]), "ReLU network output should be finite");
        TestFramework::assertTrue(std::isfinite(outputTanh[0]), "Tanh network output should be finite");
        TestFramework::assertTrue(std::isfinite(outputLeakyReLU[0]), "Leaky ReLU network output should be finite");
    });

    return suite;
}
