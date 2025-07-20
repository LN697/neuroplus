#pragma once

#include "test_framework.hpp"
#include "../include/optimizer.hpp"
#include <vector>
#include <cmath>

/**
 * @brief Tests for Optimizer functionality
 * @return TestSuite with the results
 */
TestFramework::TestSuite runOptimizerTests() {
    TestFramework::TestSuite suite("Optimizer");

    // Test SGD constructor
    suite.runTest("SGD Constructor", []() {
        SGD optimizer(0.01, 0.9);
        // No assertions needed, just checking that construction doesn't throw
    });

    // Test SGD update without momentum
    suite.runTest("SGD Update Without Momentum", []() {
        SGD optimizer(0.1, 0.0);
        
        std::vector<double> weights = {1.0, 2.0, 3.0};
        std::vector<double> gradients = {0.1, 0.2, 0.3};
        std::vector<double> expected = {
            1.0 - 0.1 * 0.1,
            2.0 - 0.1 * 0.2,
            3.0 - 0.1 * 0.3
        };
        
        optimizer.update(weights, gradients);
        
        TestFramework::assertVectorDoubleEqual(expected, weights, 1e-10, 
                                              "SGD update without momentum incorrect");
    });

    // Test SGD update with momentum
    suite.runTest("SGD Update With Momentum", []() {
        SGD optimizer(0.1, 0.9);
        
        std::vector<double> weights = {1.0, 2.0, 3.0};
        std::vector<double> gradients = {0.1, 0.2, 0.3};
        
        // First update (velocity starts at 0)
        std::vector<double> velocity = {
            0.9 * 0.0 - 0.1 * 0.1,
            0.9 * 0.0 - 0.1 * 0.2,
            0.9 * 0.0 - 0.1 * 0.3
        };
        
        std::vector<double> expected1 = {
            1.0 + velocity[0],
            2.0 + velocity[1],
            3.0 + velocity[2]
        };
        
        optimizer.update(weights, gradients);
        
        TestFramework::assertVectorDoubleEqual(expected1, weights, 1e-10, 
                                              "SGD update with momentum (first update) incorrect");
        
        // Second update (now velocity has values)
        std::vector<double> gradients2 = {0.2, 0.3, 0.4};
        
        std::vector<double> velocity2 = {
            0.9 * velocity[0] - 0.1 * 0.2,
            0.9 * velocity[1] - 0.1 * 0.3,
            0.9 * velocity[2] - 0.1 * 0.4
        };
        
        std::vector<double> expected2 = {
            expected1[0] + velocity2[0],
            expected1[1] + velocity2[1],
            expected1[2] + velocity2[2]
        };
        
        optimizer.update(weights, gradients2);
        
        TestFramework::assertVectorDoubleEqual(expected2, weights, 1e-10, 
                                              "SGD update with momentum (second update) incorrect");
    });

    // Test SGD clone
    suite.runTest("SGD Clone", []() {
        SGD optimizer(0.1, 0.9);
        
        // Update with some data to initialize internal state
        std::vector<double> weights1 = {1.0, 2.0, 3.0};
        std::vector<double> gradients = {0.1, 0.2, 0.3};
        optimizer.update(weights1, gradients);
        
        // Clone the optimizer
        std::unique_ptr<Optimizer> clonedPtr = optimizer.clone();
        auto* cloned = dynamic_cast<SGD*>(clonedPtr.get());
        
        // Verify the clone is not null and is of the correct type
        TestFramework::assertTrue(cloned != nullptr, "Clone should be an SGD optimizer");
        
        // Test with the cloned optimizer
        std::vector<double> weights2 = {1.0, 2.0, 3.0};
        std::vector<double> weights3 = {1.0, 2.0, 3.0};
        
        // Second update with original optimizer
        optimizer.update(weights2, gradients);
        
        // Update with cloned optimizer
        cloned->update(weights3, gradients);
        
        // Check that the results match
        TestFramework::assertVectorDoubleEqual(weights2, weights3, 1e-10, 
                                              "Cloned SGD optimizer should produce the same results");
    });

    // Test Adam constructor
    suite.runTest("Adam Constructor", []() {
        Adam optimizer(0.001, 0.9, 0.999, 1e-8);
        // No assertions needed, just checking that construction doesn't throw
    });

    // Test Adam update
    suite.runTest("Adam Update", []() {
        Adam optimizer(0.001, 0.9, 0.999, 1e-8);
        
        std::vector<double> weights = {1.0, 2.0, 3.0};
        std::vector<double> gradients = {0.1, 0.2, 0.3};
        
        // First update
        optimizer.update(weights, gradients);
        
        // We won't test exact values since Adam is complex, but we can check that weights changed
        TestFramework::assertFalse(weights[0] == 1.0 && weights[1] == 2.0 && weights[2] == 3.0,
                                  "Adam should update weights");
        
        // Check that weights changed in the expected direction (gradients are positive, so weights should decrease)
        TestFramework::assertTrue(weights[0] < 1.0, "Adam should decrease weight[0]");
        TestFramework::assertTrue(weights[1] < 2.0, "Adam should decrease weight[1]");
        TestFramework::assertTrue(weights[2] < 3.0, "Adam should decrease weight[2]");
    });

    // Test Adam clone
    suite.runTest("Adam Clone", []() {
        Adam optimizer(0.001, 0.9, 0.999, 1e-8);
        
        // Update with some data to initialize internal state
        std::vector<double> weights1 = {1.0, 2.0, 3.0};
        std::vector<double> gradients = {0.1, 0.2, 0.3};
        optimizer.update(weights1, gradients);
        
        // Clone the optimizer
        std::unique_ptr<Optimizer> clonedPtr = optimizer.clone();
        auto* cloned = dynamic_cast<Adam*>(clonedPtr.get());
        
        // Verify the clone is not null and is of the correct type
        TestFramework::assertTrue(cloned != nullptr, "Clone should be an Adam optimizer");
    });

    return suite;
}
