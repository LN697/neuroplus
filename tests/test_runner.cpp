#include "test_framework.hpp"
#include "test_utils.hpp"
#include "test_activation.hpp"
#include "test_dense.hpp"
#include "test_loss.hpp"
#include "test_neuralnet.hpp"
#include "test_optimizer.hpp"
#include "test_replay_buffer.hpp"

#include <iostream>
#include <vector>

int main() {
    std::cout << "Running NeuroPlus Tests...\n";

    // Vector to store all test suites
    std::vector<TestFramework::TestSuite> testSuites;

    // Run all test suites
    testSuites.push_back(runUtilsTests());
    testSuites.push_back(runActivationTests());
    testSuites.push_back(runDenseTests());
    testSuites.push_back(runLossTests());
    testSuites.push_back(runNeuralNetTests());
    testSuites.push_back(runOptimizerTests());
    testSuites.push_back(runReplayBufferTests());

    // Calculate summary
    int totalTests = 0;
    int totalPassed = 0;
    int totalFailed = 0;

    for (const auto& suite : testSuites) {
        totalPassed += suite.getPassedCount();
        totalFailed += suite.getFailedCount();
        totalTests += suite.getPassedCount() + suite.getFailedCount();
    }

    // Print summary
    std::cout << "\n===== Test Summary =====\n";
    
    // Print results for each test suite
    for (const auto& suite : testSuites) {
        if (suite.getFailedCount() > 0) {
            std::cout << "Failed tests in " << suite.getName() << " suite:\n";
            suite.printResults();
            std::cout << "\n";
        }
    }
    
    std::cout << "Total Tests: " << totalTests << "\n";
    std::cout << "Passed: " << totalPassed << "\n";
    std::cout << "Failed: " << totalFailed << "\n";
    std::cout << "Success Rate: " << (totalTests > 0 ? (100.0 * totalPassed / totalTests) : 0) << "%\n";
    
    // Return non-zero exit code if any tests failed
    return (totalFailed > 0) ? 1 : 0;
}
