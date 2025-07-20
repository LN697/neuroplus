#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <stdexcept>
#include <sstream>
#include <cmath>

namespace TestFramework {

/**
 * @brief Represents a test case result
 */
struct TestResult {
    std::string name;
    bool passed;
    std::string message;
};

/**
 * @brief Represents a test suite with multiple test cases
 */
class TestSuite {
private:
    std::string name;
    std::vector<TestResult> results;
    int passed = 0;
    int failed = 0;

public:
    /**
     * @brief Constructor
     * @param suiteName Name of the test suite
     */
    TestSuite(const std::string& suiteName) : name(suiteName) {}
    
    /**
     * @brief Gets the name of the test suite
     * @return Test suite name
     */
    const std::string& getName() const {
        return name;
    }

    /**
     * @brief Runs a test case
     * @param testName Name of the test case
     * @param testFunc Function containing the test
     */
    void runTest(const std::string& testName, std::function<void()> testFunc) {
        TestResult result;
        result.name = testName;

        try {
            testFunc();
            result.passed = true;
            result.message = "PASSED";
            passed++;
        } catch (const std::exception& e) {
            result.passed = false;
            result.message = e.what();
            failed++;
        }

        results.push_back(result);
    }

    /**
     * @brief Prints the test suite results
     */
    void printResults() const {
        std::cout << "\n===== " << name << " Tests =====\n";
        
        for (const auto& result : results) {
            std::cout << (result.passed ? "[PASS] " : "[FAIL] ") << result.name << "\n";
            if (!result.passed) {
                std::cout << "      " << result.message << "\n";
            }
        }
        
        std::cout << "\nResults: " << passed << " passed, " << failed << " failed\n";
        std::cout << "Total: " << results.size() << " tests\n";
    }

    /**
     * @brief Gets the number of passed tests
     * @return Number of passed tests
     */
    int getPassedCount() const {
        return passed;
    }

    /**
     * @brief Gets the number of failed tests
     * @return Number of failed tests
     */
    int getFailedCount() const {
        return failed;
    }
};

/**
 * @brief Test assertion failed exception
 */
class AssertionFailedException : public std::exception {
private:
    std::string message;

public:
    /**
     * @brief Constructor
     * @param msg Error message
     */
    AssertionFailedException(const std::string& msg) : message(msg) {}

    /**
     * @brief Gets the error message
     * @return Error message
     */
    const char* what() const noexcept override {
        return message.c_str();
    }
};

/**
 * @brief Assert that a condition is true
 * @param condition Condition to check
 * @param message Error message if the condition fails
 */
inline void assertTrue(bool condition, const std::string& message) {
    if (!condition) {
        throw AssertionFailedException("Assertion failed: " + message);
    }
}

/**
 * @brief Assert that a condition is false
 * @param condition Condition to check
 * @param message Error message if the condition passes
 */
inline void assertFalse(bool condition, const std::string& message) {
    if (condition) {
        throw AssertionFailedException("Assertion failed: " + message);
    }
}

/**
 * @brief Assert that two values are equal
 * @tparam T Type of the values
 * @param expected Expected value
 * @param actual Actual value
 * @param message Error message if the values are not equal
 */
template<typename T>
void assertEqual(const T& expected, const T& actual, const std::string& message) {
    if (!(expected == actual)) {
        std::ostringstream oss;
        oss << "Assertion failed: " << message << " (expected: " << expected << ", actual: " << actual << ")";
        throw AssertionFailedException(oss.str());
    }
}

/**
 * @brief Assert that two double values are approximately equal
 * @param expected Expected value
 * @param actual Actual value
 * @param epsilon Maximum allowed difference
 * @param message Error message if the values are not approximately equal
 */
inline void assertDoubleEqual(double expected, double actual, double epsilon, const std::string& message) {
    if (std::fabs(expected - actual) > epsilon) {
        std::ostringstream oss;
        oss << "Assertion failed: " << message << " (expected: " << expected << ", actual: " << actual
            << ", difference: " << std::fabs(expected - actual) << ", epsilon: " << epsilon << ")";
        throw AssertionFailedException(oss.str());
    }
}

/**
 * @brief Assert that a vector of doubles is approximately equal to another vector of doubles
 * @param expected Expected vector
 * @param actual Actual vector
 * @param epsilon Maximum allowed difference for each element
 * @param message Error message if the vectors are not approximately equal
 */
inline void assertVectorDoubleEqual(const std::vector<double>& expected, const std::vector<double>& actual,
                                    double epsilon, const std::string& message) {
    if (expected.size() != actual.size()) {
        std::ostringstream oss;
        oss << "Assertion failed: " << message << " (vectors have different sizes: expected: "
            << expected.size() << ", actual: " << actual.size() << ")";
        throw AssertionFailedException(oss.str());
    }

    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::fabs(expected[i] - actual[i]) > epsilon) {
            std::ostringstream oss;
            oss << "Assertion failed: " << message << " at index " << i
                << " (expected: " << expected[i] << ", actual: " << actual[i]
                << ", difference: " << std::fabs(expected[i] - actual[i]) << ", epsilon: " << epsilon << ")";
            throw AssertionFailedException(oss.str());
        }
    }
}

/**
 * @brief Assert that a function throws an exception
 * @tparam E Exception type
 * @param func Function to check
 * @param message Error message if the function doesn't throw the expected exception
 */
template<typename E>
void assertThrows(std::function<void()> func, const std::string& message) {
    try {
        func();
        throw AssertionFailedException("Assertion failed: " + message + " (expected exception was not thrown)");
    } catch (const E&) {
        // Expected exception was thrown
    } catch (const AssertionFailedException& e) {
        // Rethrow assertion exceptions
        throw;
    } catch (const std::exception& e) {
        std::ostringstream oss;
        oss << "Assertion failed: " << message << " (expected exception of different type, got: " << e.what() << ")";
        throw AssertionFailedException(oss.str());
    }
}

} // namespace TestFramework
