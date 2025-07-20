#pragma once

#include "test_framework.hpp"
#include "../include/replay_buffer.hpp"
#include <vector>
#include <stdexcept>
#include <unordered_set>

/**
 * @brief Tests for ReplayBuffer functionality
 * @return TestSuite with the results
 */
TestFramework::TestSuite runReplayBufferTests() {
    TestFramework::TestSuite suite("ReplayBuffer");

    // Test ReplayBuffer constructor
    suite.runTest("ReplayBuffer Constructor", []() {
        ReplayBuffer buffer(100);
        TestFramework::assertEqual(0, (int)buffer.size(), "New buffer should be empty");
        TestFramework::assertFalse(buffer.is_ready(1), "New buffer should not be ready for sampling");
    });

    // Test pushing experiences
    suite.runTest("ReplayBuffer Push", []() {
        ReplayBuffer buffer(3);
        
        // Create some test experiences
        Experience exp1 = {{1.0, 2.0}, 0, 0.5, {3.0, 4.0}, false};
        Experience exp2 = {{5.0, 6.0}, 1, 1.0, {7.0, 8.0}, true};
        
        // Push experiences
        buffer.push(exp1);
        TestFramework::assertEqual(1, (int)buffer.size(), "Buffer size should be 1 after pushing once");
        
        buffer.push(exp2);
        TestFramework::assertEqual(2, (int)buffer.size(), "Buffer size should be 2 after pushing twice");
        
        // Check readiness
        TestFramework::assertTrue(buffer.is_ready(1), "Buffer should be ready for batch size 1");
        TestFramework::assertTrue(buffer.is_ready(2), "Buffer should be ready for batch size 2");
        TestFramework::assertFalse(buffer.is_ready(3), "Buffer should not be ready for batch size 3");
    });

    // Test sampling
    suite.runTest("ReplayBuffer Sampling", []() {
        ReplayBuffer buffer(100);
        
        // Push some experiences
        for (int i = 0; i < 10; ++i) {
            Experience exp = {{(double)i, (double)(i+1)}, i % 2, (double)i/10, {(double)(i+2), (double)(i+3)}, (i % 2) == 0};
            buffer.push(exp);
        }
        
        // Sample a batch
        std::vector<Experience> batch = buffer.sample(5);
        
        // Check batch size
        TestFramework::assertEqual(5, (int)batch.size(), "Sampled batch should have size 5");
        
        // Check that sampled experiences are from the buffer
        // This is probabilistic, so we can't check exact matches, but we can check ranges
        for (const auto& exp : batch) {
            TestFramework::assertTrue(exp.state.size() == 2, "Sampled experience state size should be 2");
            TestFramework::assertTrue(exp.next_state.size() == 2, "Sampled experience next_state size should be 2");
        }
    });

    // Test circular buffer behavior
    suite.runTest("ReplayBuffer Circular Behavior", []() {
        ReplayBuffer buffer(3);
        
        // Push more experiences than capacity
        for (int i = 0; i < 5; ++i) {
            Experience exp = {{(double)i, 0.0}, 0, 0.0, {0.0, 0.0}, false};
            buffer.push(exp);
        }
        
        // Check size is capped at capacity
        TestFramework::assertEqual(3, (int)buffer.size(), "Buffer size should be capped at capacity");
        
        // Sample all experiences
        std::vector<Experience> batch = buffer.sample(3);
        
        // Check that we only have the most recent experiences
        // The oldest experiences (0, 1) should have been overwritten
        std::unordered_set<int> values;
        for (const auto& exp : batch) {
            int value = (int)exp.state[0];
            TestFramework::assertTrue(value >= 2 && value <= 4, 
                                     "Sampled values should be from the most recent experiences");
            values.insert(value);
        }
        
        // Check that we have 3 distinct values (no duplicates in the buffer)
        TestFramework::assertEqual(3, (int)values.size(), "Should have 3 distinct experiences in the buffer");
    });

    // Test sampling error when not enough experiences
    suite.runTest("ReplayBuffer Sampling Error", []() {
        ReplayBuffer buffer(100);
        
        // Push just one experience
        Experience exp = {{1.0, 2.0}, 0, 0.5, {3.0, 4.0}, false};
        buffer.push(exp);
        
        // Try to sample more than available
        TestFramework::assertThrows<std::runtime_error>(
            [&]() { buffer.sample(2); },
            "Sampling more than available should throw an exception"
        );
    });

    return suite;
}
