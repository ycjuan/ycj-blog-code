#include "ThreadPool.h"
#include <iostream>
#include <chrono>
#include <cassert>
#include <thread>

// Simple test to verify thread pool functionality
int main() {
    std::cout << "Running simple thread pool tests..." << std::endl;
    
    // Test 1: Basic functionality
    {
        ThreadPool pool(2);
        
        auto result1 = pool.enqueue([](int x) { return x * 2; }, 21);
        auto result2 = pool.enqueue([](int x) { return x * 3; }, 14);
        
        assert(result1.get() == 42);
        assert(result2.get() == 42);
        
        std::cout << "✓ Test 1 passed: Basic functionality" << std::endl;
    }
    
    // Test 2: Multiple tasks
    {
        ThreadPool pool(4);
        std::vector<std::future<int>> results;
        
        for (int i = 0; i < 10; ++i) {
            results.emplace_back(pool.enqueue([](int x) { 
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                return x * x; 
            }, i));
        }
        
        for (int i = 0; i < 10; ++i) {
            assert(results[i].get() == i * i);
        }
        
        std::cout << "✓ Test 2 passed: Multiple tasks" << std::endl;
    }
    
    // Test 3: Exception handling
    {
        ThreadPool pool(1);
        
        auto result = pool.enqueue([]() -> int {
            throw std::runtime_error("Test exception");
            return 42;
        });
        
        try {
            result.get();
            assert(false); // Should not reach here
        } catch (const std::runtime_error& e) {
            std::cout << "✓ Test 3 passed: Exception handling" << std::endl;
        }
    }
    
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
