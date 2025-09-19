#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include "thread_pool.h"

using namespace thread_pool;

// Example functions to demonstrate thread pool usage

// Simple function that simulates some work
int compute_square(int x) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Simulate work
    return x * x;
}

// Function that processes a vector of numbers
std::vector<int> process_vector(const std::vector<int>& input) {
    std::vector<int> result;
    result.reserve(input.size());
    
    for (int value : input) {
        // Simulate some processing
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        result.push_back(value * 2 + 1);
    }
    
    return result;
}

// Function that finds the maximum element in a range
int find_max_in_range(const std::vector<int>& data, size_t start, size_t end) {
    if (start >= end || start >= data.size()) return 0;
    
    int max_val = data[start];
    for (size_t i = start + 1; i < std::min(end, data.size()); ++i) {
        if (data[i] > max_val) {
            max_val = data[i];
        }
    }
    
    return max_val;
}

// Example of parallel processing with thread pool
void parallel_vector_processing() {
    std::cout << "\n=== Parallel Vector Processing Example ===" << std::endl;
    
    // Create a large vector of random numbers
    std::vector<int> data(1000);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 1000);
    
    for (auto& val : data) {
        val = dis(gen);
    }
    
    // Process with thread pool
    ThreadPool pool(4); // Use 4 threads
    std::vector<std::future<int>> futures;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Divide work among threads
    size_t chunk_size = data.size() / pool.size();
    for (size_t i = 0; i < pool.size(); ++i) {
        size_t start_idx = i * chunk_size;
        size_t end_idx = (i == pool.size() - 1) ? data.size() : (i + 1) * chunk_size;
        
        futures.push_back(
            pool.enqueue(find_max_in_range, std::ref(data), start_idx, end_idx)
        );
    }
    
    // Collect results
    int global_max = 0;
    for (auto& future : futures) {
        int local_max = future.get();
        global_max = std::max(global_max, local_max);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Parallel processing completed in " << duration.count() << " ms" << std::endl;
    std::cout << "Global maximum: " << global_max << std::endl;
}

// Example of using futures for async computation
void async_computation_example() {
    std::cout << "\n=== Async Computation Example ===" << std::endl;
    
    ThreadPool pool(2);
    
    // Submit multiple tasks and collect futures
    std::vector<std::future<int>> futures;
    
    for (int i = 1; i <= 10; ++i) {
        futures.push_back(pool.enqueue(compute_square, i));
    }
    
    // Process results as they become available
    std::cout << "Computing squares of 1-10:" << std::endl;
    for (size_t i = 0; i < futures.size(); ++i) {
        int result = futures[i].get();
        std::cout << (i + 1) << "^2 = " << result << std::endl;
    }
}

// Example of batch processing
void batch_processing_example() {
    std::cout << "\n=== Batch Processing Example ===" << std::endl;
    
    // Create multiple data batches
    std::vector<std::vector<int>> batches(5);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 100);
    
    for (auto& batch : batches) {
        batch.resize(20);
        for (auto& val : batch) {
            val = dis(gen);
        }
    }
    
    ThreadPool pool(3);
    std::vector<std::future<std::vector<int>>> futures;
    
    // Process each batch in parallel
    for (const auto& batch : batches) {
        futures.push_back(pool.enqueue(process_vector, std::ref(batch)));
    }
    
    // Collect and display results
    std::cout << "Processing " << batches.size() << " batches in parallel:" << std::endl;
    for (size_t i = 0; i < futures.size(); ++i) {
        auto result = futures[i].get();
        std::cout << "Batch " << (i + 1) << " processed: ";
        for (size_t j = 0; j < std::min(result.size(), size_t(5)); ++j) {
            std::cout << result[j] << " ";
        }
        if (result.size() > 5) std::cout << "...";
        std::cout << std::endl;
    }
}

// Example demonstrating thread pool lifecycle
void lifecycle_example() {
    std::cout << "\n=== Thread Pool Lifecycle Example ===" << std::endl;
    
    {
        // Create a thread pool with 2 threads
        ThreadPool pool(2);
        std::cout << "Thread pool created with " << pool.size() << " threads" << std::endl;
        std::cout << "Thread pool is running: " << (pool.is_running() ? "No" : "Yes") << std::endl;
        
        // Submit some tasks
        auto future1 = pool.enqueue([]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            return std::string("Task 1 completed");
        });
        
        auto future2 = pool.enqueue([]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
            return std::string("Task 2 completed");
        });
        
        std::cout << "Tasks submitted to thread pool" << std::endl;
        
        // Wait for tasks to complete
        std::cout << future1.get() << std::endl;
        std::cout << future2.get() << std::endl;
        
        std::cout << "All tasks completed" << std::endl;
    } // Thread pool is destroyed here, all threads are joined
    
    std::cout << "Thread pool destroyed" << std::endl;
}

int main() {
    std::cout << "C++ Thread Pool Implementation Examples" << std::endl;
    std::cout << "======================================" << std::endl;
    
    // Run all examples
    lifecycle_example();
    async_computation_example();
    parallel_vector_processing();
    batch_processing_example();
    
    std::cout << "\nAll examples completed successfully!" << std::endl;
    
    return 0;
}
