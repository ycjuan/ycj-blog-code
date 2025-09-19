#include "ThreadPool.h"
#include <iostream>
#include <chrono>
#include <random>
#include <vector>
#include <thread>

// Example task functions
int compute_factorial(int n) {
    if (n <= 1) return 1;
    
    // Simulate some work
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    int result = 1;
    for (int i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}

double compute_pi_estimate(int num_samples) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    int inside_circle = 0;
    for (int i = 0; i < num_samples; ++i) {
        double x = dis(gen);
        double y = dis(gen);
        if (x * x + y * y <= 1.0) {
            inside_circle++;
        }
    }
    
    return 4.0 * inside_circle / num_samples;
}

void print_message(const std::string& message, int delay_ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
    std::cout << "[Thread " << std::this_thread::get_id() << "] " << message << std::endl;
}

int main() {
    std::cout << "=== Thread Pool Demo ===" << std::endl;
    std::cout << "Hardware concurrency: " << std::thread::hardware_concurrency() << " threads" << std::endl;
    
    // Create a thread pool with 4 threads
    ThreadPool pool(4);
    std::cout << "Created thread pool with " << pool.size() << " threads" << std::endl;
    
    // Example 1: Submit tasks that return values
    std::cout << "\n--- Example 1: Computing factorials ---" << std::endl;
    std::vector<std::future<int>> factorial_results;
    
    for (int i = 1; i <= 8; ++i) {
        factorial_results.emplace_back(
            pool.enqueue(compute_factorial, i)
        );
    }
    
    // Collect results
    for (size_t i = 0; i < factorial_results.size(); ++i) {
        int result = factorial_results[i].get();
        std::cout << (i + 1) << "! = " << result << std::endl;
    }
    
    // Example 2: Submit tasks for Pi estimation
    std::cout << "\n--- Example 2: Estimating Pi ---" << std::endl;
    std::vector<std::future<double>> pi_results;
    
    for (int i = 0; i < 4; ++i) {
        pi_results.emplace_back(
            pool.enqueue(compute_pi_estimate, 1000000)
        );
    }
    
    double pi_sum = 0.0;
    for (size_t i = 0; i < pi_results.size(); ++i) {
        double estimate = pi_results[i].get();
        std::cout << "Pi estimate " << (i + 1) << ": " << estimate << std::endl;
        pi_sum += estimate;
    }
    std::cout << "Average Pi estimate: " << (pi_sum / pi_results.size()) << std::endl;
    
    // Example 3: Submit void tasks
    std::cout << "\n--- Example 3: Print messages ---" << std::endl;
    std::vector<std::future<void>> message_tasks;
    
    message_tasks.emplace_back(pool.enqueue(print_message, "Hello from task 1!", 50));
    message_tasks.emplace_back(pool.enqueue(print_message, "Hello from task 2!", 100));
    message_tasks.emplace_back(pool.enqueue(print_message, "Hello from task 3!", 150));
    message_tasks.emplace_back(pool.enqueue(print_message, "Hello from task 4!", 200));
    
    // Wait for all message tasks to complete
    for (auto& task : message_tasks) {
        task.wait();
    }
    
    // Example 4: Lambda functions
    std::cout << "\n--- Example 4: Lambda functions ---" << std::endl;
    auto lambda_result = pool.enqueue([](int x, int y) -> int {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        return x * x + y * y;
    }, 3, 4);
    
    std::cout << "3² + 4² = " << lambda_result.get() << std::endl;
    
    // Show pending tasks (should be 0 at this point)
    std::cout << "\nPending tasks: " << pool.pending_tasks() << std::endl;
    
    std::cout << "\n=== Demo completed ===" << std::endl;
    
    // ThreadPool destructor will be called automatically,
    // which will wait for all threads to finish
    
    return 0;
}
