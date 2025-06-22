#ifndef CORE_H
#define CORE_H

#include <future> // For std::async
#include <vector> // For std::vector
#include <thread> // For std::thread
#include <chrono> // For std::chrono

// Core class
class Core
{
public:
    void process(int numThreads)
    {
        // Create a vector of futures to hold the results of async tasks
        std::vector<std::future<void>> futures;
        
        // Launch multiple async tasks
        for (int i = 0; i < numThreads; ++i) {
            futures.push_back(std::async(std::launch::async, [i]() {
                // Simulate some work with a simple print statement
                //std::cout << "Thread " << i << " is working." << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(1)); // Simulate work
            }));
        }
        
        // Wait for all async tasks to complete
        for (auto& fut : futures) {
            fut.wait();
        }
    }
};

#endif // CORE_H