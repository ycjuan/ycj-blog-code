#include "ThreadPool.h"
#include <iostream>

ThreadPool::ThreadPool(size_t num_threads) : stop(false) {
    // Create worker threads
    for (size_t i = 0; i < num_threads; ++i) {
        workers.emplace_back([this] {
            for (;;) {
                std::function<void()> task;
                
                {
                    std::unique_lock<std::mutex> lock(this->queue_mutex);
                    
                    // Wait for a task or stop signal
                    this->condition.wait(lock, [this] { 
                        return this->stop.load() || !this->tasks.empty(); 
                    });
                    
                    // If we're stopping and no tasks are left, exit
                    if (this->stop.load() && this->tasks.empty()) {
                        return;
                    }
                    
                    // Get the next task
                    task = std::move(this->tasks.front());
                    this->tasks.pop();
                }
                
                // Execute the task
                try {
                    task();
                } catch (const std::exception& e) {
                    std::cerr << "Task execution failed: " << e.what() << std::endl;
                } catch (...) {
                    std::cerr << "Task execution failed with unknown exception" << std::endl;
                }
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    // Signal all threads to stop
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop.store(true);
    }
    
    // Wake up all threads
    condition.notify_all();
    
    // Wait for all threads to finish
    for (std::thread& worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

size_t ThreadPool::pending_tasks() const {
    std::unique_lock<std::mutex> lock(const_cast<std::mutex&>(queue_mutex));
    return tasks.size();
}
