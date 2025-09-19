#pragma once

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>
#include <atomic>

class ThreadPool {
public:
    // Constructor: creates a thread pool with the specified number of threads
    explicit ThreadPool(size_t num_threads);
    
    // Destructor: stops all threads and waits for them to finish
    ~ThreadPool();
    
    // Submit a task to the thread pool
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type>;
    
    // Get the number of threads in the pool
    size_t size() const { return workers.size(); }
    
    // Get the number of pending tasks
    size_t pending_tasks() const;
    
    // Check if the thread pool is stopped
    bool is_stopped() const { return stop.load(); }

private:
    // Vector of worker threads
    std::vector<std::thread> workers;
    
    // Task queue
    std::queue<std::function<void()>> tasks;
    
    // Synchronization primitives
    std::mutex queue_mutex;
    std::condition_variable condition;
    
    // Stop flag
    std::atomic<bool> stop;
};

// Template method implementation must be in header file
template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args) 
    -> std::future<typename std::result_of<F(Args...)>::type> {
    
    using return_type = typename std::result_of<F(Args...)>::type;
    
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );
    
    std::future<return_type> res = task->get_future();
    
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        
        // Don't allow enqueueing after stopping the pool
        if (stop.load()) {
            throw std::runtime_error("enqueue on stopped ThreadPool");
        }
        
        tasks.emplace([task](){ (*task)(); });
    }
    
    condition.notify_one();
    return res;
}
