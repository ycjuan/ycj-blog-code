# C++ Thread Pool Implementation

This directory contains a complete C++ thread pool implementation with examples and documentation.

## Files

- `thread_pool.h` - Header-only thread pool implementation
- `thread_pool_example.cpp` - Comprehensive examples demonstrating usage
- `Makefile.threadpool` - Build configuration for the examples
- `THREAD_POOL_README.md` - This documentation file

## Features

The thread pool implementation provides:

- **Header-only design** - No separate compilation needed
- **C++11 compatible** - Works with modern C++ compilers
- **Automatic thread management** - Creates and destroys threads automatically
- **Task queuing** - Thread-safe task submission and execution
- **Future-based results** - Get results from asynchronous tasks
- **Exception safety** - Proper cleanup on destruction
- **Configurable thread count** - Defaults to hardware concurrency

## Quick Start

### Compilation

```bash
# Build the example
make -f Makefile.threadpool

# Run the example
make -f Makefile.threadpool run

# Clean up
make -f Makefile.threadpool clean
```

### Basic Usage

```cpp
#include "thread_pool.h"

// Option 1: Use the namespace
using namespace thread_pool;
ThreadPool pool(4);

// Option 2: Use fully qualified name
thread_pool::ThreadPool pool(4);

// Submit a task and get a future
auto future = pool.enqueue([](int x) { return x * x; }, 5);

// Get the result (blocks until task completes)
int result = future.get(); // result = 25
```

## API Reference

### Constructor

```cpp
thread_pool::ThreadPool(size_t threads = std::thread::hardware_concurrency())
```

Creates a thread pool with the specified number of worker threads. Defaults to the number of hardware threads available.

### Destructor

```cpp
~thread_pool::ThreadPool()
```

Stops all worker threads and waits for them to complete. Automatically called when the pool goes out of scope.

### Enqueue Task

```cpp
template<class F, class... Args>
auto thread_pool::ThreadPool::enqueue(F&& f, Args&&... args) 
    -> std::future<typename std::result_of<F(Args...)>::type>
```

Submits a task to the thread pool for execution. Returns a `std::future` that can be used to retrieve the result.

**Parameters:**
- `f` - Callable object (function, lambda, functor)
- `args...` - Arguments to pass to the callable

**Returns:** `std::future` containing the result of the task

### Utility Methods

```cpp
size_t thread_pool::ThreadPool::size() const           // Returns number of worker threads
bool thread_pool::ThreadPool::is_running() const       // Returns true if pool is stopped
```

## Examples

### 1. Simple Task Submission

```cpp
#include "thread_pool.h"
using namespace thread_pool;

ThreadPool pool(2);

// Submit a simple task
auto future = pool.enqueue([]() {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    return 42;
});

// Do other work...
std::cout << "Result: " << future.get() << std::endl;
```

### 2. Parallel Processing

```cpp
#include "thread_pool.h"
using namespace thread_pool;

ThreadPool pool(4);
std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
std::vector<std::future<int>> futures;

// Process each element in parallel
for (int value : data) {
    futures.push_back(pool.enqueue([](int x) {
        return x * x; // Square the number
    }, value));
}

// Collect results
for (auto& future : futures) {
    std::cout << future.get() << " ";
}
```

### 3. Batch Processing

```cpp
#include "thread_pool.h"
using namespace thread_pool;

ThreadPool pool(3);
std::vector<std::vector<int>> batches = {{1,2,3}, {4,5,6}, {7,8,9}};
std::vector<std::future<std::vector<int>>> futures;

// Process each batch in parallel
for (const auto& batch : batches) {
    futures.push_back(pool.enqueue([](const std::vector<int>& data) {
        std::vector<int> result;
        for (int x : data) {
            result.push_back(x * 2);
        }
        return result;
    }, batch));
}

// Collect results
for (auto& future : futures) {
    auto result = future.get();
    // Process result...
}
```

## Performance Considerations

1. **Thread Count**: The optimal number of threads depends on your workload:
   - CPU-bound tasks: Use `std::thread::hardware_concurrency()`
   - I/O-bound tasks: May benefit from more threads
   - Memory-bound tasks: May benefit from fewer threads

2. **Task Granularity**: Balance between:
   - Too many small tasks: Overhead from task management
   - Too few large tasks: Poor parallelization

3. **Memory Usage**: Each thread has its own stack (typically 1-8MB)

## Thread Safety

- The thread pool is thread-safe for concurrent access
- Multiple threads can safely call `enqueue()` simultaneously
- The internal task queue is protected by mutexes
- Results are returned through `std::future` which is thread-safe

## Error Handling

- Tasks that throw exceptions are caught and stored in the future
- The exception will be re-thrown when `future.get()` is called
- The thread pool continues running even if individual tasks fail

## Requirements

- C++11 or later
- Threading support in the standard library
- Compiler with support for `std::thread`, `std::future`, and `std::packaged_task`

## Compilation Notes

The implementation uses:
- `std::thread` for worker threads
- `std::queue` for task storage
- `std::mutex` and `std::condition_variable` for synchronization
- `std::future` and `std::packaged_task` for result handling
- `std::bind` for argument binding

Make sure to link with the pthread library when compiling:
```bash
g++ -std=c++11 -pthread your_code.cpp
```
