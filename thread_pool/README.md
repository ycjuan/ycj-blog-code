# Thread Pool Implementation

A simple and efficient C++ thread pool implementation with modern C++11 features.

## Features

- **Thread-safe**: Uses mutexes and condition variables for synchronization
- **Template-based**: Supports any callable object (functions, lambdas, functors)
- **Future-based**: Returns `std::future` objects for result retrieval
- **Exception handling**: Properly handles exceptions in worker threads
- **RAII**: Automatic cleanup of threads in destructor
- **Configurable**: Specify the number of worker threads

## Files

- `ThreadPool.h` - Header file with class declaration and template implementations
- `ThreadPool.cpp` - Implementation of non-template methods
- `main.cpp` - Comprehensive demo showing various usage patterns
- `simple_test.cpp` - Basic unit tests
- `CMakeLists.txt` - CMake build configuration

## Building

### Using CMake (Recommended)

```bash
mkdir build
cd build
cmake ..
make
```

This will create two executables:
- `thread_pool_demo` - Full demonstration
- `simple_test` - Basic functionality tests

### Manual Compilation

```bash
g++ -std=c++11 -pthread -O2 main.cpp ThreadPool.cpp -o thread_pool_demo
g++ -std=c++11 -pthread -O2 simple_test.cpp ThreadPool.cpp -o simple_test
```

## Usage Examples

### Basic Usage

```cpp
#include "ThreadPool.h"

// Create a thread pool with 4 worker threads
ThreadPool pool(4);

// Submit a task that returns a value
auto result = pool.enqueue([](int x) { return x * 2; }, 21);
std::cout << result.get() << std::endl; // Prints: 42

// Submit a void task
auto task = pool.enqueue([]() { 
    std::cout << "Hello from thread!" << std::endl; 
});
task.wait(); // Wait for completion
```

### Function Objects

```cpp
// Regular functions
int factorial(int n) { /* ... */ }
auto result = pool.enqueue(factorial, 5);

// Member functions
class Calculator {
public:
    int multiply(int a, int b) { return a * b; }
};

Calculator calc;
auto result = pool.enqueue(&Calculator::multiply, &calc, 6, 7);
```

### Exception Handling

```cpp
auto result = pool.enqueue([]() -> int {
    throw std::runtime_error("Something went wrong!");
    return 42;
});

try {
    int value = result.get();
} catch (const std::exception& e) {
    std::cout << "Caught: " << e.what() << std::endl;
}
```

## API Reference

### Constructor
```cpp
ThreadPool(size_t num_threads)
```
Creates a thread pool with the specified number of worker threads.

### Methods

#### `enqueue`
```cpp
template<class F, class... Args>
auto enqueue(F&& f, Args&&... args) -> std::future<return_type>
```
Submits a task to the thread pool. Returns a `std::future` object for result retrieval.

#### `size`
```cpp
size_t size() const
```
Returns the number of worker threads in the pool.

#### `pending_tasks`
```cpp
size_t pending_tasks() const
```
Returns the number of tasks waiting in the queue.

#### `is_stopped`
```cpp
bool is_stopped() const
```
Returns `true` if the thread pool has been stopped.

## Implementation Details

- Uses a queue of `std::function<void()>` objects to store tasks
- Worker threads continuously poll the queue for new tasks
- Synchronization is achieved using `std::mutex` and `std::condition_variable`
- Tasks are packaged using `std::packaged_task` to enable future-based result retrieval
- Thread-safe shutdown ensures all threads complete before destruction

## Performance Considerations

- Choose the number of threads based on your workload and hardware
- For CPU-bound tasks: typically `std::thread::hardware_concurrency()`
- For I/O-bound tasks: can use more threads than CPU cores
- Consider task granularity - very small tasks may have overhead
- The queue is protected by a mutex, so very high-frequency task submission may become a bottleneck

## Thread Safety

This thread pool is fully thread-safe:
- Multiple threads can safely submit tasks simultaneously
- The internal queue and worker management are protected by mutexes
- Exception handling ensures that worker threads remain stable even if tasks throw

## License

This is sample code for educational purposes. Feel free to use and modify as needed.
