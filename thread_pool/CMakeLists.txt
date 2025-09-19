cmake_minimum_required(VERSION 3.10)
project(ThreadPoolDemo)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add compiler flags for better debugging and warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")

# Find required packages
find_package(Threads REQUIRED)

# Create the executable
add_executable(thread_pool_demo 
    main.cpp 
    ThreadPool.cpp
)

# Link against threads library
target_link_libraries(thread_pool_demo Threads::Threads)

# Optional: Add a simple test executable
add_executable(simple_test simple_test.cpp ThreadPool.cpp)
target_link_libraries(simple_test Threads::Threads)
