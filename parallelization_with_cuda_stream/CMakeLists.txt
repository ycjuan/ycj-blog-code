cmake_minimum_required(VERSION 3.18)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(parallelization_with_cuda_stream LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)

# Add executable
add_executable(parallelization_with_cuda_stream main.cu)

# Set CUDA architecture
set_property(TARGET parallelization_with_cuda_stream PROPERTY CUDA_ARCHITECTURES 89)

# Include directories
target_include_directories(parallelization_with_cuda_stream PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Link CUDA libraries
target_link_libraries(parallelization_with_cuda_stream 
    ${CUDA_LIBRARIES}
    cublas
)

# Compiler flags
target_compile_options(parallelization_with_cuda_stream PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)
