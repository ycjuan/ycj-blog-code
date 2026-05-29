cmake_minimum_required(VERSION 3.18)
project(tensorrt_inference)

set(CMAKE_CXX_STANDARD 17)

# TensorRT is typically installed at /usr/local/tensorrt or via TensorRT tar package.
# Set: cmake -DTENSORRT_ROOT=/path/to/tensorrt ..
set(TENSORRT_ROOT "" CACHE PATH "Path to TensorRT root directory")

find_package(CUDA REQUIRED)

add_executable(inference main.cpp)
target_include_directories(inference PRIVATE
    ${TENSORRT_ROOT}/include
    ${CUDA_INCLUDE_DIRS}
)
target_link_directories(inference PRIVATE ${TENSORRT_ROOT}/lib)
target_link_libraries(inference
    nvinfer
    nvonnxparser
    ${CUDA_LIBRARIES}
)
