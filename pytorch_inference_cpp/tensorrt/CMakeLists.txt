cmake_minimum_required(VERSION 3.18)
project(tensorrt_inference LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDA REQUIRED)

add_executable(inference main.cu)
target_include_directories(inference PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(inference
    nvinfer
    nvonnxparser
    ${CUDA_LIBRARIES}
)
