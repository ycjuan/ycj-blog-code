cmake_minimum_required(VERSION 3.18)
project(cuda_inference CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

add_executable(inference main.cu)
target_link_libraries(inference CUDA::cublas)
