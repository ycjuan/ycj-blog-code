cmake_minimum_required(VERSION 3.18)
project(onnxruntime_inference LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Download ONNX Runtime from https://github.com/microsoft/onnxruntime/releases
# and set: cmake -DONNXRUNTIME_ROOT=/path/to/onnxruntime ..
set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to ONNX Runtime root directory")

if (NOT ONNXRUNTIME_ROOT)
    message(FATAL_ERROR "ONNXRUNTIME_ROOT must be set to the ONNX Runtime install root (e.g. /path/to/onnxruntime)")
endif()

add_executable(inference main.cu)
target_include_directories(inference PRIVATE ${ONNXRUNTIME_ROOT}/include)
target_link_directories(inference PRIVATE ${ONNXRUNTIME_ROOT}/lib)
target_link_libraries(inference onnxruntime)
