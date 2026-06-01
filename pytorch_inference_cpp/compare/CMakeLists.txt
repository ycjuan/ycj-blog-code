cmake_minimum_required(VERSION 3.18)
project(compare_backends CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# --- LibTorch (TorchScript backend) ---
find_package(Torch REQUIRED)

# --- ONNX Runtime ---
set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to ONNX Runtime root")
if(ONNXRUNTIME_ROOT STREQUAL "")
    message(FATAL_ERROR "Set -DONNXRUNTIME_ROOT=/path/to/onnxruntime")
endif()

# --- TensorRT (system-installed via dnf) ---
find_package(CUDAToolkit REQUIRED)

add_executable(compare
    main.cu
    backend_torchscript.cu
    backend_onnxruntime.cu
    backend_tensorrt.cu
    backend_cuda.cu
)

target_include_directories(compare PRIVATE
    ${ONNXRUNTIME_ROOT}/include
)

target_link_directories(compare PRIVATE
    ${ONNXRUNTIME_ROOT}/lib
)

target_link_libraries(compare
    ${TORCH_LIBRARIES}
    onnxruntime
    nvinfer
    nvonnxparser
    CUDA::cublas
    CUDA::cudart
)

# LibTorch requires these flags
set_property(TARGET compare PROPERTY CXX_STANDARD 17)
foreach(flag ${TORCH_CXX_FLAGS})
    target_compile_options(compare PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${flag}>)
endforeach()
