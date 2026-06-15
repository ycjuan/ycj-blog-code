cmake_minimum_required(VERSION 3.18)
project(jax_compare_backends CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# --- ONNX Runtime ---
set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to ONNX Runtime root")
if(ONNXRUNTIME_ROOT STREQUAL "")
    message(FATAL_ERROR "Set -DONNXRUNTIME_ROOT=/path/to/onnxruntime")
endif()

# --- IREE runtime (from iree-dist) ---
set(IREE_ROOT "" CACHE PATH "Path to IREE dist root (contains include/ and lib/)")
if(IREE_ROOT STREQUAL "")
    message(FATAL_ERROR "Set -DIREE_ROOT=/path/to/iree-dist")
endif()

find_package(CUDAToolkit REQUIRED)

# Required by iree/base/allocator.h to expose iree_allocator_system()
add_compile_definitions(IREE_ALLOCATOR_SYSTEM_CTL=iree_allocator_libc_ctl)

add_executable(compare
    main.cu
    backend_onnxruntime.cu
    backend_iree.cu
    backend_cuda.cu
)

target_include_directories(compare PRIVATE
    ${ONNXRUNTIME_ROOT}/include
    ${IREE_ROOT}/include
)

target_link_directories(compare PRIVATE
    ${ONNXRUNTIME_ROOT}/lib
    ${IREE_ROOT}/lib
)

target_link_libraries(compare
    onnxruntime
    iree_runtime_unified
    flatcc_parsing
    flatcc_runtime
    CUDA::cublas
    CUDA::cudart
    pthread
    dl
    m
)
