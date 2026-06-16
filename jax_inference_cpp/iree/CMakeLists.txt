cmake_minimum_required(VERSION 3.18)
project(iree_inference LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Download IREE dist from https://github.com/iree-org/iree/releases
# and set: cmake -DIREE_ROOT=/path/to/iree-dist ..
set(IREE_ROOT "" CACHE PATH "Path to IREE dist root (contains include/ and lib/)")
if(IREE_ROOT STREQUAL "")
    message(FATAL_ERROR "Set -DIREE_ROOT=/path/to/iree-dist")
endif()

# Required by iree/base/allocator.h to expose iree_allocator_system()
add_compile_definitions(IREE_ALLOCATOR_SYSTEM_CTL=iree_allocator_libc_ctl)

add_executable(inference main.cu)
target_include_directories(inference PRIVATE ${IREE_ROOT}/include)
target_link_directories(inference PRIVATE ${IREE_ROOT}/lib)
target_link_libraries(inference
    iree_runtime_unified
    flatcc_parsing
    printf_printf
    pthread
    dl
    m
)
