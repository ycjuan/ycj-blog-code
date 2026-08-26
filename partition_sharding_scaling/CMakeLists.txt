cmake_minimum_required(VERSION 3.18)

# Library name
set(LIB_NAME partition_sharding_scaling)

# Export compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Project name
project(${LIB_NAME} LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)
find_package(Threads REQUIRED)

# Add library
file(GLOB_RECURSE LIB_HEADER_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/common/include/*.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/method_a_macro_partition/include/*.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/method_b_shared_dispatcher/include/*.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/method_c_unified/include/*.cuh
)
file(GLOB_RECURSE LIB_SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/common/src/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/method_a_macro_partition/src/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/method_b_shared_dispatcher/src/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/method_c_unified/src/*.cu
)

add_library(${LIB_NAME} SHARED
    ${LIB_HEADER_FILES}
    ${LIB_SOURCE_FILES}
)

# Set CUDA architecture
set_property(TARGET ${LIB_NAME} PROPERTY CUDA_ARCHITECTURES 75 80 86 89 90)

# Include directories - one per method folder, so each method's headers can be included by name
# (e.g. #include "block_manager.cuh") without needing to know the folder layout.
target_include_directories(${LIB_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/common/include
    ${CMAKE_CURRENT_SOURCE_DIR}/method_a_macro_partition/include
    ${CMAKE_CURRENT_SOURCE_DIR}/method_b_shared_dispatcher/include
    ${CMAKE_CURRENT_SOURCE_DIR}/method_c_unified/include
)

# Link CUDA libraries
target_link_libraries(${LIB_NAME}
    ${CUDA_LIBRARIES}
    cublas
    Threads::Threads
)

# Compiler flags
target_compile_options(${LIB_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

# Add executable
add_executable(test_${LIB_NAME} test/main.cu)

# Link library
target_link_libraries(test_${LIB_NAME} ${LIB_NAME} Threads::Threads)

# Set linker language to CUDA
set_target_properties(${LIB_NAME} PROPERTIES LINKER_LANGUAGE CUDA)
