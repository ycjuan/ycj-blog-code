cmake_minimum_required(VERSION 3.18)

# Library name
set(LIB_NAME cabm)

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

# Add library
set(LIB_HEADER_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/cabm.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/data_struct.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/macro.cuh
)

set(LIB_SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/cabm_cpu.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cabm_gpu.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/data_struct.cu
)

add_library(${LIB_NAME} SHARED
    ${LIB_HEADER_FILES}
    ${LIB_SOURCE_FILES}
)

# Set CUDA architecture
set_property(TARGET ${LIB_NAME} PROPERTY CUDA_ARCHITECTURES 89)

# Include directories
target_include_directories(${LIB_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Link CUDA libraries
target_link_libraries(${LIB_NAME} 
    ${CUDA_LIBRARIES}
    cublas
)

# Compiler flags
target_compile_options(${LIB_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

# Add executable
add_executable(test1 test1.cu)
add_executable(test2 test2.cu)
add_executable(test3 test3.cu)

# Link library
target_link_libraries(test1 ${LIB_NAME})
target_link_libraries(test2 ${LIB_NAME})
target_link_libraries(test3 ${LIB_NAME})