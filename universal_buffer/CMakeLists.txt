cmake_minimum_required(VERSION 3.18)

# Library name
set(LIB_NAME universalbuffer)

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
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda_malloc_raii.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/src/universal_buffer.cuh
)

#set(LIB_SOURCE_FILES
#)

add_library(${LIB_NAME} SHARED
    ${LIB_HEADER_FILES}
#    ${LIB_SOURCE_FILES}
)

# Include directories
target_include_directories(${LIB_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Link CUDA libraries
target_link_libraries(${LIB_NAME}
    ${CUDA_LIBRARIES}
    cublas
    gomp
)

# Compiler flags
target_compile_options(${LIB_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fopenmp>
)

# Add executables
add_executable(test_cuda_malloc_raii test/test_cuda_malloc_raii.cu)
target_include_directories(test_cuda_malloc_raii PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(test_cuda_malloc_raii ${LIB_NAME})

add_executable(test_universal_buffer test/test_universal_buffer.cu)
target_include_directories(test_universal_buffer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(test_universal_buffer ${LIB_NAME})

# Set linker language to CUDA
set_target_properties(${LIB_NAME} PROPERTIES LINKER_LANGUAGE CUDA)
