cmake_minimum_required(VERSION 3.18)

# Library name
set(LIB_NAME h2d_copy_test)

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
    ${CMAKE_CURRENT_SOURCE_DIR}/util.cuh
)


#set(LIB_SOURCE_FILES
#    ${CMAKE_CURRENT_SOURCE_DIR}/data.cu
#    ${CMAKE_CURRENT_SOURCE_DIR}/methods.cu
#)

add_library(${LIB_NAME} SHARED
    ${LIB_HEADER_FILES}
    #${LIB_SOURCE_FILES}
)

# Set CUDA architecture
set_property(TARGET ${LIB_NAME} PROPERTY CUDA_ARCHITECTURES 89)

# Include directories
target_include_directories(${LIB_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

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

# Add executable
add_executable(test_${LIB_NAME} main.cu)


# Link library
target_link_libraries(test_${LIB_NAME} ${LIB_NAME})

# Set linker language to CUDA
set_target_properties(${LIB_NAME} PROPERTIES LINKER_LANGUAGE CUDA)