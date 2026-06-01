cmake_minimum_required(VERSION 3.18)
project(aotinductor_inference LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Download LibTorch from https://pytorch.org/get-started/locally/
# and set: cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
find_package(Torch REQUIRED)

add_executable(inference main.cu)
target_link_libraries(inference ${TORCH_LIBRARIES})

set_property(TARGET inference PROPERTY CXX_STANDARD 17)
foreach(flag ${TORCH_CXX_FLAGS})
    target_compile_options(inference PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${flag}>)
endforeach()
