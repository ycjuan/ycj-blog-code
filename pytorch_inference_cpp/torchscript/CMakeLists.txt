cmake_minimum_required(VERSION 3.18)
project(torchscript_inference)

set(CMAKE_CXX_STANDARD 17)

# Download LibTorch from https://pytorch.org/get-started/locally/
# and set: cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
find_package(Torch REQUIRED)

add_executable(inference main.cu)
target_link_libraries(inference ${TORCH_LIBRARIES})
