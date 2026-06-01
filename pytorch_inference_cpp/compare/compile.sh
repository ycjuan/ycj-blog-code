#!/bin/bash
set -e

LIBTORCH_ROOT=~/external/libtorch
ONNXRUNTIME_ROOT=~/external/onnxruntime-linux-x64-gpu-1.26.0

mkdir -p build && cd build
cmake .. \
    -DCMAKE_PREFIX_PATH=$LIBTORCH_ROOT \
    -DONNXRUNTIME_ROOT=$ONNXRUNTIME_ROOT
make -j$(nproc)
