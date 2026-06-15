#!/usr/bin/env bash
set -e
mkdir -p build && cd build
cmake .. \
    -DONNXRUNTIME_ROOT=~/external/onnxruntime-linux-x64-gpu-1.26.0 \
    -DIREE_ROOT=~/external \
    -DTorch_DIR=~/external/libtorch/share/cmake/Torch
cmake --build . -j"$(nproc)"
