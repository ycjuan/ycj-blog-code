#!/bin/bash
set -e

ONNXRUNTIME_ROOT=~/external/onnxruntime-linux-x64-gpu-1.26.0

mkdir -p build && cd build
cmake .. -DONNXRUNTIME_ROOT=$ONNXRUNTIME_ROOT
make -j$(nproc)
