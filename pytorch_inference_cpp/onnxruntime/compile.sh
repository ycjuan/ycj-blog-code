#!/bin/bash
set -e

ONNXRUNTIME_ROOT=$1

if [ -z "$ONNXRUNTIME_ROOT" ]; then
    echo "Usage: ./compile.sh /path/to/onnxruntime"
    exit 1
fi

mkdir -p build && cd build
cmake .. -DONNXRUNTIME_ROOT=$ONNXRUNTIME_ROOT
make -j$(nproc)
