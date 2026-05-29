#!/bin/bash
set -e

TENSORRT_ROOT=$1

if [ -z "$TENSORRT_ROOT" ]; then
    echo "Usage: ./compile.sh /path/to/tensorrt"
    exit 1
fi

mkdir -p build && cd build
cmake .. -DTENSORRT_ROOT=$TENSORRT_ROOT
make -j$(nproc)
