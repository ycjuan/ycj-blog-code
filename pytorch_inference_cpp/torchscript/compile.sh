#!/bin/bash
set -e

LIBTORCH_ROOT=$1

if [ -z "$LIBTORCH_ROOT" ]; then
    echo "Usage: ./compile.sh /path/to/libtorch"
    exit 1
fi

mkdir -p build && cd build
cmake .. -DCMAKE_PREFIX_PATH=$LIBTORCH_ROOT
make -j$(nproc)
