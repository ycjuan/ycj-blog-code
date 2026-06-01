#!/bin/bash
set -e

LIBTORCH_ROOT=~/external/libtorch

mkdir -p build && cd build
cmake .. -DCMAKE_PREFIX_PATH=$LIBTORCH_ROOT
make -j$(nproc)
