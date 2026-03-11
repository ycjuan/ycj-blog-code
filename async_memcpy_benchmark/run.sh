#!/bin/bash

set -e

if [[ "$1" == "-a" ]]; then
    rm -rf build
fi
mkdir -p build
cd build
cmake ..
make -j $(nproc)
./test_async_memcpy_benchmark