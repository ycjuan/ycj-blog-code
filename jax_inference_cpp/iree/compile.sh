#!/usr/bin/env bash
set -e
mkdir -p build && cd build
cmake .. -DIREE_ROOT=~/external/iree-311-cuda
cmake --build . -j"$(nproc)"
