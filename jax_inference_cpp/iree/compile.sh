#!/usr/bin/env bash
set -e
mkdir -p build && cd build
cmake .. -DIREE_ROOT=~/external
cmake --build . -j"$(nproc)"
