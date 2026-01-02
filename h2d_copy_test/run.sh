#!/bin/bash

set -e

#rm -rf build
mkdir -p build
cd build
cmake ..
make -j 4
./test_h2d_copy_test