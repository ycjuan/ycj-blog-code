#!/bin/bash

set -e

#rm -rf build
mkdir -p build
cd build
cmake ..
make -j 4
./test1
./test2
./test3