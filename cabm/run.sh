#!/bin/bash

set -e

#rm -rf build
mkdir -p build
cd build
cmake ..
make
./test1
./test2