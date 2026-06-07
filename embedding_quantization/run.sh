#!/bin/bash

set -e

rm -rf build
mkdir -p build
cd build
cmake ..
make -j 4
./test_embedding_quantization_unit_tests
./test_embedding_quantization