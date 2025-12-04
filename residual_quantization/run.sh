#!/bin/bash

set -e

#rm -rf build
#mkdir -p build
cd build
cmake ..
make -j 4
./test_residual_quantization
#./test_residual_quantization_unit_tests