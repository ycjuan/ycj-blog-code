#!/bin/bash

set -e

#rm -rf build
mkdir -p build
cd build
cmake ..
make -j 4
./test_topk_with_bucket_sort