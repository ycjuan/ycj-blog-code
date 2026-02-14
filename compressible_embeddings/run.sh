#!/bin/bash

set -e

#rm -rf build
mkdir -p build
cd build
cmake ..
make -j $(nproc)
./test_compressible_embeddings