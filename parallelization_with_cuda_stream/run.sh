#!/bin/bash

mkdir build
cd build
cmake ..
make
./parallelization_with_cuda_stream