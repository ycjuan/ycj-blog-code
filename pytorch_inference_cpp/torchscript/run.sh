#!/bin/bash
set -e

LD_LIBRARY_PATH=~/external/libtorch/lib:$LD_LIBRARY_PATH ./build/inference
