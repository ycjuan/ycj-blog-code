#!/bin/bash
set -e

LD_LIBRARY_PATH=~/external/onnxruntime-linux-x64-gpu-1.26.0/lib:$LD_LIBRARY_PATH ./build/inference
