#!/usr/bin/env bash
set -e
# cuDNN is bundled with jax[cuda12]; add its lib dir so ORT's CUDA EP can find it
CUDNN_LIB=~/external/venv311/lib/python3.11/site-packages/nvidia/cudnn/lib
LD_LIBRARY_PATH=~/external/onnxruntime-linux-x64-gpu-1.26.0/lib:${CUDNN_LIB}:$LD_LIBRARY_PATH \
    ./build/compare
