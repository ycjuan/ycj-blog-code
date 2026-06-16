#!/usr/bin/env bash
set -e
# cuDNN is bundled with the nvidia-cudnn pip package; add it if present
CUDNN_LIB=~/.local/lib/python3.9/site-packages/nvidia/cudnn/lib
LD_LIBRARY_PATH=~/external/onnxruntime-linux-x64-gpu-1.26.0/lib:${CUDNN_LIB}:$LD_LIBRARY_PATH \
    ./build/compare
