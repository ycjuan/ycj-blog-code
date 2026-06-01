#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "Step 1: Export model"
echo "============================================================"
python3 export.py

echo ""
echo "============================================================"
echo "Step 2: TorchScript"
echo "============================================================"
cd torchscript && ./compile.sh && ./run.sh
cd "$SCRIPT_DIR"

echo ""
echo "============================================================"
echo "Step 3: ONNX Runtime"
echo "============================================================"
cd onnxruntime && ./compile.sh && ./run.sh
cd "$SCRIPT_DIR"

echo ""
echo "============================================================"
echo "Step 4: TensorRT"
echo "============================================================"
cd tensorrt && ./compile.sh && ./run.sh
cd "$SCRIPT_DIR"

echo ""
echo "============================================================"
echo "Step 5: Pure CUDA"
echo "============================================================"
cd cuda && ./compile.sh && ./run.sh
cd "$SCRIPT_DIR"

echo ""
echo "============================================================"
echo "Step 6: Compare all backends + benchmark"
echo "============================================================"
cd compare && ./compile.sh && ./run.sh
cd "$SCRIPT_DIR"
