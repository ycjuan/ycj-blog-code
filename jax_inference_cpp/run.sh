#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "Step 1: Export model"
echo "============================================================"
python3 export.py

echo ""
echo "============================================================"
echo "Step 2: ONNX Runtime"
echo "============================================================"
cd onnxruntime && ./compile.sh && ./run.sh
cd "$SCRIPT_DIR"

echo ""
echo "============================================================"
echo "Step 3: IREE"
echo "============================================================"
cd iree && ./compile.sh && ./run.sh
cd "$SCRIPT_DIR"

echo ""
echo "============================================================"
echo "Step 4: Pure CUDA"
echo "============================================================"
cd cuda && ./compile.sh && ./run.sh
cd "$SCRIPT_DIR"

echo ""
echo "============================================================"
echo "Step 5: Compare all backends + benchmark"
echo "============================================================"
cd compare && ./compile.sh && ./run.sh
cd "$SCRIPT_DIR"
