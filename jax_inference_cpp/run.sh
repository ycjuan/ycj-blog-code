#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DEVICE="${1:-cpu}"
case "$DEVICE" in
    cpu|gpu) ;;
    *) echo "Usage: $0 [cpu|gpu]" >&2; exit 1 ;;
esac

echo "============================================================"
echo "Step 1: Export model (JAX_PLATFORMS=$DEVICE)"
echo "============================================================"
JAX_PLATFORMS=$DEVICE python3 export.py

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
