#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DEVICE="${1:-cpu}"
case "$DEVICE" in
    cpu|gpu) ;;
    *) echo "Usage: $0 [cpu|gpu]" >&2; exit 1 ;;
esac

# GPU export needs jax[cuda12], which lives in the venv311 virtualenv (see README);
# CPU export works with the system python3.
PYTHON=python3
JAX_PLATFORM=cpu
if [ "$DEVICE" = "gpu" ]; then
    PYTHON=~/external/venv311/bin/python3
    JAX_PLATFORM=cuda
fi

echo "============================================================"
echo "Step 1: Export model (JAX_PLATFORMS=$JAX_PLATFORM)"
echo "============================================================"
JAX_PLATFORMS=$JAX_PLATFORM $PYTHON export.py

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
