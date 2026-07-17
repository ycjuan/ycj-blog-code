#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# JAX_PLATFORMS only controls which device JAX itself uses while tracing/exporting
# the model in Step 1 (export.py) — it has no effect on which backends run in
# Steps 2-5, since those C++ binaries always build and run their own CPU/GPU
# code paths against the exported model files (model.onnx, .vmfb, weights/*.bin).
#
# Hardcoded to "cpu" because:
#   - the system python3 (no venv needed) is sufficient — JAX's PRNG is
#     device-independent, so exported weights/artifacts are bit-identical
#     whether traced on CPU or GPU
#   - this model is tiny, so CPU tracing is effectively free
#
# Switch to "cuda" (and use venv311's python, which has jax[cuda12] — see
# README) only if:
#   - you're validating that the venv311 CUDA-enabled jax install itself works
#   - the model becomes large enough that CPU tracing/compilation is slow
JAX_PLATFORM=cpu
PYTHON=python3

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
