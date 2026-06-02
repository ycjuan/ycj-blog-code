#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for dir in torchscript onnxruntime tensorrt cuda aotinductor compare; do
    rm -rf "$SCRIPT_DIR/$dir/build"
done

echo "Done."
