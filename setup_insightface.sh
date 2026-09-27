#!/bin/bash
set -e

echo "=== 1. Creating Python Virtual Environment ==="
# InsightFace requires Python 3.10+ and NumPy 1.x
python3 -m venv insight_env
source insight_env/bin/activate

echo "=== 2. Upgrading Pip and Core Dependencies ==="
pip install --upgrade pip setuptools wheel
# InsightFace and ONNX models require NumPy < 2.0 to prevent compatibility errors
pip install "numpy<2" opencv-python tqdm

echo "=== 3. Installing InsightFace ==="
pip install insightface

echo "=== 3b. Swapping the CPU onnxruntime wheel for the MIGraphX build ==="
# `pip install insightface` pulls the CPU-only `onnxruntime` wheel from PyPI.
# It has to go: that wheel's C extension is named
# onnxruntime_pybind11_state.cpython-<ver>-x86_64-linux-gnu.so, and CPython's
# import machinery prefers the tagged name over the plain
# onnxruntime_pybind11_state.so shipped by the MIGraphX wheel. The result is a
# CPU-only extension that loads while the Python package reports 1.31.0, so
# MIGraphXExecutionProvider never registers and every Session silently falls
# back to CPU.
MIGRAPHX_WHEEL=$(ls -1 ../onnxruntime/onnxruntime/build/Linux/Release/dist/onnxruntime_migraphx-*.whl 2>/dev/null | head -n 1)
if [ -z "$MIGRAPHX_WHEEL" ]; then
    echo "ERROR: no onnxruntime_migraphx-*.whl found under ../onnxruntime/onnxruntime/build/Linux/Release/dist/" >&2
    echo "       Build it first (./build.sh), then re-run this script." >&2
    exit 1
fi
echo "Using wheel: $MIGRAPHX_WHEEL"
pip uninstall -y onnxruntime
# --no-deps: the runtime deps (numpy, flatbuffers, packaging, protobuf) are
# already satisfied above, and letting pip resolve them risks pulling the CPU
# onnxruntime straight back in.
pip install --no-deps --force-reinstall "$MIGRAPHX_WHEEL"

echo "=== 3c. Verifying the MIGraphX provider is active ==="
python - <<'PY'
import sys
import onnxruntime as ort
providers = ort.get_available_providers()
print("onnxruntime:", ort.__version__)
print("providers  :", providers)
if "MIGraphXExecutionProvider" not in providers:
    print("ERROR: MIGraphXExecutionProvider is not available", file=sys.stderr)
    sys.exit(1)
PY

echo "=== 4. Setting up Local Model Directories ==="
# InsightFace looks for its models inside ~/.insightface/models/ by default
MODEL_DIR="$HOME/.insightface/models"
mkdir -p "$MODEL_DIR"

echo "=== 5. Fetching the Default 'buffalo_l' Model Pack ==="
# We download the official model zip archive manually to skip unpredictable runtime downloads
cd "$MODEL_DIR"
if [ ! -f "buffalo_l.zip" ]; then
    echo "Downloading buffalo_l.zip..."
    curl -L -o buffalo_l.zip "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
else
    echo "buffalo_l.zip already exists. Skipping download."
fi

echo "=== 6. Extracting Models ==="
# Unzip creates the folder structure required by the FaceAnalysis API
unzip -o buffalo_l.zip -d buffalo_l

echo "=== Setup Complete ==="
echo "To begin your hobby project, activate your environment with:"
echo "source $(pwd)/../../insight_env/bin/activate"

