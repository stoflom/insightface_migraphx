# InsightFace with AMD GPU (MIGraphX)

Face detection, recognition, and embedding extraction using [InsightFace](https://github.com/deepinsight/insightface), accelerated on AMD GPUs via a custom **MIGraphX** build of ONNX Runtime. Includes a digiKam tagger that writes face regions into image metadata.

## Repository layout

| File | Purpose |
|---|---|
| `setup_insightface.sh` | One-shot setup: venv, dependencies, MIGraphX onnxruntime swap, model download |
| `digikam_migraphx_tagger.py` | Detects faces and writes them as digiKam-readable MWG Region Info XMP via ExifTool |
| `INSIGHT.md` | Minimal example: detect faces and get 512-d embeddings |
| `test_insightface.py` | Smoke test using InsightFace's built-in demo image |
| `insight_env/` | Python virtual environment (created by the setup script) |

## Licensing — important

The `buffalo_l` model pack (and all other pretrained models distributed by the
[InsightFace project](https://github.com/deepinsight/insightface)) is licensed
**for non-commercial use only** (research and development).

Consequences for this project:

- Using it for personal/hobby work (e.g. indexing your own photo library) is
  fine.
- You may **not** use it in commercial products or services, or redistribute
  the model files (`~/.insightface/models/buffalo_l/`, `buffalo_l.zip`) in a
  product.
- For a commercial face-recognition pipeline you'd need models under a
  permissive license or a commercial license from the model provider.

The Python code of the InsightFace repository itself is MIT-licensed; the
restriction applies to the **pretrained model weights**.

## Requirements

- Python 3.10+
- NumPy 1.x (InsightFace is not compatible with NumPy 2.0)
- An AMD GPU with ROCm, and a locally built MIGraphX onnxruntime wheel at
  `../onnxruntime/onnxruntime/build/Linux/Release/dist/onnxruntime_migraphx-*.whl`
  (see [Building the MIGraphX onnxruntime](#building-the-migraphx-onnxruntime) if you don't have one yet)

## Setup

```bash
./setup_insightface.sh
```

The script:

1. Creates the `insight_env` virtual environment
2. Installs dependencies (`numpy<2`, `opencv-python`, `tqdm`) and `insightface`
3. **Replaces** the CPU-only `onnxruntime` wheel with the local MIGraphX build
4. Verifies that `MIGraphXExecutionProvider` is registered
5. Downloads and extracts the `buffalo_l` model pack to `~/.insightface/models/`

## Using the environment

```bash
source insight_env/bin/activate
# to leave the environment later:
deactivate
```

The `activate` script also exports `ORT_MIGRAPHX_MODEL_CACHE_PATH` (see
[MIGraphX compile cache](#migraphx-compile-cache)) — don't skip it.

### Run the smoke test

```bash
python test_insightface.py
```

This runs `FaceAnalysis` on a built-in demo image and prints the active
execution providers. Confirm `MIGraphXExecutionProvider` is in the list.

### Tag images for digiKam

```bash
sudo dnf install perl-Image-ExifTool   # Fedora/RHEL
python digikam_migraphx_tagger.py
```

**Note on coordinate systems:** to write face regions that digiKam reads
natively, the tags must follow the Metadata Working Group (MWG) Region Info
Schema, embedded in the image's standard XMP block. One key difference:
InsightFace reports absolute pixel coordinates `[xmin, ymin, xmax, ymax]`,
whereas digiKam stores normalized fractions `[x, y, width, height]` in
`0.0–1.0`. ExifTool is used for the XMP writes so files are not corrupted by
naive text editing.

### Get face embeddings (example)

See [`INSIGHT.md`](INSIGHT.md) for a minimal example. Each detected face
yields a 512-d embedding that can be stored in a lightweight vector database
(SQLite, FAISS, Milvus, ...) for cross-file matching outside any photo app.

## Upgrading InsightFace

```bash
pip install -U insightface
```

This will reinstall the CPU-only `onnxruntime` wheel and **break the MIGraphX
GPU path**. Re-run the swap afterwards (or re-run `./setup_insightface.sh`).

## MIGraphX notes

### Building the MIGraphX onnxruntime

The GPU path is the **MIGraphX execution provider**, built from source. The
system libraries live in `/usr/local/lib64`:

```bash
ls /usr/local/lib64/libonnxruntime.so*                          # e.g. libonnxruntime.so.1.31.0
ls /usr/local/lib64/libonnxruntime_providers_migraphx.so
```

The venv must use the locally built wheel, which **replaces** the CPU-only
`onnxruntime` wheel that `pip install insightface` pulls in:

```bash
source insight_env/bin/activate
pip uninstall -y onnxruntime
pip install --no-deps --force-reinstall \
    ../onnxruntime/onnxruntime/build/Linux/Release/dist/onnxruntime_migraphx-*.whl
```

`--no-deps` matters: without it pip re-resolves dependencies and can
reinstall the CPU `onnxruntime` wheel, undoing the swap.

**Why the swap is needed at all:** the PyPI wheel's C extension is named
`onnxruntime_pybind11_state.cpython-<ver>-x86_64-linux-gnu.so`, and CPython's
import machinery prefers that tagged name over the plain
`onnxruntime_pybind11_state.so` from the MIGraphX wheel. The result is a
CPU-only extension loading while the package still reports 1.31.0, so
`MIGraphXExecutionProvider` never registers and every Session silently runs
on CPU.

### Verifying the provider

```bash
python -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers(), ort.get_device())"
# 1.31.0 ['MIGraphXExecutionProvider', 'CPUExecutionProvider'] CPU-MIGRAPHX
```

Then select the provider in your scripts:

```python
import insightface
from insightface.app import FaceAnalysis

def get_face_analyzer():
    """Initialize InsightFace with explicit AMD MIGraphX provider ordering."""
    providers = [
        'MIGraphXExecutionProvider',  # high-performance AMD graph optimizer
        'CPUExecutionProvider',       # ultimate fallback
    ]
    app = FaceAnalysis(name='buffalo_l', providers=providers)
    app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 = first GPU
    return app
```

**There is no `ROCMExecutionProvider` in this build.** Asking for one does not
fail loudly — ONNX Runtime logs an `EP Error` and quietly runs on CPU. Always
check `ort.get_available_providers()` rather than assuming acceleration.

### MIGraphX compile cache

MIGraphX compiles each graph with the ROCm compiler, which takes minutes on
the first run. Sourcing `insight_env/bin/activate` exports
`ORT_MIGRAPHX_MODEL_CACHE_PATH` (`~/.cache/migraphx`), which caches the
compiled `.mxr` files, so only the first run is slow (~149 s cold vs ~3.6 s
warm).

This is a **correctness** requirement, not just a speed one. The 1.31
MIGraphX EP reports its default `migraphx_model_cache_dir` as the two-character
string `""` (literal double quotes), and InsightFace copies provider options
from its reference Session onto the per-resolution static Sessions it creates.
That bogus quoted path is therefore passed back in, and every static Session
dies with:

```
RuntimeException: 6 : RUNTIME_EXCEPTION : Exception during initialization:
Failed to call function
```

Pointing the EP at a real directory via `ORT_MIGRAPHX_MODEL_CACHE_PATH`
bypasses the broken default. If you ever run without sourcing `activate`,
either export that variable yourself or pass `static_shape_sessions=False` to
`FaceAnalysis` so the reference Session is reused instead of cloned.
