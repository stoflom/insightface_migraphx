# InsightFace with AMD GPU (MIGraphX)

Experimental local prototype for face detection, recognition, and embedding extraction using [InsightFace](https://github.com/deepinsight/insightface), accelerated on AMD GPUs via a custom **MIGraphX** build of ONNX Runtime.

This project is in an early stage: the setup is intentionally specific to one AMD/ROCm environment, the workflows are still being validated, and some features may change as the stack evolves. The digiKam tagger is included as a prototype and should be used cautiously with backups in place.

## Current status

- Early-stage proof-of-concept: not a polished end-user installation
- Designed for a local AMD GPU + ROCm + custom MIGraphX ONNX Runtime environment
- Useful for smoke testing and experimentation, especially around provider selection and model caching
- Some features are still exploratory or unverified, especially metadata output and broader automation

## Repository layout

| File | Purpose |
|---|---|
| `setup_insightface.sh` | One-shot setup: venv, dependencies, MIGraphX onnxruntime swap, model download |
| `digikam_migraphx_tagger.py` | Detects faces and writes them as digiKam-readable MWG Region Info XMP via ExifTool |
| `INSIGHT.md` | Minimal example: detect faces and get 512-d embeddings |
| `test_insightface.py` | Smoke test using InsightFace's built-in demo image |
| `insight_env/` | Python virtual environment (created by the setup script) |
| `activate.custom` | Patched copy of the venv's `activate` script; the venv is always activated with this file |

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
- NumPy 2.x (insightface 0.7.3's dependency tree requires it)
- An AMD GPU with ROCm, and a locally built MIGraphX onnxruntime wheel at
  `../onnxruntime/onnxruntime/build/Linux/Release/dist/onnxruntime_migraphx-*.whl`
  (see [Building the MIGraphX onnxruntime](#building-the-migraphx-onnxruntime) if you don't have one yet)

## Setup

```bash
./setup_insightface.sh
```

The script:

1. Creates the `insight_env` virtual environment (kept **pristine**) and
   generates `activate.custom` from a fresh copy of its `activate` script,
   patched to export `ORT_MIGRAPHX_MODEL_CACHE_PATH`
2. Installs dependencies (`numpy`, `opencv-python-headless`, `tqdm`, `flatbuffers`, `rawpy`) and `insightface`
3. **Installs** the local MIGraphX onnxruntime build (removing a CPU-only
   `onnxruntime` wheel first, if one is present)
4. Verifies that `MIGraphXExecutionProvider` is registered
5. Downloads and extracts the `buffalo_l` model pack to `~/.insightface/models/`

## Using the environment

```bash
source activate.custom
# to leave the environment later:
deactivate
```

`activate.custom` is the patched copy of the venv's `activate` script and is
the only way to activate the environment: it exports
`ORT_MIGRAPHX_MODEL_CACHE_PATH` (see
[MIGraphX compile cache](#migraphx-compile-cache)). The venv's own
`insight_env/bin/activate` is left pristine and does **not** set that
variable, so sourcing it gives you the venv but **no compile cache** —
every run re-compiles the models (minutes instead of seconds). Don't use
it.

`activate.custom` starts life as a copy of `insight_env/bin/activate`, so it
may stop working if an upgrade changes the venv's activation script into an
incompatible form (e.g., a venv re-creation or Python upgrade regenerates it).
In that case, just re-run `./setup_insightface.sh`, which always re-copies
the current venv script and re-applies the patch.

### Run the smoke test

```bash
python test_insightface.py
```

This runs `FaceAnalysis` on a built-in demo image and prints the active
execution providers. Confirm `MIGraphXExecutionProvider` is in the list.

### Tag images for digiKam

```bash
sudo dnf install perl-Image-ExifTool   # Fedora/RHEL, one-time
python digikam_migraphx_tagger.py      # interactive: prompts for the folder to tag
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

insightface 0.7.3 does not declare `onnxruntime` as a dependency, so an
upgrade does **not** touch the MIGraphX wheel — the GPU path survives. Re-run
the provider verification below afterwards. (If a future insightface release
starts depending on `onnxruntime`, re-run `./setup_insightface.sh` to restore
the swap.)

## MIGraphX notes

### Building the MIGraphX onnxruntime

The GPU path is the **MIGraphX execution provider**, built from source, refer to  https://github.com/stoflom/onnxruntime . The
system libraries live in `/usr/local/lib64`:

```bash
ls /usr/local/lib64/libonnxruntime.so*                          # e.g. libonnxruntime.so.1.32.0
ls /usr/local/lib64/libonnxruntime_providers_migraphx.so
```

The venv must use the locally built wheel (insightface 0.7.3 does not pull in
`onnxruntime` itself; the only danger is a CPU-only wheel already present in
the venv):

```bash
source activate.custom
pip uninstall -y onnxruntime
pip install --no-deps --force-reinstall \
    ../onnxruntime/onnxruntime/build/Linux/Release/dist/onnxruntime_migraphx-*.whl
```

`--no-deps` keeps the install minimal: the wheel's only declared deps are
flatbuffers, numpy, packaging, and protobuf (all installed by the setup
script), so nothing it would pull in could undo the swap.

**Why the swap is needed at all:** the PyPI wheel's C extension is named
`onnxruntime_pybind11_state.cpython-<ver>-x86_64-linux-gnu.so`, and CPython's
import machinery prefers that tagged name over the plain
`onnxruntime_pybind11_state.so` from the MIGraphX wheel. The result is a
CPU-only extension loading while the package still reports a version, so
`MIGraphXExecutionProvider` never registers and every Session silently runs
on CPU.

### Verifying the provider

```bash
python -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers(), ort.get_device())"
# 1.32.0 ['MIGraphXExecutionProvider', 'CPUExecutionProvider'] CPU-MIGRAPHX
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
the first run. Sourcing `activate.custom` exports
`ORT_MIGRAPHX_MODEL_CACHE_PATH` (`~/.cache/migraphx`), which caches the
compiled `.mxr` files, so only the first run is slow (~149 s cold vs ~3.6 s
warm).

Without the variable the EP falls back to its built-in default for
`migraphx_model_cache_dir`, which is the two-character string `""` (literal
double quotes — visible in the `Applied providers` options InsightFace
prints). The EP then caches nothing, so **every** run pays the full compile
cost. On the older stack (onnxruntime 1.31 + insightface with
per-resolution static Sessions) that quoted default was also fatal: it was
copied onto the cloned Sessions and killed each one with
`RuntimeException: 6 : RUNTIME_EXCEPTION : Exception during initialization:
Failed to call function`. insightface 0.7.3 no longer clones Sessions, so
today this is a speed problem, not a correctness one.

If you ever run without sourcing `activate.custom`, export
`ORT_MIGRAPHX_MODEL_CACHE_PATH` yourself to keep warm runs fast.
