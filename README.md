To run insightface start the environment
```bash
source insight_env/bin/activate
```
then run the indexer

```bash
python face_indexer.oy
```

or use the didgiKam type tagger to update digiKam using the perl tool

```bash
sudo dnf install perl-Image-ExifTool   # Fedora/RHEL
```

then

```bash
python digikam_migraphx_tagger.py
```

Note:
To write face region rectangles that digiKam can read natively, the tags must follow the Metadata Working Group (MWG) Region Info Schema.DigiKam embeds these directly inside the image's embedded XMP standard block. A key difference to watch out for is coordinates: InsightFace uses standard absolute pixel coordinates [xmin, ymin, xmax, ymax], whereas digiKam stores coordinates as relative normalized fractions [x, y, width, height] ranging strictly from 0.0 to 1.0.The standard approach for automating this on Linux is utilizing ExifTool, which handles XMP parsing without risking file corruption.


**MIGraphX
The GPU path here is the **MIGraphX** execution provider, built from source and
installed into `/usr/local/lib64`:

```bash
ls /usr/local/lib64/libonnxruntime.so*            # libonnxruntime.so.1.31.0
ls /usr/local/lib64/libonnxruntime_providers_migraphx.so
```

The venv gets it from the locally built wheel, which must **replace** the
CPU-only `onnxruntime` wheel that `pip install insightface` pulls in:

```bash
source insight_env/bin/activate
pip uninstall -y onnxruntime
pip install --no-deps --force-reinstall \
    ../onnxruntime/onnxruntime/build/Linux/Release/dist/onnxruntime_migraphx-*.whl
```

`--no-deps` matters: without it pip re-resolves dependencies and can reinstall
the CPU `onnxruntime` wheel, undoing the swap.

Check it took effect:

```bash
python -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers(), ort.get_device())"
# 1.31.0 ['MIGraphXExecutionProvider', 'CPUExecutionProvider'] CPU-MIGRAPHX
```

Then set the provider in the scripts:

```python
import insightface
from insightface.app import FaceAnalysis

def get_face_analyzer():
    """
    Initializes InsightFace with explicit AMD MIGraphX execution provider ordering.
    """
    # 1. Provide the exact execution sequence string for the MIGraphX engine
    providers = [
        'MIGraphXExecutionProvider',  # High Performance AMD Graph Optimizer
        'CPUExecutionProvider'        # Ultimate fallback
    ]

    print("🚀 Initializing InsightFace on AMD GPU with MIGraphX Acceleration...")
    app = FaceAnalysis(name='buffalo_l', providers=providers)

    # 2. Prepare context (ctx_id=0 maps to your first available GPU index)
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app
```

Note: there is no `ROCMExecutionProvider` in this build. Asking for it does not
fail loudly — ONNX Runtime logs an `EP Error` and quietly runs on CPU, so
check `get_available_providers()` rather than assuming acceleration.

**MIGraphX compile cache
MIGraphX compiles each graph with the ROCm compiler, which takes minutes on the
first run. `insight_env/bin/activate` exports `ORT_MIGRAPHX_MODEL_CACHE_PATH`
(`~/.cache/migraphx`) to cache the compiled `.mxr` files, so only the first run
is slow — about 149s cold versus 3.6s warm.

This is also a correctness requirement, not just a speed one. The 1.31 MIGraphX
EP reports its default `migraphx_model_cache_dir` as the two-character string
`""` (literal double quotes), and InsightFace copies provider options from its
reference Session onto the per-resolution static Sessions it creates. That
bogus quoted path is therefore passed back in, and every static Session dies
with:

```
RuntimeException: 6 : RUNTIME_EXCEPTION : Exception during initialization:
Failed to call function
```

Pointing the EP at a real directory via `ORT_MIGRAPHX_MODEL_CACHE_PATH` bypasses
the broken default. If you ever run without sourcing `activate`, either export
that variable yourself or pass `static_shape_sessions=False` to `FaceAnalysis`
so the reference Session is reused instead of cloned.

