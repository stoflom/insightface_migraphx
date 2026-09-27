# InsightFace embeddings — minimal example

Detect faces in an image and extract their 512-d **face embeddings**
("signatures"). These vectors can be stored in a lightweight vector database
and used to match the same person across an entire photo library — entirely
outside any photo-editing application.

> Run this inside the project venv (`source insight_env/bin/activate`),
> which also exports `ORT_MIGRAPHX_MODEL_CACHE_PATH` (see
> [README](README.md#migraphx-compile-cache)).

## Detect and extract

```python
import cv2
import onnxruntime
import insightface
from insightface.app import FaceAnalysis

# Verify the GPU provider is actually registered before assuming acceleration.
# There is no 'ROCMExecutionProvider' in this build — the correct name is
# 'MIGraphXExecutionProvider'.
print(onnxruntime.get_available_providers())

# 1. Initialize the analysis engine
app = FaceAnalysis(
    name='buffalo_l',
    providers=['MIGraphXExecutionProvider', 'CPUExecutionProvider'],
)
app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 = first GPU

# 2. Load your image
img = cv2.imread('your_photo.jpg')
assert img is not None, 'image not found'

# 3. Detect faces and extract signatures
for face in app.get(img):
    embedding = face.embedding          # 512-d float vector, the face signature
    bbox = face.bbox.astype(int)        # [xmin, ymin, xmax, ymax] in pixels
    print(f'Face at {bbox}, embedding shape: {embedding.shape}')

    # Store (embedding, image path, bbox) in your database here.
```

## Matching faces

InsightFace embeddings are already **L2-normalized**, so plain dot product is
cosine similarity. Two faces of the same person typically score ≈ 0.5–0.8;
different people usually score well below ~0.3. A threshold around **0.4** is
a reasonable starting point — tune it on your own library.

```python
import numpy as np

def same_person(a: np.ndarray, b: np.ndarray, threshold: float = 0.4) -> bool:
    """a and b are 512-d InsightFace embeddings."""
    return float(np.dot(a, b)) >= threshold
```

## Where to store them

- **SQLite** — simple: store the embedding as a BLOB (`embedding.tobytes()`)
  or JSON, plus image path and bbox. Fine for tens of thousands of faces.
- **FAISS** — in-memory ANN index for fast nearest-neighbour search over large
  sets.
- **Milvus / Qdrant** — full-featured vector databases if the index needs to
  be shared or queried by a service.

For per-identity grouping inside digiKam itself, prefer the
`digikam_migraphx_tagger.py` in this repo, which writes the face regions
directly into image metadata.
