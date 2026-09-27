import cv2
import onnxruntime  #only for printing the available providers below
import insightface
from insightface.app import FaceAnalysis

# Ensure the environment is activated before running this script
# Initialize FaceAnalysis to pull the local 'buffalo_l' pack you just downloaded
print("Initializing FaceAnalysis framework...")
app = FaceAnalysis(name='buffalo_l', providers=[
    'MIGraphXExecutionProvider',  # High Performance AMD Graph Optimizer
    'CPUExecutionProvider'
    ])

print(onnxruntime.get_available_providers())


# Prepare the model context (ctx_id=0 points to your primary AMD GPU index)
app.prepare(ctx_id=0, det_size=(640, 640))

# Fetch a built-in demo sample image from the library to test the framework
print("Running a smoke test on sample data...")
from insightface.data import get_image
img = get_image('t1')

# Detect faces and generate your independent 512-dimensional face vectors
faces = app.get(img)

print(f"\n--- Success! Detected {len(faces)} face(s) ---")
for i, face in enumerate(faces):
    print(f"Face #{i+1}:")
    print(f"  - Bounding Box: {face.bbox.astype(int)}")
    print(f"  - Vector Shape: {face.embedding.shape} (512-dimensional signature)")
