import cv2
import insightface
import numpy as np

# 1. Initialize the InsightFace Analysis engine 
# It automatically selects the ONNXROCMExecutionProvider if built correctly
app = insightface.app.FaceAnalysis(name='buffalo_l', providers=['ROCMExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# 2. Load your image
img = cv2.imread("your_photo.jpg")

# 3. Detect and extract face signatures
faces = app.get(img)

for face in faces:
    # This 512-dimensional array is your independent face signature
    embedding = face.embedding  
    bbox = face.bbox.astype(int)
    print(f"Found face at {bbox}, signature vector shape: {embedding.shape}")




From here, you can save these raw 512-dimensional vector math hashes into a lightweight independent database (like SQLite, Milvus, or FAISS) to perform instant mathematical matching across your files entirely outside of any specific photo editing application.
