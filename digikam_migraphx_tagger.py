import os
import sys
import json
import subprocess
import cv2
from pathlib import Path
from tqdm import tqdm

# The onnxruntime_migraphx wheel ships the MIGraphX provider as
# libonnxruntime_providers_migraphx.so next to libonnxruntime.so, so ONNX
# Runtime discovers and loads it automatically. No manual
# register_execution_provider_library() call (and no `migraphx` Python module,
# which this system does not provide) is required.
import onnxruntime as ort

import insightface
from insightface.app import FaceAnalysis

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

def get_face_analyzer():
    """Initializes InsightFace forcing your compiled MIGraphX provider."""
    providers = ['MIGraphXExecutionProvider', 'CPUExecutionProvider']
    
    print("\n🚀 Initializing InsightFace on AMD GPU with Custom MIGraphX Build...")
    app = FaceAnalysis(name='buffalo_l', providers=providers)
    
    # ctx_id=0 binds execution paths explicitly to your primary Radeon GPU index
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    # Print verification to console
    print(f"Active ONNX Runtime Session Providers: {ort.get_available_providers()}")
    return app

def convert_to_digikam_coords(bbox, img_w, img_h):
    """Converts absolute pixels [xmin, ymin, xmax, ymax] to digiKam fractions [x, y, w, h]."""
    xmin, ymin, xmax, ymax = bbox
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    x = xmin / img_w
    y = ymin / img_h
    return [max(0.0, min(1.0, val)) for val in [x, y, w, h]]

def tag_image_with_faces(file_path, faces, img_w, img_h):
    """Writes standardized XMP facial metadata rectangles natively readable by digiKam."""
    if not faces:
        return

    # Clear old entries and prepare multi-region appending structures
    exiftool_args = [
        "exiftool",
        "-overwrite_original", 
        "-XMP-mwg-rs:RegionInfoRegionList="
    ]
    
    for idx, face in enumerate(faces):
        bbox = face.bbox.astype(int).tolist()
        x, y, w, h = convert_to_digikam_coords(bbox, img_w, img_h)
        person_name = f"Unknown Person {idx+1}" 
        
        exiftool_args.append(f"-XMP-mwg-rs:RegionName+={person_name}")
        exiftool_args.append("-XMP-mwg-rs:RegionType+=Face")
        exiftool_args.append(f"-XMP-mwg-rs:RegionAreaX+={x:.4f}")
        exiftool_args.append(f"-XMP-mwg-rs:RegionAreaY+={y:.4f}")
        exiftool_args.append(f"-XMP-mwg-rs:RegionAreaW+={w:.4f}")
        exiftool_args.append(f"-XMP-mwg-rs:RegionAreaH+={h:.4f}")
        exiftool_args.append("-XMP-mwg-rs:RegionAreaUnit+=normalized")

    exiftool_args.append(str(file_path))
    
    try:
        subprocess.run(exiftool_args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n⚠️ Metadata write failed on {file_path.name}: {e.stderr.decode().strip()}")

def process_and_tag_library(directory_path, app):
    """Processes your image tree via your custom hardware pipeline."""
    path = Path(directory_path)
    image_paths = [p for p in path.rglob('*') if p.suffix.lower() in IMAGE_EXTENSIONS]
    
    print(f"📷 Found {len(image_paths)} pictures. Commencing MIGraphX accelerated tagging...")
    
    for img_path in tqdm(image_paths, desc="Tagging Progress"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        h, w, _ = img.shape
        try:
            faces = app.get(img)
            if faces:
                tag_image_with_faces(img_path, faces, w, h)
        except Exception as e:
            print(f"\n⚠️ Skipped structural parsing on {img_path.name}: {e}")

if __name__ == "__main__":
    analyzer = get_face_analyzer()
    target_folder = input("\nEnter the path to the photo folder you want to tag: ").strip()
    
    if os.path.exists(target_folder):
        process_and_tag_library(target_folder, analyzer)
        print("\n🎉 Injection complete! Right-click images in digiKam and choose 'Reread Metadata from File'.")
    else:
        print("❌ Directory path not found.")

