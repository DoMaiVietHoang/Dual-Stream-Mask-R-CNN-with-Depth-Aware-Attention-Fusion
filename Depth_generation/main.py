import os
import torch
import numpy as np
from PIL import Image, ImageFile
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import cv2
from tqdm import tqdm

# Cho phép load ảnh bị truncated (tùy chọn)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ================= CONFIG =================
INPUT_DIR = "/mnt/disk2/home/vlir_hoang/BAMFOREST/val2023"
OUTPUT_DIR = "/mnt/disk2/home/vlir_hoang/ICCE_2026/Depth_dataset/BAMFOREST/val"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= LOAD MODEL =================
image_processor = AutoImageProcessor.from_pretrained(
    "depth-anything/Depth-Anything-V2-Small-hf"
)
model = AutoModelForDepthEstimation.from_pretrained(
    "depth-anything/Depth-Anything-V2-Small-hf"
).to(DEVICE)
model.eval()

# ================= PROCESS FOLDER =================
failed_images = []  # Lưu danh sách ảnh lỗi

for fname in tqdm(os.listdir(INPUT_DIR)):
    if not fname.lower().endswith((".jpg", ".png", ".jpeg", ".tif")):
        continue

    img_path = os.path.join(INPUT_DIR, fname)
    
    try:
        image = Image.open(img_path).convert("RGB")

        inputs = image_processor(images=image, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)

        post_processed = image_processor.post_process_depth_estimation(
            outputs,
            target_sizes=[(image.height, image.width)]
        )

        predicted_depth = post_processed[0]["predicted_depth"]

        # ===== Normalize for visualization =====
        depth_norm = (predicted_depth - predicted_depth.min()) / (
            predicted_depth.max() - predicted_depth.min() + 1e-6
        )
        depth_uint8 = (depth_norm * 255).detach().cpu().numpy().astype(np.uint8)

        # ===== Save grayscale =====
        depth_gray = Image.fromarray(depth_uint8)
        depth_gray.save(os.path.join(OUTPUT_DIR, fname.replace(".", "_depth.")))

        # ===== Save colored depth =====
        depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
        cv2.imwrite(
            os.path.join(OUTPUT_DIR, fname.replace(".", "_depth_color.")),
            depth_color
        )

        # ===== Save float depth (for training / research) =====
        depth_float = predicted_depth.detach().cpu().numpy().astype(np.float32)
        np.save(
            os.path.join(OUTPUT_DIR, fname.replace(".", "_depth.npy")),
            depth_float
        )
        
    except Exception as e:
        failed_images.append((fname, str(e)))
        print(f"\n⚠️ Skipped {fname}: {e}")
        continue

print("\n✅ Done processing folder!")

# In ra danh sách ảnh lỗi (nếu có)
if failed_images:
    print(f"\n⚠️ {len(failed_images)} images failed:")
    for fname, error in failed_images:
        print(f"  - {fname}: {error}")