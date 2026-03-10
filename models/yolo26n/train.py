from ultralytics import YOLO
import os
import shutil

# Debug info
print("Current working dir:", os.getcwd())
print("Content in cwd:", os.listdir())

# Data yaml
data_yaml = '/home/runner/work/MLOps/MLOps/data/coco128.yaml'  # absolute path to yaml
output_dir = '/home/runner/work/MLOps/MLOps/outputs/model_weights'  # absolute desired output directory

# Train YOLO model
model = YOLO("yolov8n.pt")
results = model.train(data=data_yaml, epochs=10, imgsz=640)

# Get the actual YOLO output directory (works for YOLOv8+)
yolo_saved_dir = results.save_dir  # e.g. '/opt/hostedtoolcache/.../runs/detect/train'
print("YOLO outputs saved to:", yolo_saved_dir)

# Path to best.pt file
best_pt = os.path.join(yolo_saved_dir, "weights", "best.pt")
print("best.pt file location:", best_pt)

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Copy best model to output_dir/model_a_best.pt
dst_path = os.path.join(output_dir, "model_a_best.pt")
if os.path.isfile(best_pt):
    shutil.copy(best_pt, dst_path)
    print(f"Best model copied to: {dst_path}")
else:
    print(f"best.pt not found at: {best_pt}")
    exit(1)  # Fail if not found

print('Training done. Best model saved.')