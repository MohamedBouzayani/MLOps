from ultralytics import YOLO
import os

import os
print("Current working dir:", os.getcwd())
print("Content in cwd:", os.listdir())
print("Does coco128.yaml exist?", os.path.exists('coco128.yaml'))
data_yaml = os.path.abspath('/home/runner/work/MLOps/MLOps/data/coco128.yaml')
output_dir = os.path.abspath('../../outputs/model_weights/')

model = YOLO("yolov8n.pt")
model.train(data=data_yaml, epochs=10, imgsz=640)

# Save best model to outputs (after training, it's saved in runs/train/exp*/weights)
import shutil, glob
#best_pt = glob.glob('/opt/hostedtoolcache/Python/3.9.25/x64/lib/python3.9/site-packages/tests/tmp/runs/detect/train')[0]
#shutil.copy(best_pt, os.path.join(output_dir, 'model_a_best.pt'))

# Find and copy best model
best_pt = '/opt/hostedtoolcache/Python/3.9.25/x64/lib/python3.9/site-packages/tests/tmp/runs/detect/train/weights/best.pt'
shutil.copy(best_pt, os.path.join(output_dir, 'model_a_best.pt'))

print('Training done. Best model saved.')