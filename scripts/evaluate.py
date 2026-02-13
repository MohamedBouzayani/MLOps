import os
from ultralytics import YOLO

model = YOLO('/home/runner/work/MLOps/MLOps/outputs/model_weights/model_a_best.pt')
metrics = model.val(data=os.path.abspath('/home/runner/work/MLOps/MLOps/data/coco128.yaml'))
print(metrics)