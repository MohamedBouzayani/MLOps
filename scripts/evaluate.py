#import os
#from ultralytics import YOLO

#model = YOLO('/home/runner/work/MLOps/MLOps/outputs/model_weights/model_a_best.pt')
#metrics = model.val(data=os.path.abspath('/home/runner/work/MLOps/MLOps/data/coco128.yaml'))
#print(metrics)
import os
from ultralytics import YOLO

model_path = '/home/runner/work/MLOps/MLOps/outputs/model_weights/model_a_best.pt'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"{model_path} not found! Did training and copy step complete successfully?")

model = YOLO(model_path)
metrics = model.val(data=os.path.abspath('/home/runner/work/MLOps/MLOps/data/coco128.yaml'))
print(metrics)