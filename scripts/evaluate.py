from ultralytics import YOLO

model = YOLO('../outputs/model_weights/model_a_best.pt')
metrics = model.val(data='../data/coco128.yaml')
print(metrics)