from ultralytics import YOLO

model = YOLO('../outputs/model_weights/model_a_best.pt')
metrics = model.val(data='../data/my_custom_dataset.yaml')
print(metrics)