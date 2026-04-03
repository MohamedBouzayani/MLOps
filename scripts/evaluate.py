import os
import json
from ultralytics import YOLO
import mlflow

model_path = (
    '/home/runner/work/MLOps/MLOps/outputs/model_weights/model_a_best.pt'
)
if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"{model_path} not found! Did training and copy step complete success?"
    )

with mlflow.start_run(run_name="YOLO_Evaluation"):
    model = YOLO(model_path)
    data_yaml = os.path.abspath(
        '/home/runner/work/MLOps/MLOps/data/coco128.yaml'
    )
    results = model.val(data=data_yaml)
    print(results)

    metrics_dict = None
    metrics_path = "outputs/metrics.json"

    # Defensive extraction covering YOLOv8+ formats
    if hasattr(results, "results_dict"):
        metrics_dict = results.results_dict
    elif hasattr(results, "metrics"):
        metrics_dict = results.metrics
    elif isinstance(results, dict):
        metrics_dict = results
    elif isinstance(results, list):
        print("Warning: results is a list, likely no metrics extracted.")
        metrics_dict = None

    if metrics_dict:
        os.makedirs("outputs", exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics_dict, f, indent=2)
        # Log each metric to MLflow (flatten names, ensure float)
        for k, v in metrics_dict.items():
            key = str(k).replace("/", "_").replace(" ", "_")
            try:
                mlflow.log_metric(key, float(v))
            except Exception:
                continue
        mlflow.log_artifact(metrics_path, artifact_path="eval")
    else:
        print("Could not extract metrics for MLflow log.")
