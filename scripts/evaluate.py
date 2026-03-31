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

# Start MLflow run
with mlflow.start_run(run_name="YOLO_Evaluation"):
    model = YOLO(model_path)
    data_yaml = os.path.abspath(
        '/home/runner/work/MLOps/MLOps/data/coco128.yaml'
    )
    metrics = model.val(data=data_yaml)
    print(metrics)

    metrics_dict = None
    metrics_path = "outputs/metrics.json"
    # Try to get a metrics dictionary if available
    if hasattr(metrics, "keys"):
        metrics_dict = dict(metrics)
    elif hasattr(metrics, "results_dict"):
        metrics_dict = metrics.results_dict
    elif hasattr(metrics, "metrics"):
        metrics_dict = metrics.metrics

    if metrics_dict:
        os.makedirs("outputs", exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics_dict, f, indent=2)
        for k, v in metrics_dict.items():
            try:
                mlflow.log_metric(k, float(v))
            except Exception:
                pass
        mlflow.log_artifact(metrics_path, artifact_path="eval")
    else:
        print("Could not extract metrics for MLflow logging.")
