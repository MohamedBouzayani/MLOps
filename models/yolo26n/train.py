from ultralytics import YOLO
import os
import shutil
import mlflow

# Start MLflow run
with mlflow.start_run(run_name="YOLO_Training"):
    # Debug info
    print("Current working dir:", os.getcwd())
    print("Content in cwd:", os.listdir())

    # Data yaml
    data_yaml = '/home/runner/work/MLOps/MLOps/data/coco128.yaml'
    output_dir = '/home/runner/work/MLOps/MLOps/outputs/model_weights'

    # Log hyperparameters to MLflow
    mlflow.log_param("yolo_model", "yolov8n.pt")
    mlflow.log_param("epochs", 10)
    mlflow.log_param("imgsz", 640)
    mlflow.log_param("data_yaml", data_yaml)

    # Train YOLO model
    model = YOLO("yolov8n.pt")
    results = model.train(data=data_yaml, epochs=10, imgsz=640)

    # Get the actual YOLO output directory (works for YOLOv8+)
    yolo_saved_dir = results.save_dir
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
        # Log model artifact in MLflow
        mlflow.log_artifact(dst_path, artifact_path="model")
    else:
        print(f"best.pt not found at: {best_pt}")
        exit(1)

    # Log training metrics if available (as MLflow metrics)
    metrics_dict = None
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
        for k, v in metrics_dict.items():
            # Flatten keys for MLflow (no slashes or spaces)
            key = str(k).replace("/", "_").replace(" ", "_")
            try:
                mlflow.log_metric(key, float(v))
            except Exception:
                continue

    print('Training done. Best model saved and logged to MLflow.')