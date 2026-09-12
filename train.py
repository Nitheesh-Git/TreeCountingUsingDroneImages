from ultralytics import YOLO

def main():
    # Load a pre-trained YOLOv8 Nano model
    model = YOLO("yolov8n.pt")

    # Train the model with CPU-friendly settings
    # We use a smaller batch size (8) and limit epochs with early stopping (patience)
    print("Starting YOLO baseline training...")
    results = model.train(
        data="datasets/Tree-Top-View/data.yaml",
        epochs=30,
        imgsz=416,
        batch=8,
        patience=5,
        project="tree_detection",
        name="baseline_run",
        exist_ok=True
    )
    
    print("\nTraining completed.")
    print("Results and best weights are saved in: tree_detection/baseline_run/")

if __name__ == "__main__":
    main()
