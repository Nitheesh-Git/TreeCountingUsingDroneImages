from ultralytics import YOLO
import os


def main():
    # Load the pre-trained YOLOv8 Nano model
    model = YOLO("yolov8n.pt")

    # Dataset configuration
    dataset = "datasets/Tree-Top-View/data.yaml"

    # Training configuration
    epochs = 30
    image_size = 416
    batch_size = 8
    patience = 5

    # Output configuration
    project_name = "tree_detection"
    run_name = "baseline_run"

    print("=" * 60)
    print("YOLOv8 Tree Detection - Baseline Training")
    print("=" * 60)

    print(f"Dataset       : {dataset}")
    print(f"Model         : YOLOv8 Nano")
    print(f"Epochs        : {epochs}")
    print(f"Image Size    : {image_size}x{image_size}")
    print(f"Batch Size    : {batch_size}")
    print(f"Patience      : {patience}")
    print("=" * 60)

    # Verify that the dataset configuration exists
    if not os.path.exists(dataset):
        raise FileNotFoundError(
            f"Dataset configuration not found: {dataset}"
        )

    print("\nStarting YOLO baseline training...\n")

    # Train the model
    results = model.train(
        data=dataset,
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        patience=patience,
        project=project_name,
        name=run_name,
        exist_ok=True,
        verbose=True
    )

    # Output directory
    output_dir = os.path.join(project_name, run_name)

    print("\n" + "=" * 60)
    print("Training completed successfully.")
    print("=" * 60)
    print(f"Training results : {output_dir}/")
    print(f"Best weights     : {output_dir}/weights/best.pt")
    print(f"Last weights     : {output_dir}/weights/last.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()