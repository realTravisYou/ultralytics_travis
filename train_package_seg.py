from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-seg.pt")  # load a pretrained segmentation model (recommended for training)

# Train the model on the Package Segmentation dataset
results = model.train(data="package-seg.yaml", epochs=100, imgsz=640)