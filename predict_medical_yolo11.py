from ultralytics import YOLO

# Load a model
model = YOLO("/home/srs/git/travisyou/ultralytics_travis/runs/detect/train/weights/best.pt")  # load a fine-tuned model

# Inference using the model
results = model.predict("https://ultralytics.com/assets/medical-pills-sample.jpg")