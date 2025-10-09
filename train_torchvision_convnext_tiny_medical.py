from ultralytics import YOLO

# Model with ConvNeXt backbone
model = YOLO("torchVision_convnext_tiny.yaml")
results = model.train(data="medical-pills.yaml", epochs=100)
# model.info()