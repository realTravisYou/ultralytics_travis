from ultralytics import YOLO

# Model with ConvNeXt backbone
model = YOLO("TimmVision_medical.yaml")
results = model.train(data="medical-pills.yaml", epochs=100)
# model.info()