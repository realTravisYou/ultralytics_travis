# train_timm_backbone.py, 此文件和README.md处于同一级目录
from ultralytics import YOLO

# Model with ConvNeXt backbone
model = YOLO("custom_model.yaml", task="classify")
# model.info()
results = model.train(data="caltech101_split", epochs=100)
