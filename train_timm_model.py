from ultralytics import YOLO

# Model with ConvNeXt backbone
model = YOLO("/home/srs/git/travisyou/ultralytics_travis/custom_model.yaml")
results = model.train(data="caltech101_split", epochs=100)