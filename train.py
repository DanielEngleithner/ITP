# Trainings Code

from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolov8n.yaml")


#Train the model
results = model.train(data="config.yaml", epochs=85, imgsz=640, batch=16)
