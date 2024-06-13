from ultralytics import YOLO

#Simple Inference

#Config:
model_path = "yolov8n.pt"
confidence = 0.6
src_path = "img.png"  #.png || .mp4

model = YOLO(model_path) 
model.predict(source=src_path , imgsz=640, conf=confidence, save=True)

