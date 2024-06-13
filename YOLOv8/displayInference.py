from ultralytics import YOLO
import cv2

#Display inferece and tracking

#Config:
model_path = "yolov8n.pt"
confidence = 0.4
video_src = 1 #Select camera Index || "url/path"


model = YOLO(model_path) 

cap = cv2.VideoCapture(video_src) 

if not cap.isOpened():
    print("Failed opening camera.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Falha captura do frrame.")
        break

    results = model.track(frame,conf=confidence, persist=True)

    res_plotted  = results[0].plot()

    resized_frame = cv2.resize(res_plotted,(1280,720))

    cv2.imshow('YOLOv8 Inference', resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
