import cv2
from ultralytics import YOLO
import numpy as np
import time
import pyrealsense2 as rs
import math
# Initialize YOLO model
model = YOLO(r"C:\Users\CaioM\Downloads\yolov8x-seg.pt")

# Initialize video capture
video_path = "https://192.168.0.90/mjpg/video.mjpg"
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FPS, 30)

while True:
    start = time.time()
    success, img = cap.read()
    
    if not success:
        print("Failed to capture image")
        continue

    # Process image with YOLO model
    results = model(img, device=0)
    
    for result in results:
        frame1 = result.plot(masks=True, boxes=True)  # Plot segmentation masks
        cv2.imshow('img_segmentada', frame1)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
