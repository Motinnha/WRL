from djitellopy import Tello
import cv2
import time
from ultralytics import YOLO
pressedKey = cv2.waitKey(1) & 0xFF
# Inicializa o drone
tello = Tello()
# Conecta ao drone
tello.connect()
tello.takeoff()
while True:
    if pressedKey == ord('w'):
        tello.move_forward(100)


    if pressedKey == ord('q'):
        tello.land()



