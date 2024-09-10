from djitellopy import Tello
import cv2
import time
from ultralytics import YOLO
model = YOLO(r"C:\Users\CaioM\Downloads\yolov8s-seg.pt")

# Inicializa o drone
tello = Tello()

# Conecta ao drone
tello.connect()

# Exibe a bateria do drone
print(f"Bateria: {tello.get_battery()}%")

# Inicia o vídeo
tello.streamon()

tello.takeoff()



print(tello.get_height())

# Inicializa a captura de vídeo com o OpenCV
while True:
    # Captura um frame do streaming de vídeo do drone
    frame = tello.get_frame_read().frame

    # Converte o frame para o formato correto (BGR para exibição com o OpenCV)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    results = model(frame, device=0,classes=0)
    for result in results:
        frame1 = result.plot(masks=True, boxes=True)  # Plot segmentation masks
        cv2.imshow('img_segmentada', frame1)
    # Saia do loop quando a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Pousa o drone
tello.land()

print(tello.get_flight_time())

# Encerra o streaming de vídeo
tello.streamoff()

print(f"Bateria: {tello.get_battery()}%")
# Desconecta do drone
tello.end()

# Fecha todas as janelas do OpenCV
cv2.destroyAllWindows()
