from djitellopy import Tello
import time
import cv2
from ultralytics import YOLO

# Inicializa o drone
tello = Tello()

model = YOLO(r"C:\Users\CaioM\Downloads\yolov8s-seg.pt")

# Conecta ao drone
tello.connect()

# Verifica o nível da bateria
print(f"Nível da bateria: {tello.get_battery()}%")

# Decola
tello.takeoff()

# Voa 1,6 metros para cima
tello.move_up(160)

# Liga a câmera e começa a transmissão de vídeo
tello.streamon()

# Função para mostrar o vídeo e aguardar a tecla 'q' para parar o voo
def show_video():
    while True:
        frame = tello.get_frame_read().frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = model(frame, device=0, classes=0)

        for result in results:
            frame1 = result.plot(masks=True, boxes=True)  # Plot segmentation masks
            cv2.imshow('img_segmentada', frame1)
        
        # Saia do loop quando a tecla 'q' for pressionada
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Mostra o vídeo
show_video()

# Para a transmissão de vídeo
tello.streamoff()

# Pousa
tello.land()

# Desconecta do drone
tello.end()

# Fecha todas as janelas do OpenCV
cv2.destroyAllWindows()
