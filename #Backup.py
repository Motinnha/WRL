import cv2
from ultralytics import YOLO
import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops, regionprops_table
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull, Delaunay
import pyvista as pv
import time
# Load a pretrained YOLOv8n model
model = YOLO('C:\\Users\\CaioM\\Downloads\\BicoEdital109.v5i.yolov8\\train7\\weights\\best.pt')
pi_value = np.pi

class DepthCamera:
    def __init__(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()
        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device.query_sensors()[0].set_option(rs.option.laser_power, 15)
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
        config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 60)

               
        # Start streaming
        self.pipeline.start(config)
    def get_frame(self):
        frames = self.pipeline.wait_for_frames(timeout_ms=2000)
        colorizer = rs.colorizer()
        colorized = colorizer.process(frames)
        ply = rs.save_to_ply("1.ply")
        ply.set_option(rs.save_to_ply.option_ply_binary, True)
        ply.set_option(rs.save_to_ply.option_ply_normals, False)
        ply.process(colorized)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        infrared = frames.get_infrared_frame()
        infra_image = np.asanyarray(infrared.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        if not depth_frame or not color_frame:
            return False, None, None
        return True, depth_image, color_image, infra_image
    def depth(self):
            frames = self.pipeline.wait_for_frames(timeout_ms=2000)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            if not depth_frame or not color_frame:
                return False, None, None
            return depth_image
    def get_depth_scale(self):
        self.depth_sensor = self.pipeline.get_active_profile().get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        return self.depth_scale,self.depth_sensor
    def release(self):
        self.pipeline.stop()
dc=DepthCamera() #inicia a camera
number = int() #ordenar as detecções
depth_scale= dc.get_depth_scale() #constante usada para transformação pixel/mm
depth_image = dc.depth() #imagem de distancia
distance=np.array(depth_image) #distancia em array 

#definir ponto de distancia a partir do click do mouse
def pontoglobal(event,x,y,flags,params):
    global point
    point = (x,y)
    distance = depth_frame[x,y]

def click(event,x,y,flags,params):
        distance = depth_frame[240,320]        
        fonte = cv2.FONT_HERSHEY_PLAIN
        ponto = "{}mm".format(distance)        
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.putText(color_frame,ponto,(x,y),fonte,1,(255,255,0),2)
            cv2.circle(color_frame, (x,y), 4, (0, 0, 255))
            cv2.imshow('teste',color_frame)
        if event == cv2.EVENT_RBUTTONDOWN:
            cv2.putText(color_frame,ponto,(320,240),fonte,1,(255,255,0),2)
            cv2.imshow('teste',color_frame)    
            cv2.circle(color_frame, (320,240), 4, (0, 0, 255))
while True:
    ret,depth_frame, color_frame, infra_image = dc.get_frame() #chamar as propriedades da camera
    cv2.imshow('camera',infra_image)
    imagem_bgr = cv2.cvtColor(infra_image, cv2.COLOR_GRAY2BGR)       
    pressedKey = cv2.waitKey(1) & 0xFF #definir uma tecla do waitKey
    if pressedKey == ord('q'):
        number += 1
        nome = ('detect')+(str(number))
        tempo = time.time()
        results = model(imagem_bgr,device = 0,retina_masks=True, save = True, save_crop = True,save_frames=True,overlap_mask=True, project ="C:\\Users\\CaioM\\Edital 109",name = nome, save_txt = True)
        for result in results:
            frame1 = results[0].plot(masks= True) #plotar a segmentação
            #cv2.imshow('img_segmentada',frame1) #mostrar a imagem com a segmentação
            mascaras = result.masks.data
            depth_data_numpy_binaria = mascaras.cpu().numpy()   #tranformar array em np.array
            detections = len(result)  #quantidades de detecções
            '''for cont in range(detections):
                plt.imshow(depth_data_numpy_binaria[cont],cmap='gray')  #plotar as formas das mascaras
                if cont == 0:
                    plt.title(f'bico de lança')
                    plt.show()
                else:
                    plt.title(f'furo{cont}')
                    plt.show()'''
            depth_data_numpy_coordenada=np.argwhere(depth_data_numpy_binaria[0] == 1) #transformar formascara em coordenada nos pontos em que tem mascara
            x = depth_data_numpy_coordenada[0:len(depth_data_numpy_coordenada),0]
            y = depth_data_numpy_coordenada[0:len(depth_data_numpy_coordenada),1]
            z = depth_frame[x,y]
            indices_remover = []
            for i, (j_z) in enumerate(zip(z)):
                if j_z[0] == 0 or j_z[0] >= 750:
                    indices_remover.append(i)
            # Remover elementos de filtered_x usando os índices calculados
            print(x,y,z)
            print("Tempo yolov8: ", time.time()-tempo)
            tempo=time.time()
            filtered_x = np.array([v for i, v in enumerate(x) if i not in indices_remover])
            filtered_y = np.array([v for i, v in enumerate(y) if i not in indices_remover])
            filtered_z = np.array([v for i, v in enumerate(z) if i not in indices_remover])
            # Criar a matriz de entrada para a regressão
            X = np.column_stack((np.ones_like(filtered_x), filtered_x, filtered_y, filtered_x**2, filtered_y**2, filtered_x*filtered_y))

            # Calcular os coeficientes da regressão
            coefficients, _, _, _ = np.linalg.lstsq(X, filtered_z, rcond=None)
            print(coefficients)
            print("Tempo coefficients", time.time()-tempo)
            tempo=time.time()
            # Função para calcular z previsto
            '''
            def predict_z(filtered_x, filtered_y):
                return coefficients[0] + coefficients[1]*filtered_x + coefficients[2]*filtered_y + coefficients[3]*filtered_x**2 + coefficients[4]*filtered_y**2 + coefficients[5]*filtered_x*filtered_y

            # Criar uma grade de pontos para plotar o plano ajustado
            filtered_x_range = np.linspace(min(filtered_x), max(filtered_x), 50)
            filtered_y_range = np.linspace(min(filtered_y), max(filtered_y), 50)
            X_grid, filtered_Y_grid = np.meshgrid(filtered_x_range, filtered_y_range)
            Z_grid = predict_z(X_grid, filtered_Y_grid)
            '''
            # Plotar os dados e o plano ajustado
            '''fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            #ax.scatter(filtered_x, filtered_y, filtered_z, color='red', label='Dados Observados')
            ax.plot_surface(X_grid, filtered_Y_grid, Z_grid, alpha=0.5, rstride=100, cstride=100, color='blue', label='Plano de Regressão')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('Regressão Polinomial de Segundo Grau em 3D')
            ax.legend()
            plt.show()'''
            '''depth_data_numpy_binaria_1 = (depth_data_numpy_binaria[0] == 1)
            depth_value_soma = 0
            for i in range(len(depth_data_numpy_coordenada)):
                print(depth_data_numpy_coordenada[i])
                ponto = depth_data_numpy_coordenada[i]
                depth_value = depth_frame[ponto[0]][ponto[1]]
                depth_value_soma += depth_value
                print(depth_frame[ponto[0]][ponto[1]])
            media = depth_value_soma/len(depth_data_numpy_coordenada)
            print(media)
            depth_data_numpy_binaria[0][depth_data_numpy_binaria_1] = media
            print(len(depth_data_numpy_coordenada))'''
            for j in range (detections):
                depth_data_numpy_coordenada=np.argwhere(depth_data_numpy_binaria[j] == 1)
                print(depth_data_numpy_coordenada)
                for i in range(len(depth_data_numpy_coordenada)): #para o bico de lança
                    x = depth_data_numpy_coordenada[i][0].astype(int) #coordenada x da mascara do bico de lança
                    y = depth_data_numpy_coordenada[i][1].astype(int) #coordenada y da mascara do bico de lança
                    v = (coefficients[0]) + (coefficients[1]*x) + (coefficients[2]*y) + (coefficients[3]*x**2) + (coefficients[4]*y**2) + (coefficients[5]*x*y)
                    depth_data_numpy_binaria[j][x][y] = v*depth_scale[0]
            for i in range(detections):
                np.sum(depth_data_numpy_binaria[])
            print("Tempo PxP",time.time()-tempo)

            
    if pressedKey == ord('a'):  
        break
    def release(self):
        self.pipeline.stop()()