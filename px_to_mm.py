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
import sympy as sp
import math
# Load a pretrained YOLOv8n model
model = YOLO('C:\\Users\\CaioM\\Downloads\\Bico 9\\BicoEdital109.v9-teste.yolov8\\runs\\segment\\train2\\weights\\best.pt')
pi_value = np.pi
sqrt = np.sqrt

class DepthCamera:
    def __init__(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()
        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device.query_sensors()[0].set_option(rs.option.laser_power, 12)
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)          

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
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        Abertura = math.degrees(2*math.atan(depth_intrin.width/(2*depth_intrin.fx)))
        infra_image = np.asanyarray(infrared.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        if not depth_frame or not color_frame:
            return False, None, None
        return True, depth_image, color_image, infra_image, Abertura
    def depth(self):
            frames = self.pipeline.wait_for_frames(timeout_ms=2000)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            if not depth_frame or not color_frame:
                return False, None, None
            return depth_image
    pressedKey = cv2.waitKey(1) & 0xFF #definir uma tecla do waitKey
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

ret,depth_frame, color_frame, infra_image, Abertura = dc.get_frame()
imagem_bgr = cv2.cvtColor(infra_image, cv2.COLOR_GRAY2BGR)
number += 1
nome = ('detect')+(str(number))
results = model(imagem_bgr,device = 0,retina_masks=True, save = True, save_crop = True,save_frames=True,overlap_mask=True, project ="C:\\Users\\CaioM\\Edital 109",name = nome, save_txt = True, show_boxes=False)


while True:
    pressedKey = cv2.waitKey(1) & 0xFF
    ret,depth_frame, color_frame, infra_image, Abertura = dc.get_frame() #chamar as propriedades da camera
    # Draw a red filled circle
    imagem_bgr = cv2.cvtColor(infra_image, cv2.COLOR_GRAY2BGR)
    teste = infra_image
    cv2.circle(teste, (320, 240), 140, (0, 255, 255),5, 1)
    cv2.imshow('camera',teste)
    altura = depth_frame[320,240]
    print(altura)
    if pressedKey == ord('q'):
        
        if altura > 500 or altura < 600:
            number += 1
            nome = ('detect')+(str(number))
            tempo = time.time()
            results = model(imagem_bgr,device = 0,retina_masks=True, save = True, save_crop = True,save_frames=True,overlap_mask=True, project ="C:\\Users\\CaioM\\Edital 109",name = nome, save_txt = True, show_boxes=False)
            print("Tempo yolov8: ", time.time()-tempo)
            for result in results:
                frame1 = results[0].plot(masks= True, boxes=False) #plotar a segmentação
                cv2.imshow('img_segmentada',frame1)
                #mostrar a imagem com a segmentação
                mascaras = result.masks.data
                depth_data_numpy_binaria = mascaras.cpu().numpy()   #tranformar array em np.array
                
                detections = len(result)  #quantidades de detecções
                depth_data_numpy_coordenada=np.argwhere(depth_data_numpy_binaria[0] == 1)#transformar formascara em coordenada nos pontos em que tem mascara
                x = depth_data_numpy_coordenada[0:len(depth_data_numpy_coordenada),0]
                y = depth_data_numpy_coordenada[0:len(depth_data_numpy_coordenada),1]
                z = depth_frame[x,y]
                for j in range (detections):
                    cnz_furos = []
                    cnz_furos.append(np.count_nonzero(depth_data_numpy_binaria[j]))

                for i in range (detections):
                    plt.imshow(depth_data_numpy_binaria[i], cmap='gray') 
                    plt.show()
                indices_remover = []
                for i, (j_z) in enumerate(zip(z)):
                    if j_z[0] == 0:
                        indices_remover.append(i)
                # Remover elementos de filtered_x usando os índices calculados
                print(Abertura)
                tempo=time.time()
                filtered_x = np.array([f for i, f in enumerate(x) if i not in indices_remover])
                filtered_y = np.array([f for i, f in enumerate(y) if i not in indices_remover])
                filtered_z = np.array([f for i, f in enumerate(z) if i not in indices_remover])
                # Criar a matriz de entrada para a regressão
                X = np.column_stack((np.ones_like(filtered_x), filtered_x, filtered_y, filtered_x**2, filtered_y**2, filtered_x*filtered_y))

                # Calcular os coeficientes da regressão
                coefficients, _, _, _ = np.linalg.lstsq(X, filtered_z, rcond=None)
                print("Tempo coefficients", time.time()-tempo)
                tempo=time.time()
                def predict_z(filtered_x, filtered_y):
                    return coefficients[0] + coefficients[1]*filtered_x + coefficients[2]*filtered_y + coefficients[3]*filtered_x**2 + coefficients[4]*filtered_y**2 + coefficients[5]*filtered_x*filtered_y

                # Criar uma grade de pontos para plotar o plano ajustado
                filtered_x_range = np.linspace(min(filtered_x), max(filtered_x), 50)
                filtered_y_range = np.linspace(min(filtered_y), max(filtered_y), 50)
                X_grid, filtered_Y_grid = np.meshgrid(filtered_x_range, filtered_y_range)
                Z_grid = predict_z(X_grid, filtered_Y_grid)
                # Plotar os dados e o plano ajustado
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(X_grid, filtered_Y_grid, Z_grid, alpha=1, rstride=100, cstride=100, color='red', label='Plano de Regressão')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title('Malha de Regressão Polinomial de Segundo Grau em 3D')
                ax.legend()
                plt.show()
                pcd = o3d.geometry.PointCloud()
                points = np.column_stack((x,y,predict_z(x,y)))
                pcd.points = o3d.utility.Vector3dVector(points)
                o3d.visualization.draw_geometries([pcd])
                pcd = o3d.geometry.PointCloud()
                points = np.column_stack((filtered_x,filtered_y,depth_frame[filtered_x,filtered_y]))
                pcd.points = o3d.utility.Vector3dVector(points)
                o3d.visualization.draw_geometries([pcd])
                for j in range (detections):
                    depth_data_numpy_coordenada=np.argwhere(depth_data_numpy_binaria[:] == 1)
                    for i in range(len(depth_data_numpy_coordenada)): #para o bico de lança
                        x = depth_data_numpy_coordenada[i,1].astype(int) #coordenada x da mascara do bico de lança
                        y = depth_data_numpy_coordenada[i,2].astype(int) #coordenada y da mascara do bico de lança
                        depth_data_numpy_binaria[j][x,y] = ((math.tan(Abertura/2*math.pi/180)*predict_z(x,y)*2)/640)
            
                #separar as mascaras
                furo_1 = depth_data_numpy_binaria[6]        
                furo_2 = (depth_data_numpy_binaria[5]-depth_data_numpy_binaria[6])
                furo_3 = (depth_data_numpy_binaria[4]-depth_data_numpy_binaria[5])
                furo_4 = (depth_data_numpy_binaria[3]-depth_data_numpy_binaria[4])
                furo_5 = (depth_data_numpy_binaria[2]-depth_data_numpy_binaria[3])
                furo_6 = (depth_data_numpy_binaria[1]-depth_data_numpy_binaria[2])
                bico_completo = (depth_data_numpy_binaria[0])
            #separar as areas dos furos e juntar as mascaras na mascara do bico    
            area_furo_1 = np.sum(furo_1)
            diametro_furo_1_mm = 2*(np.sqrt(area_furo_1/math.pi))
            print('diametro furo 1 mm = ',diametro_furo_1_mm)
            print('area furo 1 = ',area_furo_1)
            area_furo_2 = np.sum(furo_2)
            diametro_furo_2_mm = 2*(np.sqrt(area_furo_2/math.pi))
            print('diametro furo 2 mm = ',diametro_furo_2_mm)
            print('area furo 2 = ',area_furo_2)
            area_furo_3 = np.sum(furo_3)
            diametro_furo_3_mm = 2*(np.sqrt(area_furo_3/math.pi))
            print('diametro furo 3 mm = ',diametro_furo_3_mm)
            print('area furo 3 = ',area_furo_3)
            area_furo_4 = np.sum(furo_4)
            diametro_furo_4_mm = 2*(np.sqrt(area_furo_4/math.pi))
            print('diametro furo 4 mm = ',diametro_furo_4_mm)
            print('area furo 4 = ',area_furo_4)
            area_furo_5 = np.sum(furo_5)
            diametro_furo_5_mm = 2*(np.sqrt(area_furo_5/math.pi))
            print('diametro furos 5 mm = ',diametro_furo_5_mm)
            print('area furo 5 = ',area_furo_5)
            area_furo_6 = np.sum(furo_6)
            diametro_furo_6_mm = 2*(np.sqrt(area_furo_6/math.pi))
            print('diametro furo 6 mm = ',diametro_furo_6_mm)
            print('area furos = ',area_furo_6)
            print('CNZ: ',np.count_nonzero(depth_data_numpy_binaria))
            num_zeros = np.count_nonzero(depth_data_numpy_binaria == 0)
            print('CZ', num_zeros)
            area_total = np.sum(depth_data_numpy_binaria)
            diametro_total = 2*(np.sqrt(area_total/math.pi))
            print('diametro externo = ',diametro_total)
            print('tempo total do programa:',time.time()-tempo)


        else:
            print('FORA DA AREA DE PRECISÃO')
        if pressedKey == ord('a'):  
            break
        def release(self):
            self.pipeline.stop()()
        