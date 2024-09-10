import pydicom
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import imageio
import cv2
from ultralytics import YOLO

Tk().withdraw()
file = askopenfilename()

def convert_to_png(file):
    ds = pydicom.dcmread(file)
    shape = ds.pixel_array.shape
    image_2d = ds.pixel_array.astype(float)
    image_2d_scaled = (np.maximum(image_2d, 0) / image_2d.max()) * 255.0
    image_2d_scaled = np.uint8(image_2d_scaled)
    png_file_path = f'{file.rstrip(".dcm")}.png'
    try:
        imageio.imwrite(png_file_path, image_2d_scaled)
        print(f'Successfully saved PNG to {png_file_path}')
    except Exception as e:
        print(f'Failed to save PNG: {e}')

    return png_file_path

# chamando a foto para teste
png_file_path = convert_to_png(file)
image = cv2.imread(png_file_path)

cv2.imshow('DICOM to PNG', image)

model = YOLO(" ")
results = model()






cv2.waitKey(0)
cv2.destroyAllWindows()