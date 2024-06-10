import cv2
from PIL import Image, ImageDraw
import numpy as np
import time
import os

# Configura la imagen y los parámetros de tiles
list_image = os.listdir('D:\\Usuarios\\Carlos\\Documentos\\MUIT\\TFM\\Code\\AirbusAircraft\\archive\\test')
image_path = f'D:\\Usuarios\\Carlos\\Documentos\\MUIT\\TFM\\Code\\AirbusAircraft\\archive\\test\\{list_image[0]}'
tile_size = 512
overlap = 0.2

# Lee la imagen
image = Image.open(image_path)
width, height = image.size
stride = int(tile_size * (1 - overlap))

# Inicializa una lista para los frames
color1 = (255, 0, 0, 100) # Rojo
color2 = (0, 255, 0, 100) # Verde
color_cuadrado1 = color1
color_cuadrado2 = color2
frames = []
x_ant = []
y_ant = []
color_rectangulo = 'green'
# Procesa la imagen en tiles
for y in range(0, height, stride):
    for x in range(0, width, stride):
        right = min(x + tile_size, width)
        bottom = min(y + tile_size, height)
        tile = image.crop((x, y, right, bottom))

        if tile.size != (tile_size, tile_size):
            tile = Image.new('RGB', (tile_size, tile_size), (0, 0, 0))
            tile.paste(image.crop((x, y, right, bottom)), (0, 0))
        
        # Crea una copia de la imagen original para dibujar el tile actual
        draw_image = image.copy()
        draw = ImageDraw.Draw(draw_image)
        
        # Dibuja el rectángulo del tile procesado con relleno transparente
        if x_ant or y_ant:
            overlay = Image.new('RGBA', (tile_size, tile_size), color_cuadrado1)  # Rojo semi-transparente
            draw_image.paste(overlay, (x_ant, y_ant), overlay)
        overlay = Image.new('RGBA', (tile_size, tile_size), color_cuadrado2)  # Rojo semi-transparente
        draw_image.paste(overlay, (x, y), overlay)
        draw.rectangle([x, y, right, bottom], outline=color_rectangulo, width=5)
                
        # Convierte la imagen a numpy array y luego a BGR para OpenCV
        frame = cv2.cvtColor(np.array(draw_image), cv2.COLOR_RGB2BGR)
        frames.append(frame)
        x_ant = x
        y_ant = y
        if color_cuadrado1 == color1:
            color_cuadrado1 = color2
            color_cuadrado2 = color1
            color_rectangulo = 'red'
        else:
            color_cuadrado1 = color1
            color_cuadrado2 = color2
            color_rectangulo = 'green'
        

# Configura el video writer
output_path = 'Video/output_video.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 0.5  # Frames per second
video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Escribe los frames al video
for frame in frames:
    video.write(frame)

video.release()

print(f"Video saved at {output_path}")
