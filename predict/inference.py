import os
from PIL import Image, ImageDraw, ImageFont
import time
import cv2
import sys
import numpy as np
sys.path.append(os.getcwd())
import config
from utils.utils import (
    non_max_suppression,
    cellboxes_to_boxes,
    )

from utils.load_save_model import load_model
from utils.plot_predictions import plot_image_with_pil_
from predict.predict_functions import do_prediction_tile


def do_prediction_image(model, img_path, file_name, path_opt_model=None):
    image = Image.open(img_path).convert("RGB")
    image_array, boxes = config.predict_transforms(image, [])
    
    x = image_array.unsqueeze(0).to(config.DEVICE)
    start_time = time.time()
    bboxes = cellboxes_to_boxes(model(x))
    bboxes = non_max_suppression(bboxes[0], iou_threshold=0.5, threshold=0.3, box_format="midpoint")
    end_time = time.time()
    
    print(f"Inference per image, Time: {end_time - start_time} seconds, FPS: {1 / (end_time - start_time)}")
    image_predicted = plot_image_with_pil_(image, bboxes)
    
    base_name = os.path.splitext(file_name)[0]
    if path_opt_model is None:
        os.makedirs(os.path.join(config.ROOT_DIR, config.BACKBONE, config.TOTAL_PATH, 'test'), exist_ok=True)
        output_path = os.path.join(config.ROOT_DIR, config.BACKBONE, config.TOTAL_PATH, 'test', base_name + '_predicted.jpg')
    else:
        os.makedirs(os.path.join(config.ROOT_DIR, config.BACKBONE, config.TOTAL_PATH, 'model_opt', path_opt_model, 'test'), exist_ok=True)
        output_path = os.path.join(config.ROOT_DIR, config.BACKBONE, config.TOTAL_PATH, 'model_opt', path_opt_model, 'test' ,base_name + '_predicted.jpg')
    print(f"Image saved in: {output_path}")
    image_predicted.save(output_path)

    return image_predicted

def process_frame(frame, model):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image_array, boxes = config.predict_transforms(image, [])
    
    x = image_array.unsqueeze(0).to(config.DEVICE)
    bboxes = cellboxes_to_boxes(model(x))
    bboxes = non_max_suppression(bboxes[0], iou_threshold=0.5, threshold=0.3, box_format="midpoint")
    
    return bboxes


def process_large_image(model, img_path, file_name, tile_size=256, overlap=0.2, path_opt_model=None, generate_video_tiles=False):
    image = Image.open(img_path).convert("RGB")
    width, height = image.size
    stride = int(tile_size * (1 - overlap))
    base_name = os.path.splitext(file_name)[0]
    if path_opt_model is None:
        os.makedirs(os.path.join(config.ROOT_DIR, config.BACKBONE, config.TOTAL_PATH, 'test'), exist_ok=True)
        output_path = os.path.join(config.ROOT_DIR, config.BACKBONE, config.TOTAL_PATH, 'test', base_name + '_predicted.jpg')
        video_output_path = os.path.join(config.ROOT_DIR, config.BACKBONE, config.TOTAL_PATH, 'test', base_name + '_video.avi')
    else:
        os.makedirs(os.path.join(config.ROOT_DIR, config.BACKBONE, config.TOTAL_PATH, 'model_opt', path_opt_model, 'test'), exist_ok=True)
        output_path = os.path.join(config.ROOT_DIR, config.BACKBONE, config.TOTAL_PATH, 'model_opt', path_opt_model, 'test', base_name + '_predicted.jpg')
        video_output_path = os.path.join(config.ROOT_DIR, config.BACKBONE, config.TOTAL_PATH, 'model_opt', path_opt_model, 'test', base_name + '_video.avi')

    all_bboxes = []
    tile_times = []
    start_time = time.time()
    
    # Preparar el vídeo
    video_frames = []

    for y in range(0, height, stride):
        for x in range(0, width, stride):
            right = min(x + tile_size, width)
            bottom = min(y + tile_size, height)
            tile = image.crop((x, y, right, bottom))
            
            if tile.size != (tile_size, tile_size):
                tile = Image.new('RGB', (tile_size, tile_size), (0, 0, 0))
                tile.paste(image.crop((x, y, right, bottom)), (0, 0))
            
            # bboxes, tile_time = do_prediction_tile(model, tile, output_tile_path)
            bboxes, tile_time = do_prediction_tile(model, tile)

            tile_times.append(tile_time)
            for bbox in bboxes:
                # convert x_center of the tile to the x_center of the image
                bbox[2] = (bbox[2] * tile_size + x) / width
                # convert y_center of the tile to the y_center of the image
                bbox[3] = (bbox[3] * tile_size + y) / height
                # convert width of the tile to the width of the image
                bbox[4] = bbox[4] * tile_size / width
                # convert height of the tile to the height of the image
                bbox[5] = bbox[5] * tile_size / height
                
                all_bboxes.append(bbox)
            
            # Crear un frame de la imagen con las predicciones actuales y la caja roja
            current_predicted_image = image.copy()
            draw = ImageDraw.Draw(current_predicted_image)
            draw.rectangle([x, y, right, bottom], outline="red", width=4)
            current_predicted_image = plot_image_with_pil_(current_predicted_image, all_bboxes)
            frame = np.array(current_predicted_image)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_frames.append(frame)

    all_bboxes = non_max_suppression(all_bboxes, iou_threshold=0.1, threshold=0.4, box_format="midpoint")
    end_time = time.time()
    total_time = end_time - start_time
    average_tile_time = total_time / len(video_frames)
    fps = 1 / average_tile_time
    print(f"Inference large image, Time: {total_time} seconds, FPS: {fps}")
    print(f"Average inference time per tile: {average_tile_time:.6f} seconds")
    image_predicted = plot_image_with_pil_(image, all_bboxes, colors = ['blue'])

    print(f"Image saved in: {output_path}")
    image_predicted.save(output_path)

    # Agregar el frame post-NMS al vídeo
    final_frame = np.array(image_predicted)
    final_frame = cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR)
    video_frames.append(final_frame)

    # Añadir copias del último frame para que dure más tiempo
    extra_frames = int(fps * 3)  # 3 segundos de duración extra
    
    if generate_video_tiles:
        for _ in range(extra_frames):
            video_frames.append(final_frame)

        # Guardar el vídeo
        frame_height, frame_width, _ = video_frames[0].shape
        out = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'XVID'), fps / 4, (frame_width, frame_height))
        for frame in video_frames:
            out.write(frame)
        out.release()

        print(f"Video saved in: {video_output_path}")
    return image_predicted

def process_large_frame(model, frame, tile_size=256, overlap=0.2):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    width, height = image.size
    stride = int(tile_size * (1 - overlap))
    
    all_bboxes = []
    tile_times = []
    start_time = time.time()
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            right = min(x + tile_size, width)
            bottom = min(y + tile_size, height)
            tile = image.crop((x, y, right, bottom))
            
            if tile.size != (tile_size, tile_size):
                tile = Image.new('RGB', (tile_size, tile_size), (0, 0, 0))
                tile.paste(image.crop((x, y, right, bottom)), (0, 0))
            
            bboxes, tile_time = do_prediction_tile(model, tile)
            tile_times.append(tile_time)
            for bbox in bboxes:
                # convert x_center of the tile to the x_center of the image
                bbox[2] = (bbox[2] * tile_size + x) / width
                # convert y_center of the tile to the y_center of the image
                bbox[3] = (bbox[3] * tile_size + y) / height
                # convert width of the tile to the width of the image
                bbox[4] = bbox[4] * tile_size / width
                # convert height of the tile to the height of the image
                bbox[5] = bbox[5] * tile_size / height
                
                all_bboxes.append(bbox)
    all_bboxes = non_max_suppression(all_bboxes, iou_threshold=0.4, threshold=0.4, box_format="midpoint")

    return all_bboxes

def draw_bboxes_on_frame(frame, bboxes):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image_with_bboxes = plot_image_with_pil_(image, bboxes)
    return cv2.cvtColor(np.array(image_with_bboxes), cv2.COLOR_RGB2BGR)

def do_prediction_video(model, video_path, file_name,  tile_size=256, overlap= 0.2, path_opt_model=None):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    base_name = os.path.splitext(file_name)[0]
    if path_opt_model is None:
        os.makedirs(os.path.join(config.ROOT_DIR, config.BACKBONE, config.TOTAL_PATH, 'test'), exist_ok=True)
        output_path = os.path.join(config.ROOT_DIR, config.BACKBONE, config.TOTAL_PATH, 'test',base_name + '_predicted.mp4')
    else:
        os.makedirs(os.path.join(config.ROOT_DIR, config.BACKBONE, config.TOTAL_PATH, 'model_opt', path_opt_model, 'test'), exist_ok=True)
        output_path = os.path.join(config.ROOT_DIR, config.BACKBONE, config.TOTAL_PATH, 'model_opt', path_opt_model, 'test', base_name + '_predicted.mp4')
    print(f"Output video path: {output_path}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    inference_times = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.time()
        
        bboxes = process_large_frame(model, frame, tile_size=tile_size, overlap= overlap)
        frame_with_bboxes = draw_bboxes_on_frame(frame, bboxes)
        end_time = time.time()
        
        inference_time = end_time - start_time
        inference_times.append(inference_time)
        
        out.write(frame_with_bboxes)
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    average_inference_time = sum(inference_times) / len(inference_times)
    print(f"Average inference time per frame: {average_inference_time:.6f} seconds")






def process_media(folder_test, files_to_test, tile_size=256, overlap=0.2, path_opt_model=None, generate_video_tiles = False):
    model, _ = load_model(folder_model=path_opt_model)
    i = 1
    for file_name in files_to_test:
        file_path = os.path.join(folder_test, file_name)
        print(f'[{i}/{len(files_to_test)}] - Processing: {file_name}')
        i += 1 
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext in ['.jpg', '.jpeg', '.png']:
            process_large_image(model, file_path, file_name, tile_size=tile_size, overlap= overlap,path_opt_model = path_opt_model, generate_video_tiles=generate_video_tiles)

        elif ext == '.mp4':
            cap = cv2.VideoCapture(file_path)
            cap.release()
            do_prediction_video(model, file_path, file_name, tile_size=tile_size, overlap= overlap, path_opt_model = path_opt_model)
        else:
            print(f"Unsupported file format: {ext}")

if __name__ == "__main__":
    folder_test = 'test'
    # files_to_test = ['Aeropuerto.mp4', 'large_image.jpg', 'img1.jpg', 'Barajas.jpg']
    files_to_test = ['babb0ef2-ef2d-4cab-b3e2-230ae2418cdc.jpg']
    # files_to_test = os.listdir(folder_test)
    process_media(folder_test, files_to_test)
