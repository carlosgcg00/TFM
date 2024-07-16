"""
Main file for training Yolo model on Pascal VOC dataset

"""
import os
import sys
sys.path.append(os.getcwd())
from utils.utils import (
    non_max_suppression,
    cellboxes_to_boxes,
    )
from utils.plot_predictions import plot_image_with_pil_
from utils.load_save_model import load_model
import config
import os
import numpy as np
from PIL import Image
import time
import cv2
from radiation.SEU_weights import SEU_simmulation
from radiation.SET_activation_functions import register_hooks
import config



def do_prediction_image(model, img_path, file_name, folder_model=None):
    image = Image.open(img_path).convert("RGB")
    image_array, boxes = config.predict_transforms(image, [])
    
    x = image_array.unsqueeze(0).to(config.DEVICE)
    start_time = time.time()  
    bboxes = cellboxes_to_boxes(model(x))
    bboxes = non_max_suppression(bboxes[0], iou_threshold=0.2, threshold=0.4, box_format="midpoint")
    end_time = time.time()
    
    print(f"Inference per image, Time: {end_time - start_time} seconds, FPS: {1 / (end_time - start_time)}")
    image_predicted = plot_image_with_pil_(image, bboxes)
    
    base_name = os.path.splitext(file_name)[0]
    if folder_model is None:
        os.makedirs(os.path.join(config.ROOT_DIR, config.BACKBONE, config.TOTAL_PATH, 'test_noise'), exist_ok=True)
        output_path = os.path.join(config.ROOT_DIR, config.BACKBONE, config.TOTAL_PATH, 'test_noise', base_name + '_predicted.jpg')
    else:
        os.makedirs(os.path.join(config.ROOT_DIR, config.BACKBONE, config.TOTAL_PATH, 'model_opt', folder_model, 'test_noise'), exist_ok=True)
        output_path = os.path.join(config.ROOT_DIR, config.BACKBONE, config.TOTAL_PATH, 'model_opt', folder_model, 'test_noise', base_name + '_predicted.jpg')
    print(f"Image saved in: {output_path}")
    image_predicted.save(output_path)
    return image_predicted

def process_frame(frame, model):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image_array, boxes = config.predict_transforms(image, [])
    
    x = image_array.unsqueeze(0).to(config.DEVICE)
    bboxes = cellboxes_to_boxes(model(x))
    bboxes = non_max_suppression(bboxes[0], iou_threshold=0.2, threshold=0.4, box_format="midpoint")
    
    return bboxes


def process_large_image(model, img_path, file_name, tile_size=256, overlap=0.2, folder_model=None):
    image = Image.open(img_path).convert("RGB")
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
    all_bboxes = non_max_suppression(all_bboxes, iou_threshold=0.2, threshold=0.4, box_format="midpoint")
    end_time = time.time()
    print(f"Inference large image, Time: {end_time - start_time} seconds, FPS: {1 / (end_time - start_time)}")
    print(f"Average inference time per tile: {sum(tile_times)/len(tile_times):.6f} seconds")
    image_predicted = plot_image_with_pil_(image, all_bboxes)

    base_name = os.path.splitext(file_name)[0]
    if folder_model is None:
        os.makedirs(os.path.join(config.ROOT_DIR, config.BACKBONE, config.TOTAL_PATH, 'test_noise'), exist_ok=True)
        output_path = os.path.join(config.ROOT_DIR, config.BACKBONE, config.TOTAL_PATH, 'test_noise', base_name + '_predicted.jpg')
    else:
        os.makedirs(os.path.join(config.ROOT_DIR, config.BACKBONE, config.TOTAL_PATH, 'model_opt', folder_model, 'test_noise'), exist_ok=True)
        output_path = os.path.join(config.ROOT_DIR, config.BACKBONE, config.TOTAL_PATH, 'model_opt', folder_model, 'test_noise', base_name + '_predicted.jpg')
    print(f"Image saved in: {output_path}")
    image_predicted.save(output_path)
    return image_predicted

def process_large_frame(model, frame, tile_size=256, overlap=0.1, folder_model=None):
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
    all_bboxes = non_max_suppression(all_bboxes, iou_threshold=0.2, threshold=0.4, box_format="midpoint")

    return all_bboxes

def draw_bboxes_on_frame(frame, bboxes):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image_with_bboxes = plot_image_with_pil_(image, bboxes)
    return cv2.cvtColor(np.array(image_with_bboxes), cv2.COLOR_RGB2BGR)

def do_prediction_video(model, video_path, file_name, high_res=False, folder_model=None):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    base_name = os.path.splitext(file_name)[0]
    if folder_model is None:
        os.makedirs(os.path.join(config.ROOT_DIR, config.BACKBONE, config.TOTAL_PATH, 'test_noise'), exist_ok=True)
        output_path = os.path.join(config.ROOT_DIR, config.BACKBONE, config.TOTAL_PATH, 'test_noise',base_name + '_predicted.mp4')
    else:
        os.makedirs(os.path.join(config.ROOT_DIR, config.BACKBONE, config.TOTAL_PATH, 'model_opt', folder_model, 'test_noise'), exist_ok=True)
        output_path = os.path.join(config.ROOT_DIR, config.BACKBONE, config.TOTAL_PATH, 'model_opt', folder_model, 'test_noise',base_name + '_predicted.mp4')
    print(f"Output video path: {output_path}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    inference_times = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.time()
        if high_res and (width > 256 or height > 256):
            bboxes = process_large_frame(model, frame, folder_model=folder_model)
            frame_with_bboxes = draw_bboxes_on_frame(frame, bboxes)
        else:
            bboxes = process_frame(frame, model, folder_model=folder_model)
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


def do_prediction_tile(model, image_tile):
    image_array, boxes = config.predict_transforms(image_tile, [])
    x = image_array.unsqueeze(0).to(config.DEVICE)
    start_time = time.time()
    bboxes = cellboxes_to_boxes(model(x))
    bboxes = non_max_suppression(bboxes[0], iou_threshold=0.2, threshold=0.3, box_format="midpoint")
    end_time = time.time()

    return bboxes, end_time - start_time




def process_media(folder_test, files_to_test, case1=False, percentage_layers_case_1 = 0, slices_case1 = 0, interval_1 = (-12, 2), percentage_tensors_1= 0.05,
               case2=False, percentage_layers_case_2 = 0, slices_case2 = 0, interval_2 = (-12, 2), percentage_tensors_2= 0.01,
               folder_model = None):
    model, _ = load_model(folder_model=folder_model)
    model.eval()


    '''
    CASE2: Add NoisyLayer to the model if CASE2 flag is True
    '''
    if case1:
        slice_1 = slices_case1[0] if slices_case1[0] < slices_case1[1] else slices_case1[1]-1
        register_hooks(model, slice_1 = slice_1, 
                        slices_case1 = slices_case1, 
                        percentage_layers_case_1 = percentage_layers_case_1, 
                        interval_1 = interval_1,
                        percentage_tensor_1=percentage_tensors_1)
    
    '''
    CASE3: Add noise to model weights if CASE3 flag is True
    '''
    if case2:
        model = SEU_simmulation(model, slices_case2=slices_case2, percentage_layers_case_2=percentage_layers_case_2, interval_2=interval_2, percentage_tensors_2=percentage_tensors_2)
    i = 1
    for file_name in files_to_test:
        file_path = os.path.join(folder_test, file_name)
        print(f'[{i}/{len(files_to_test)}] - Processing: {file_name}')
        i += 1 
        ext = os.path.splitext(file_path)[1].lower()
        if case1:
            file_name = f'1_slices_{slices_case1[1]}_sli_{slices_case1[0]}_perc_{percentage_layers_case_1}_int_{interval_1[0]}_{interval_1[1]}_p_{percentage_tensors_1}_{file_name}'
        if case2:
            file_name = f'2_slices_{slices_case2[1]}_sli_{slices_case2[0]}_perc_{percentage_layers_case_2}_int_{interval_2[0]}_{interval_2[1]}_p_{percentage_tensors_2}_{file_name}'

        if ext in ['.jpg', '.jpeg', '.png']:
            width, height = Image.open(file_path).size
            if width > 256 or height > 256:
                process_large_image(model, file_path, file_name, folder_model=folder_model)
            else:
                do_prediction_image(model, file_path, file_name, folder_model=folder_model)
        elif ext == '.mp4':
            cap = cv2.VideoCapture(file_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            do_prediction_video(model, file_path, file_name, high_res=(width > 256 or height > 256), folder_model=folder_model)
        else:
            print(f"Unsupported file format: {ext}")

if __name__ == "__main__":
    folder_test = 'test'
    # files_to_test = ['Aeropuerto.mp4', 'large_image.jpg', 'img1.jpg', 'Barajas.jpg']
    files_to_test = ['babb0ef2-ef2d-4cab-b3e2-230ae2418cdc_1024.jpg']
    case1 = True
    case2 = False

    percentage_layers_case_1  = 0.5 if case1 else 0
    percentage_layers_case_2 = 0.5 if case2 else 0
    # Con este parámetro seleccionamos la zona en la que queremos afectar a las activacionies o pesos
    # es decir, slices = [slice_i, n_slices], con esto dividimos todas las capas en n_slices y afectamos a las capas que esten en slice_i
    '''
    |slice 0 | slice 1 | slice 2 | slice 3 | slice 4 | 
    '''
    interval_1 = (-2, 2)
    interval_2 = (-12, 2)
    slices_case1 = [3, 3] 
    slices_case2 = [3,3]
    percentage_tensors_1 = 0.05
    percentage_tensors_2 = 0.01
    
    process_media(folder_test, files_to_test, case1=case1, percentage_layers_case_1=percentage_layers_case_1, slices_case1 = slices_case1, interval_1 = interval_1, percentage_tensors_1= percentage_tensors_1,
               case2=case2, percentage_layers_case_2=percentage_layers_case_2, slices_case2 = slices_case2, interval_2 = interval_2, percentage_tensors_2= percentage_tensors_2,)
