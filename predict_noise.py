"""
Main file for training Yolo model on Pascal VOC dataset

"""

import torch
import torch.optim as optim
from backbone import vgg16, resnet50, efficientnet_b0
from tinyissimo_model import tinyissimoYOLO
from ext_tinyissimo_model import ext_tinyissimoYOLO
from bed_model import bedmodel
from YOLOv1 import Yolov1
from utils import (
    non_max_suppression,
    cellboxes_to_boxes,
    find_the_best_model,
    load_checkpoint,
    
    )
import config
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time
import cv2
from eval_noise import add_noise_to_tensor, add_noise_to_weights, NoisyLayer
import torch.nn as nn

from PIL import ImageDraw, ImageFont
import config

def plot_image_with_pil_(image, boxes):
    """Edit the image to draw bounding boxes using PIL but does not show or save the image."""
    if config.DATASET == "Airbus" or config.DATASET == "reduceAirbus" or config.DATASET == 'Airbus_256':
        class_labels = config.AIRBUS_LABELS
    elif config.DATASET == "PASCAL":
        class_labels = config.PASCAL_LABELS
    # Define colors using PIL
    num_classes = len(class_labels)
    colors = config.colors[:num_classes]
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for box in boxes:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = int(box[0])
        confidence_score = box[1]
        x, y, w, h = box[2], box[3], box[4], box[5]

        upper_left_x = (x - w / 2) * width
        upper_left_y = (y - h / 2) * height
        lower_right_x = (x + w / 2) * width
        lower_right_y = (y + h / 2) * height

        # Ensure that the coordinates are in the correct order
        if upper_left_y > lower_right_y:
            upper_left_y, lower_right_y = lower_right_y, upper_left_y
        if upper_left_x > lower_right_x:
            upper_left_x, lower_right_x = lower_right_x, upper_left_x

        draw.rectangle([upper_left_x, upper_left_y, lower_right_x, lower_right_y], outline=colors[class_pred], width=3)
        font = ImageFont.truetype("font_text/04B_08__.TTF", 20)
        draw.text((upper_left_x, upper_left_y), f'{confidence_score:.2f} {class_labels[class_pred]}', fill=colors[class_pred], font=font)
    return image


def load_model():
    if config.BACKBONE == 'resnet50':
        model = resnet50(split_size=config.SPLIT_SIZE, num_boxes=config.NUM_BOXES, num_classes=config.NUM_CLASSES, pretrained=True).to(config.DEVICE)
    elif config.BACKBONE == 'vgg16':
        model = vgg16(split_size=config.SPLIT_SIZE, num_boxes=config.NUM_BOXES, num_classes=config.NUM_CLASSES, pretrained=True).to(config.DEVICE)
    elif config.BACKBONE == 'efficientnet':
        model = efficientnet_b0(split_size=config.SPLIT_SIZE, num_boxes=config.NUM_BOXES, num_classes=config.NUM_CLASSES).to(config.DEVICE)
    elif config.BACKBONE == 'tinyissimoYOLO':
        model = tinyissimoYOLO(split_size=config.SPLIT_SIZE, num_boxes=config.NUM_BOXES, num_classes=config.NUM_CLASSES).to(config.DEVICE)
    elif config.BACKBONE == 'ext_tinyissimoYOLO':
        model = ext_tinyissimoYOLO(split_size=config.SPLIT_SIZE, num_boxes=config.NUM_BOXES, num_classes=config.NUM_CLASSES).to(config.DEVICE)
    elif config.BACKBONE == 'bed_model':
        model = bedmodel(split_size=config.SPLIT_SIZE, num_boxes=config.NUM_BOXES, num_classes=config.NUM_CLASSES).to(config.DEVICE)
    elif config.BACKBONE == 'Yolov1':
        model = Yolov1(split_size=config.SPLIT_SIZE, num_boxes=config.NUM_BOXES, num_classes=config.NUM_CLASSES).to(config.DEVICE)

    if config.OPTIMIZER == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config.INIT_lr, weight_decay=config.WEIGHT_DECAY, momentum=0.7)
    elif config.OPTIMIZER == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config.INIT_lr, weight_decay=config.WEIGHT_DECAY)
    elif config.OPTIMIZER == 'NAdam':
        optimizer = optim.NAdam(model.parameters(), lr=config.INIT_lr, weight_decay=config.WEIGHT_DECAY)  

    best_model, epoch_best_model = find_the_best_model(os.path.join(config.DRIVE_PATH,f'{config.BACKBONE}/{config.TOTAL_PATH}/model'))

    load_checkpoint(torch.load(os.path.join(config.DRIVE_PATH,f'{config.BACKBONE}/{config.TOTAL_PATH}/model/{best_model}')), model, optimizer)
    model.eval()
    return model

def do_prediction_image(model, img_path, file_name, case1=False, noise1=0.05):
    image = Image.open(img_path).convert("RGB")
    image_array, boxes = config.predict_transforms(image, [])
    
    x = image_array.unsqueeze(0).to(config.DEVICE)
    start_time = time.time()
    if case1:
        x = add_noise_to_tensor(x, noise1)    
    bboxes = cellboxes_to_boxes(model(x))
    bboxes = non_max_suppression(bboxes[0], iou_threshold=0.5, threshold=0.3, box_format="midpoint")
    end_time = time.time()
    
    print(f"Inference per image, Time: {end_time - start_time} seconds, FPS: {1 / (end_time - start_time)}")
    image_predicted = plot_image_with_pil_(image, bboxes)
    
    base_name = os.path.splitext(file_name)[0]
    os.makedirs(os.path.join(config.DRIVE_PATH, config.BACKBONE, config.TOTAL_PATH, 'test_noise'), exist_ok=True)
    output_path = os.path.join(config.DRIVE_PATH, config.BACKBONE, config.TOTAL_PATH, 'test_noise', base_name + '_predicted.jpg')
    print(f"Image saved in: {output_path}")
    image_predicted.save(output_path)
    return image_predicted

def process_frame(frame, model, case1=False, noise1=0.05):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image_array, boxes = config.predict_transforms(image, [])
    
    x = image_array.unsqueeze(0).to(config.DEVICE)
    if case1:
        x = add_noise_to_tensor(x, noise1)
    bboxes = cellboxes_to_boxes(model(x))
    bboxes = non_max_suppression(bboxes[0], iou_threshold=0.5, threshold=0.3, box_format="midpoint")
    
    return bboxes


def process_large_image(model, img_path, file_name, tile_size=256, overlap=0.2, case1=False, noise1=0.05):
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
            
            bboxes, tile_time = do_prediction_tile(model, tile, case1=case1, noise1=noise1)
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
    end_time = time.time()
    print(f"Inference large image, Time: {end_time - start_time} seconds, FPS: {1 / (end_time - start_time)}")
    print(f"Average inference time per tile: {sum(tile_times)/len(tile_times):.6f} seconds")
    image_predicted = plot_image_with_pil_(image, all_bboxes)

    base_name = os.path.splitext(file_name)[0]
    os.makedirs(os.path.join(config.DRIVE_PATH, config.BACKBONE, config.TOTAL_PATH, 'test_noise'), exist_ok=True)
    output_path = os.path.join(config.DRIVE_PATH, config.BACKBONE, config.TOTAL_PATH, 'test_noise', base_name + '_predicted.jpg')
    print(f"Image saved in: {output_path}")
    image_predicted.save(output_path)
    return image_predicted

def process_large_frame(model, frame, tile_size=256, overlap=0.2, case1=False, noise1=0.05):
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
            
            bboxes, tile_time = do_prediction_tile(model, tile, case1=case1, noise1=noise1)
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

def do_prediction_video(model, video_path, file_name, high_res=False, case1=False, noise1=0.05):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    base_name = os.path.splitext(file_name)[0]
    os.makedirs(os.path.join(config.DRIVE_PATH, config.BACKBONE, config.TOTAL_PATH, 'test_noise'), exist_ok=True)
    output_path = os.path.join(config.DRIVE_PATH, config.BACKBONE, config.TOTAL_PATH, 'test_noise',base_name + '_predicted.mp4')
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
            bboxes = process_large_frame(model, frame, case1=case1, noise1=noise1)
            frame_with_bboxes = draw_bboxes_on_frame(frame, bboxes)
        else:
            bboxes = process_frame(frame, model, case1=case1, noise1=noise1)
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


def do_prediction_tile(model, image_tile, case1=False, noise1=0.05):
    image_array, boxes = config.predict_transforms(image_tile, [])
    x = image_array.unsqueeze(0).to(config.DEVICE)
    start_time = time.time()
    if case1:
        x = add_noise_to_tensor(x, noise1)
    bboxes = cellboxes_to_boxes(model(x))
    bboxes = non_max_suppression(bboxes[0], iou_threshold=0.5, threshold=0.3, box_format="midpoint")
    end_time = time.time()

    return bboxes, end_time - start_time




def process_media(folder_test, files_to_test, case1=False, case2=False, case3=False, noise1 = 0.05, noise2 = 0.1, noise3 = 0.0003):
    model = load_model()
    '''
    CASE2: Add NoisyLayer to the model if CASE2 flag is True
    '''
    if case2:
        for name, module in model.named_modules():
            if isinstance(module, nn.ReLU) or isinstance(module, nn.LeakyReLU):  # Choose appropriate layers
                setattr(model, name, nn.Sequential(module, NoisyLayer(noise_level=noise2)))
    
    '''
    CASE3: Add noise to model weights if CASE3 flag is True
    '''
    if case3:
        model.apply(lambda m: add_noise_to_weights(m, noise_level=noise3))
    i = 1
    for file_name in files_to_test:
        file_path = os.path.join(folder_test, file_name)
        print(f'[{i}/{len(files_to_test)}] - Processing: {file_name}')
        i += 1 
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext in ['.jpg', '.jpeg', '.png']:
            width, height = Image.open(file_path).size
            if width > 256 or height > 2560:
                process_large_image(model, file_path, file_name, case1=case1, noise1=noise1)
            else:
                do_prediction_image(model, file_path, file_name, case1=case1, noise1=noise1)
        elif ext == '.mp4':
            cap = cv2.VideoCapture(file_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            do_prediction_video(model, file_path, file_name, high_res=(width > 256 or height > 256) , case1=case1, noise1=noise1)
        else:
            print(f"Unsupported file format: {ext}")

if __name__ == "__main__":
    folder_test = 'test'
    # files_to_test = ['Aeropuerto.mp4', 'large_image.jpg', 'img1.jpg', 'Barajas.jpg']
    # files_to_test = ['img1.jpg']
    files_to_test = os.listdir(folder_test)
    
    process_media(folder_test, files_to_test, case1=True, case2=True, case3=False, noise1 = 0.05, noise2 = 0.1, noise3 = 0.0003)