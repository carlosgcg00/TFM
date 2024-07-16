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
import eval_noise
import torch.nn as nn
import random
import math
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


def load_model(folder_model=None, model_name=None):
    if config.BACKBONE == 'resnet50':
        from backbone import resnet50
        model = resnet50(split_size=config.SPLIT_SIZE, num_boxes=config.NUM_BOXES, num_classes=config.NUM_CLASSES, pretrained=True).to(config.DEVICE)
    elif config.BACKBONE == 'vgg16':
        from backbone import vgg16
        model = vgg16(split_size=config.SPLIT_SIZE, num_boxes=config.NUM_BOXES, num_classes=config.NUM_CLASSES, pretrained=True).to(config.DEVICE)
    elif config.BACKBONE == 'efficientnet':
        from backbone import efficientnet_b0
        model = efficientnet_b0(split_size=config.SPLIT_SIZE, num_boxes=config.NUM_BOXES, num_classes=config.NUM_CLASSES).to(config.DEVICE)
    elif config.BACKBONE == 'tinyissimoYOLO':
        from tinyissimo_model import tinyissimoYOLO
        model = tinyissimoYOLO(split_size=config.SPLIT_SIZE, num_boxes=config.NUM_BOXES, num_classes=config.NUM_CLASSES).to(config.DEVICE)
    elif config.BACKBONE == 'ext_tinyissimoYOLO':
        from ext_tinyissimo_model import ext_tinyissimoYOLO
        model = ext_tinyissimoYOLO(split_size=config.SPLIT_SIZE, num_boxes=config.NUM_BOXES, num_classes=config.NUM_CLASSES).to(config.DEVICE)
    elif config.BACKBONE == 'bed_model':
        from bed_model import bedmodel
        model = bedmodel(split_size=config.SPLIT_SIZE, num_boxes=config.NUM_BOXES, num_classes=config.NUM_CLASSES).to(config.DEVICE)
    elif config.BACKBONE == 'Yolov1':
        from YOLOv1 import Yolov1
        model = Yolov1(split_size=config.SPLIT_SIZE, num_boxes=config.NUM_BOXES, num_classes=config.NUM_CLASSES).to(config.DEVICE)

    if config.OPTIMIZER == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config.INIT_lr, weight_decay=config.WEIGHT_DECAY, momentum=0.7)
    elif config.OPTIMIZER == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config.INIT_lr, weight_decay=config.WEIGHT_DECAY)
    elif config.OPTIMIZER == 'NAdam':
        optimizer = optim.NAdam(model.parameters(), lr=config.INIT_lr, weight_decay=config.WEIGHT_DECAY)  

    best_model, epoch_best_model = find_the_best_model(os.path.join(config.DRIVE_PATH,f'{config.BACKBONE}/{config.TOTAL_PATH}/model'))
    for param in model.parameters():
        param.requires_grad = True


    if folder_model is None:
        best_model, epoch_best_model = find_the_best_model(os.path.join(config.DRIVE_PATH, f'{config.BACKBONE}/{config.TOTAL_PATH}/model'))
        load_checkpoint(torch.load(os.path.join(config.DRIVE_PATH, f'{config.BACKBONE}/{config.TOTAL_PATH}/model/{best_model}')), model, optimizer)
        print(f"Model loaded: {config.BACKBONE}/{config.TOTAL_PATH}/model/{best_model}")
    else:
        load_checkpoint(torch.load(os.path.join(config.DRIVE_PATH, f'{config.BACKBONE}/{config.TOTAL_PATH}/model_opt/{folder_model}/{model_name}')), model, optimizer)
        print(f"Model loaded: {config.BACKBONE}/{config.TOTAL_PATH}/model_opt/{folder_model}/{model_name}")
    model.eval()
    return model, optimizer

def do_prediction_image(model, img_path, file_name, folder_model=None, model_name=None):
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
        os.makedirs(os.path.join(config.DRIVE_PATH, config.BACKBONE, config.TOTAL_PATH, 'test_noise'), exist_ok=True)
        output_path = os.path.join(config.DRIVE_PATH, config.BACKBONE, config.TOTAL_PATH, 'test_noise', base_name + '_predicted.jpg')
    else:
        os.makedirs(os.path.join(config.DRIVE_PATH, config.BACKBONE, config.TOTAL_PATH, 'model_opt', folder_model, 'test_noise'), exist_ok=True)
        output_path = os.path.join(config.DRIVE_PATH, config.BACKBONE, config.TOTAL_PATH, 'model_opt', folder_model, 'test_noise', base_name + '_predicted.jpg')
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


def process_large_image(model, img_path, file_name, tile_size=256, overlap=0.2, folder_model=None, model_name=None):
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
        os.makedirs(os.path.join(config.DRIVE_PATH, config.BACKBONE, config.TOTAL_PATH, 'test_noise'), exist_ok=True)
        output_path = os.path.join(config.DRIVE_PATH, config.BACKBONE, config.TOTAL_PATH, 'test_noise', base_name + '_predicted.jpg')
    else:
        os.makedirs(os.path.join(config.DRIVE_PATH, config.BACKBONE, config.TOTAL_PATH, 'model_opt', folder_model, 'test_noise'), exist_ok=True)
        output_path = os.path.join(config.DRIVE_PATH, config.BACKBONE, config.TOTAL_PATH, 'model_opt', folder_model, 'test_noise', base_name + '_predicted.jpg')
    print(f"Image saved in: {output_path}")
    image_predicted.save(output_path)
    return image_predicted

def process_large_frame(model, frame, tile_size=256, overlap=0.1, folder_model=None, model_name=None):
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

def do_prediction_video(model, video_path, file_name, high_res=False, folder_model=None, model_name=None):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    base_name = os.path.splitext(file_name)[0]
    if folder_model is None:
        os.makedirs(os.path.join(config.DRIVE_PATH, config.BACKBONE, config.TOTAL_PATH, 'test_noise'), exist_ok=True)
        output_path = os.path.join(config.DRIVE_PATH, config.BACKBONE, config.TOTAL_PATH, 'test_noise',base_name + '_predicted.mp4')
    else:
        os.makedirs(os.path.join(config.DRIVE_PATH, config.BACKBONE, config.TOTAL_PATH, 'model_opt', folder_model, 'test_noise'), exist_ok=True)
        output_path = os.path.join(config.DRIVE_PATH, config.BACKBONE, config.TOTAL_PATH, 'model_opt', folder_model, 'test_noise',base_name + '_predicted.mp4')
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
            bboxes = process_large_frame(model, frame, folder_model=folder_model, model_name=model_name)
            frame_with_bboxes = draw_bboxes_on_frame(frame, bboxes)
        else:
            bboxes = process_frame(frame, model, folder_model=folder_model, model_name=model_name)
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
               folder_model = None, model_name = None):
    model, _ = load_model(folder_model=folder_model, model_name=model_name)
    model.eval()


    '''
    CASE2: Add NoisyLayer to the model if CASE2 flag is True
    '''
    if case1:
        slice_1 = slices_case1[0] if slices_case1[0] < slices_case1[1] else slices_case1[1]-1
        eval_noise.register_hooks(model, slice_1 = slice_1, 
                        slices_case1 = slices_case1, 
                        percentage_layers_level_1 = percentage_layers_case_1, 
                        interval_1 = interval_1,
                        percentage_tensors_1=percentage_tensors_1)
    
    '''
    CASE3: Add noise to model weights if CASE3 flag is True
    '''
    if case2:
        # Layer_array contains the name of the layers that are weights or bias
        layer_array = [name for name, layer in model.named_parameters() if 'weight' or 'bias' in name]
        total_param = len(layer_array)

        # limit_n is the number of layers that we will affect in each slice
        limit_n = math.ceil(total_param//slices_case2[1])

        slice_2 = slices_case2[0] if slices_case2[0] < slices_case2[1] else slices_case2[1]-1
        # max_limit is the number of layers that we belongs in the corresponding slice
        max_limit = limit_n*(slice_2+1) if limit_n*(slice_2+1) < total_param else total_param

        # n_samples is the number of layers that we will affect in the corresponding slice
        n_samples = math.ceil((max_limit - limit_n*slice_2)*percentage_layers_case_2)

        # layers_random is the name of the layers that we will affect in the corresponding slice
        layers_random = random.sample(layer_array[limit_n*slice_2:max_limit], n_samples)

        max_limit_2 = []
        for sl2 in range(slices_case2[1]):
            max_limit = limit_n*(sl2+1) - limit_n*sl2 +1
            max_limit = total_param  - max_limit*sl2 if max_limit*(sl2+1) >= total_param else max_limit
            max_limit_2.append(max_limit)


        n_weights_affected = 0
        for indx, (name, param) in enumerate(model.named_parameters()):
            if name in layers_random:
                

                
                n_weights_affected += 1
                max_val = param.max().item()
                bits_int_max = interval_2[1] if interval_2[1] <= len(bin(int(max_val))[2:])+1 else len(bin(int(max_val))[2:])+1 # +1 for the sign bit


                # exponential = bits_int_max -1 # Test Case  if we want to see how it changes the sign
                # exponential = 0 # Test Case if we want to see how it changes the more significant bit
                # exponential = -1 #< 0 Test Case if we want to see how it change an specific bit of the fractional part
                with torch.no_grad():
                    # exponential is a random number between the interval_2[0] and bits_int_max
                    exponential = int(torch.empty(1).uniform_(interval_2[0], bits_int_max).item())
                    # exponential = -1
                    param_flatten = torch.flatten(param, start_dim = 0)
                    
                    # As param_flatten is a view of the original tensor, we can modify a random index of the flatten tensor
                    random_index_array = random.sample(range(0, param_flatten.numel() - 1), math.ceil(param_flatten.numel()*percentage_tensors_2)) # Choose a random index of the parameters
                    # random_index_array = [0]

                    for random_index in random_index_array:
                        if exponential == bits_int_max-1:
                            binary_int_part = [int(bit) for bit in bin(abs(int(param_flatten[0])))[2:]]
                            binary_fractional_part = eval_noise.convert_fractional_to_binary(param_flatten[random_index])


                            if len(binary_fractional_part) != 0:
                                param_flatten[random_index] = -param_flatten[random_index] # Change the sign of the parameter


                            param = param_flatten.view(param.size())
                        else:
                            if exponential < 0:
                                exponential = abs(exponential)
                                binary_fractional_part = eval_noise.convert_fractional_to_binary(param_flatten[random_index])
                                signo = np.sign(param_flatten[random_index].item())

                                binary_int_part = [int(bit) for bit in bin(abs(int(param_flatten[0])))[2:]]
                                binary_int_part.reverse()
                                
                                exponential = len(binary_fractional_part)-1 if exponential>=len(binary_fractional_part) else exponential
                                
                                if len(binary_fractional_part) != 0:
                                    binary_fractional_part[exponential-1] = 0 if binary_fractional_part[exponential-1] else 1

                                param_flatten[random_index] = signo*abs((eval_noise.convert_fractional_binary_to_fractional(abs(int(param_flatten[random_index])), binary_fractional_part)))


                            elif exponential >=0:
                                param_flatten = torch.flatten(param, start_dim = 0)
                                random_index = random.randint(0, param_flatten.numel() - 1) # Choose a random index of the parameters

                                binary_fractional_part = eval_noise.convert_fractional_to_binary(param_flatten[random_index])
                                binary_int_part = [int(bit) for bit in bin(abs(int(param_flatten[0])))[2:]]
                                binary_int_part.reverse()
                                signo = np.sign(param_flatten[random_index].item())
                                

                                exponential = len(binary_int_part)-1 if exponential >= len(binary_int_part) else exponential
                                
                                if len(binary_int_part) != 0:
                                    binary_int_part[exponential] = 0 if binary_int_part[exponential] else 1
                                param_flatten[random_index] = signo*abs(eval_noise.convert_fractional_binary_to_fractional(eval_noise.convert_int_binary_to_int(binary_int_part), binary_fractional_part))
                                


                            param = param_flatten.view(param.size())

                            dict(model.named_parameters())[name].data = param
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
                process_large_image(model, file_path, file_name, folder_model=folder_model, model_name=model_name)
            else:
                do_prediction_image(model, file_path, file_name, folder_model=folder_model, model_name=model_name)
        elif ext == '.mp4':
            cap = cv2.VideoCapture(file_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            do_prediction_video(model, file_path, file_name, high_res=(width > 256 or height > 256), folder_model=folder_model, model_name=model_name)
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
