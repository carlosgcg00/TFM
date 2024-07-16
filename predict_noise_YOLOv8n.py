import torch
import torch.optim as optim
from backbone import vgg16, resnet50, efficientnet_b0
from tinyissimo_model import tinyissimoYOLO
from ext_tinyissimo_model import ext_tinyissimoYOLO
from bed_model import bedmodel
from YOLOv1 import Yolov1
from utils import (
    get_test_loader,
    non_max_suppression,
    get_bboxes_yolov8,
    get_loaders,
    cellboxes_to_boxes_test,
    mean_average_precision
    )
import config
import os
import numpy as np
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import time
import cv2
from tqdm import tqdm
from torchvision.transforms import v2   
from ultralytics import YOLO
import math
import random
from eval_noise_yolov8n import convert_fractional_to_binary, convert_fractional_binary_to_fractional, convert_int_binary_to_int, string_bit

debug = 0
path = '/home/jovyan/Carlos_Gonzalez/YOLO/runs/detect/train'

def load_model(folder_model=None, model_name=None):
    model = YOLO(os.path.join(path,'weights/best.pt'))
    _, val_loader, _ = get_loaders()
    _, _ = get_bboxes_yolov8(val_loader, model, threshold=0.5)
    if folder_model is None:
        print(f'Model loaded from {os.path.join(path,"weights/best.pt")}')
    else:
        checkpoint = torch.load(os.path.join(path, 'model_opt', folder_model, model_name), map_location=config.DEVICE)
        chk = checkpoint['state_dict']
        layers_of_interest = []
        for param_indx, (name, param) in enumerate(model.named_parameters()):
            if name.endswith('weight') or name.endswith('bias'):
                layers_of_interest.append(name)
        for param_indx, (name, param) in enumerate(model.named_parameters()):
            dict(model.named_parameters())[name].data.copy_(chk[name])
        print(f'Model loaded from {os.path.join(path, "model_opt", folder_model, model_name)}')
    return model

def do_prediction_tile(model, image_tile, iou_threshold=0.5, threshold=0.4):
    image_array, boxes = config.predict_transforms(image_tile, [])
    x = image_array.unsqueeze(0).to(config.DEVICE)
    with torch.no_grad():
        bboxes = []
        start_time = time.time()
        prediction_yolo = model.predict(x, verbose=False)
        end_time = time.time()
        for predict in prediction_yolo:
            for i in range(len(predict.boxes.cls)):
                class_prd = predict.boxes.cls[i].cpu().tolist()
                conf = predict.boxes.conf[i].cpu().tolist()
                x = predict.boxes.xywhn[i][0].cpu().tolist()
                y = predict.boxes.xywhn[i][1].cpu().tolist()
                w = predict.boxes.xywhn[i][2].cpu().tolist()
                h = predict.boxes.xywhn[i][3].cpu().tolist()

                prediction = [class_prd, conf, x, y, w, h]
                bboxes.append(prediction)
    return bboxes, end_time - start_time

def plot_image_with_pil_(image, pred_boxes, true_boxes=None):
    class_labels = config.AIRBUS_LABELS
    num_classes = len(class_labels)
    colors = config.colors[:num_classes]
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for box in pred_boxes:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = int(box[0])
        confidence_score = box[1]
        x, y, w, h = box[2], box[3], box[4], box[5]
        upper_left_x = (x - w / 2) * width
        upper_left_y = (y - h / 2) * height
        lower_right_x = (x + w / 2) * width
        lower_right_y = (y + h / 2) * height
        draw.rectangle([upper_left_x, upper_left_y, lower_right_x, lower_right_y], outline=colors[class_pred], width=3)
        draw.text((upper_left_x, upper_left_y), f'{confidence_score:.2f} {class_labels[class_pred]}', fill=colors[class_pred], font=ImageFont.truetype("font_text/04B_08__.TTF", 20))
    
    if true_boxes is not None:
        for box in true_boxes:
            if box[1] > 0.4:
                assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
                class_pred = int(box[0])
                confidence_score = box[1]
                x, y, w, h = box[2], box[3], box[4], box[5]
                upper_left_x = (x - w / 2) * width
                upper_left_y = (y - h / 2) * height
                lower_right_x = (x + w / 2) * width
                lower_right_y = (y + h / 2) * height
                draw.rectangle([upper_left_x, upper_left_y, lower_right_x, lower_right_y], outline='red', width=5)
    return image

def get_bboxes_test(model, test_loader, tile_size=2560, overlap=0.2, threshold=0.4, iou_threshold=0.4, folder_model=None):
    all_pred_boxes = []
    all_true_boxes = []
    loop = tqdm(test_loader, total=len(test_loader), leave=True)
    image_times = []
    tile_times = []
    for idx, (x, labels) in enumerate(loop):
        x = x.to(config.DEVICE)
        image = v2.ToPILImage()(x.view(3, 2560, 2560))
        width, height = image.size
        stride = int(tile_size * (1 - overlap))
        
        all_bboxes = []
        start_time = time.time()
        n_tiles = 0 
        for y in range(0, height, stride):
            for x in range(0, width, stride):
                n_tiles += 1
                right = min(x + tile_size, width)
                bottom = min(y + tile_size, height)
                tile = image.crop((x, y, right, bottom))
                
                if tile.size != (tile_size, tile_size):
                    tile = Image.new('RGB', (tile_size, tile_size), (0, 0, 0))
                    tile.paste(image.crop((x, y, right, bottom)), (0, 0))
                
                bboxes, tile_time = do_prediction_tile(model, tile, iou_threshold=iou_threshold, threshold=threshold)
                tile_times.append(tile_time)
                for bbox in bboxes:
                    bbox[2] = (bbox[2] * tile_size + x) / width
                    bbox[3] = (bbox[3] * tile_size + y) / height
                    bbox[4] = bbox[4] * tile_size / width
                    bbox[5] = bbox[5] * tile_size / height
                    
                    all_bboxes.append(bbox)
        all_bboxes = non_max_suppression(all_bboxes, iou_threshold=iou_threshold, threshold=threshold, box_format="midpoint")
        end_time = time.time()
        image_times.append(end_time - start_time)
        true_bboxes = cellboxes_to_boxes_test(labels, SPLIT_SIZE=100, NUM_BOXES=2, NUM_CLASSES=config.NUM_CLASSES)
        for nms_box in all_bboxes:
            all_pred_boxes.append([idx] + nms_box)
        for box in true_bboxes[0]:
            if box[1] > threshold:
                all_true_boxes.append([idx] + box)
        image_predicted = plot_image_with_pil_(image, all_bboxes, true_bboxes[0])
        if folder_model is None:
            os.makedirs(os.path.join(path, 'test_noise'), exist_ok=True)
            output_path = os.path.join(path, 'test_noise', f'{idx}' + '_predicted.jpg')
        else:
            os.makedirs(os.path.join(path, 'model_opt', folder_model, 'test_noise'), exist_ok=True)
            output_path = os.path.join(path, 'model_opt', folder_model, 'test_noise', f'{idx}' + '_predicted.jpg') 
        image_predicted.save(output_path)
    print(f"Inference large image, Time: {sum(image_times)/len(image_times):.3f} seconds, FPS: {1 / sum(image_times)/len(image_times)}")
    print(f"Average inference time per tile: {sum(tile_times)/len(tile_times):.6f} seconds, Time Tiles * n_tiles: {n_tiles*sum(tile_times)/len(tile_times):.3f}")
    print(f"Images saved in {os.path.join( path, 'model_opt', folder_model, 'test_noise' ) }")
    return all_pred_boxes, all_true_boxes

def draw_bboxes_on_frame(frame, bboxes):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image_with_bboxes = plot_image_with_pil_(image, bboxes)
    return cv2.cvtColor(np.array(image_with_bboxes), cv2.COLOR_RGB2BGR)

def process_frame(frame, model, iou_threshold=0.5, threshold=0.4):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image_array, _ = config.predict_transforms(image, [])
    x = image_array.unsqueeze(0).to(config.DEVICE)
    with torch.no_grad():
        bboxes = []
        prediction_yolo = model.predict(x, verbose=False)
        for predict in prediction_yolo:
            for i in range(len(predict.boxes.cls)):
                class_prd = predict.boxes.cls[i].cpu().tolist()
                conf = predict.boxes.conf[i].cpu().tolist()
                x = predict.boxes.xywhn[i][0].cpu().tolist()
                y = predict.boxes.xywhn[i][1].cpu().tolist()
                w = predict.boxes.xywhn[i][2].cpu().tolist()
                h = predict.boxes.xywhn[i][3].cpu().tolist()
                prediction = [class_prd, conf, x, y, w, h]
                bboxes.append(prediction)
    return bboxes

def process_large_frame(model, frame, tile_size=640, overlap=0.2, iou_threshold=0.5, threshold=0.4):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    width, height = image.size
    stride = int(tile_size * (1 - overlap))
    all_bboxes = []
    tile_times = []
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            right = min(x + tile_size, width)
            bottom = min(y + tile_size, height)
            tile = image.crop((x, y, right, bottom))
            if tile.size != (tile_size, tile_size):
                tile = Image.new('RGB', (tile_size, tile_size), (0, 0, 0))
                tile.paste(image.crop((x, y, right, bottom)), (0, 0))
            bboxes, tile_time = do_prediction_tile(model, tile, iou_threshold=iou_threshold, threshold=threshold)
            tile_times.append(tile_time)
            for bbox in bboxes:
                bbox[2] = (bbox[2] * tile_size + x) / width
                bbox[3] = (bbox[3] * tile_size + y) / height
                bbox[4] = bbox[4] * tile_size / width
                bbox[5] = bbox[5] * tile_size / height
                all_bboxes.append(bbox)
    all_bboxes = non_max_suppression(all_bboxes, iou_threshold=iou_threshold, threshold=threshold, box_format="midpoint")
    return all_bboxes, tile_times

def do_prediction_video(video_path, file_name, tile_size=640, overlap=0.0, iou_threshold=0.2, threshold=0.4, folder_model=None, model_name=None):
    model = load_model(folder_model=folder_model, model_name=model_name)

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    base_name = os.path.splitext(file_name)[0]
    if folder_model is None:
        os.makedirs(os.path.join(path, 'test_noise'), exist_ok=True)
        output_path = os.path.join(path, 'test_noise', base_name + '_predicted.mp4')
    else:
        os.makedirs(os.path.join(path, 'model_opt', folder_model, 'test_noise'), exist_ok=True)
        output_path = os.path.join(path, 'model_opt', folder_model, 'test_noise', base_name + '_predicted.mp4')
    print(f"Output video path: {output_path}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    inference_times = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break        
        start_time = time.time()
        if width > tile_size or height > tile_size:
            bboxes, tile_times = process_large_frame(model, frame, tile_size=tile_size, overlap=overlap, iou_threshold=iou_threshold, threshold=threshold)
            frame_with_bboxes = draw_bboxes_on_frame(frame, bboxes)
            inference_times.extend(tile_times)
        else:
            bboxes = process_frame(frame, model, iou_threshold=iou_threshold, threshold=threshold)
            frame_with_bboxes = draw_bboxes_on_frame(frame, bboxes)
            inference_times.append(time.time() - start_time)
        
        out.write(frame_with_bboxes)
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    average_inference_time = sum(inference_times) / len(inference_times)
    print(f"Average inference time per frame: {average_inference_time:.6f} seconds")

def test_prediction(tile_size=512, overlap=0.2, threshold=0.4, iou_threshold=0.4,
                    case2=False, percentage_layers_case_2=0, slices_case2=0, interval_2=(-12, 2), percentage_tensors_2=0.1,
                    folder_model=None, model_name=None):
    model = load_model(folder_model=folder_model, model_name=model_name)
    test_loader = get_test_loader()

    if case2:
        slice_2 = slices_case2[0]
        layer_array = [name for name, layer in model.named_parameters() if 'weight' or 'bias' in name]
        total_param = len(layer_array)

        limit_n = math.ceil(total_param // slices_case2[1])
        max_limit = limit_n * (slice_2 + 1) if limit_n * (slice_2 + 1) < total_param else total_param
        n_samples = math.ceil((max_limit - limit_n * slice_2) * percentage_layers_case_2)
        layers_random = random.sample(layer_array[limit_n * slice_2:max_limit], n_samples)
        max_limit_2 = []
        for sl2 in range(slices_case2[1]):
            max_limit = limit_n * (sl2 + 1) - limit_n * sl2 + 1
            max_limit = total_param - max_limit * sl2 if max_limit * (sl2 + 1) >= total_param else max_limit
            max_limit_2.append(max_limit)
        
        total_layers_affected_2 = math.ceil(max_limit_2[slice_2] * percentage_layers_case_2)
        if debug:
            print(f'Number of layers affected: {len(layers_random)}')

        n_weights_affected = 0
        array_param_shape = 0
        for indx, (name, param) in enumerate(model.named_parameters()):
            if name in layers_random:
                if debug:
                    print(f'Name: {name}, Shape: {param.shape}, Flattened: {param.flatten().shape}')
                array_param_shape += param.flatten().shape[0]
                n_weights_affected += 1
                max_val = param.max().item()
                bits_int_max = interval_2[1] if interval_2[1] <= len(bin(int(max_val))[2:]) + 1 else len(bin(int(max_val))[2:]) + 1

                with torch.no_grad():
                    exponential = int(torch.empty(1).uniform_(interval_2[0], bits_int_max).item())
                    param_flatten = torch.flatten(param, start_dim=0)
                    param_flatten_numel = param_flatten.numel()
                    if param_flatten_numel > 1:
                        random_index_array = random.sample(range(0, param_flatten_numel - 1), math.ceil(param_flatten_numel * percentage_tensors_2))
                    else:
                        random_index_array = [0]
                    if debug:
                        print(f'Random index: {len(random_index_array)}')
                    for random_index in random_index_array:
                        if exponential == bits_int_max - 1:
                            binary_int_part = [int(bit) for bit in bin(abs(int(param_flatten[0])))[2:]]
                            binary_fractional_part = convert_fractional_to_binary(param_flatten[random_index])
                            if debug:
                                string_bits = string_bit(np.sign(param_flatten[random_index].item()), binary_int_part, binary_fractional_part)
                                print(f'Random index: {random_index} Exponential:  {exponential} \n\tParam: {param_flatten[random_index]}, Bits: {string_bits}')
                            if len(binary_fractional_part) != 0:
                                param_flatten[random_index] = -param_flatten[random_index]
                            if debug:
                                string_bits = string_bit(np.sign(param_flatten[random_index].item()), binary_int_part, binary_fractional_part)
                                print(f'\tParam: {param_flatten[random_index]}, Bits: {string_bits}')
                            param = param_flatten.view(param.size())
                        else:
                            if exponential < 0:
                                exponential = abs(exponential)
                                binary_fractional_part = convert_fractional_to_binary(param_flatten[random_index])
                                signo = np.sign(param_flatten[random_index].item())
                                binary_int_part = [int(bit) for bit in bin(abs(int(param_flatten[0])))[2:]]
                                binary_int_part.reverse()
                                if debug:
                                    string_bits = string_bit(signo, binary_int_part, binary_fractional_part)
                                    print(f'Random index: {random_index} Exponential:  {-exponential} \n\tParam: {param_flatten[random_index]}, Bits: {string_bits}')
                                exponential = len(binary_fractional_part) - 1 if exponential >= len(binary_fractional_part) else exponential
                                if len(binary_fractional_part) != 0:
                                    binary_fractional_part[exponential - 1] = 0 if binary_fractional_part[exponential - 1] else 1
                                param_flatten[random_index] = signo * abs((convert_fractional_binary_to_fractional(abs(int(param_flatten[random_index])), binary_fractional_part)))
                                if debug:
                                    string_bits = string_bit(signo, binary_int_part, binary_fractional_part)
                                    print(f'\tParam: {param_flatten[random_index]}, Bits: {string_bits}')
                            elif exponential >= 0:
                                param_flatten = torch.flatten(param, start_dim=0)
                                random_index = random.randint(0, param_flatten.numel() - 1)
                                binary_fractional_part = convert_fractional_to_binary(param_flatten[random_index])
                                binary_int_part = [int(bit) for bit in bin(abs(int(param_flatten[0])))[2:]]
                                binary_int_part.reverse()
                                signo = np.sign(param_flatten[random_index].item())
                                if debug:
                                    string_bits = string_bit(signo, binary_int_part, binary_fractional_part)
                                    print(f'Random index: {random_index} Exponential:  {exponential} \n\tParam: {param_flatten[random_index]}, Bits: {string_bits}')
                                exponential = len(binary_int_part) - 1 if exponential >= len(binary_int_part) else exponential
                                if len(binary_int_part) != 0:
                                    binary_int_part[exponential] = 0 if binary_int_part[exponential] else 1
                                param_flatten[random_index] = signo * abs(convert_fractional_binary_to_fractional(convert_int_binary_to_int(binary_int_part), binary_fractional_part))
                                if debug:
                                    string_bits = string_bit(signo, binary_int_part, binary_fractional_part)
                                    print(f'\tParam: {param_flatten[random_index]}, Bits: {string_bits}')
                            param = param_flatten.view(param.size())
                            dict(model.named_parameters())[name].data = param
                            if debug:
                                print(f'Number of weights affected: {n_weights_affected}')

    pred_boxes, target_boxes = get_bboxes_test(model, test_loader, tile_size, overlap, threshold, iou_threshold, folder_model=folder_model)
    test_mAP50, test_mAP75, test_mAP90 = mean_average_precision(pred_boxes, target_boxes, iou_thresholds=[0.5, 0.75, 0.9], box_format="midpoint", num_classes=config.NUM_CLASSES, mode='test_noise')
    print(f"Tile size: {tile_size}, Overlap: {overlap}")
    print(f"mAP50: {test_mAP50}, mAP75: {test_mAP75}, mAP90: {test_mAP90}")

if __name__ == '__main__':
    test_prediction()


