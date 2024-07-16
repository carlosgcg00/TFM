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

        layers_of_interest=[]
        for param_indx, (name, param) in enumerate(model.named_parameters()):
            # Verificar si el módulo tiene parámetros entrenables
            if name.endswith('weight') or name.endswith('bias'):
                layers_of_interest.append(name)
        for param_indx, (name, param) in enumerate(model.named_parameters()):
            dict(model.named_parameters())[name].data.copy_(chk[name])

    return model

def do_prediction_tile(model, image_tile, iou_threshold=0.5, threshold=0.4):
    image_array, boxes = config.predict_transforms(image_tile, [])
    x = image_array.unsqueeze(0).to(config.DEVICE)
    with torch.no_grad():
        bboxes = []
        start_time = time.time()
        prediction_yolo = model.predict(x, verbose = False)
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
    """Edit the image to draw bounding boxes using PIL but does not show or save the image."""
    class_labels = config.AIRBUS_LABELS

    # Define colores utilizando PIL
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
        draw.rectangle([upper_left_x, upper_left_y, lower_right_x, lower_right_y], outline=colors[class_pred], width=5)
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



def get_bboxes_test(model, test_loader, tile_size = 256, overlap = 0.2, threshold = 0.4, iou_threshold = 0.4, folder_model=None):
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
                
                bboxes, tile_time = do_prediction_tile(model, tile, iou_threshold = iou_threshold, threshold = threshold)
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
        all_bboxes = non_max_suppression(all_bboxes, iou_threshold=iou_threshold, threshold=threshold, box_format="midpoint")
        end_time = time.time()
        image_times.append(end_time - start_time)
        true_bboxes = cellboxes_to_boxes_test(labels, SPLIT_SIZE = 100, NUM_BOXES=2, NUM_CLASSES=config.NUM_CLASSES)
        for nms_box in all_bboxes:
            all_pred_boxes.append([idx] + nms_box)
        for box in true_bboxes[0]:
            # many will get converted to 0 pred
            if box[1] > threshold:
                all_true_boxes.append([idx] + box)
        image_predicted = plot_image_with_pil_(image, all_bboxes, true_bboxes[0])
        if folder_model is None:
            os.makedirs(os.path.join(path, 'test'), exist_ok=True)
            output_path = os.path.join(path, 'test', f'{idx}' + '_predicted.jpg')
        else:
            os.makedirs(os.path.join(path, 'model_opt', folder_model, 'test'), exist_ok=True)
            output_path = os.path.join(path, 'model_opt', folder_model, 'test', f'{idx}' + '_predicted.jpg') 
        image_predicted.save(output_path) 
    print(f"Inference large image, Time: {sum(image_times)/len(image_times):.3f} seconds, FPS: {1 / sum(image_times)/len(image_times)}")
    print(f"Average inference time per tile: {sum(tile_times)/len(tile_times):.6f} seconds, Time Tiles * n_tiles: {n_tiles*sum(tile_times)/len(tile_times):.3f}")
    print(f'Image saved in {output_path}')
 

    return all_pred_boxes, all_true_boxes        
    


def test_prediction(tile_size=512, overlap=0.2, threshold=0.4, iou_threshold=0.4,folder_model=None, model_name=None):
    model = load_model(folder_model=folder_model, model_name=model_name)
    test_loader = get_test_loader()
  
    pred_boxes, target_boxes = get_bboxes_test(model, test_loader, tile_size, overlap, threshold, iou_threshold, folder_model=folder_model)
    test_mAP50, test_mAP75, test_mAP90 = mean_average_precision(pred_boxes, target_boxes, iou_thresholds=[0.5, 0.75, 0.9], box_format="midpoint", num_classes=config.NUM_CLASSES, mode='Test')

    print(f"Tile size: {tile_size}, Overlap: {overlap}")
    print(f"mAP50: {test_mAP50}, mAP75: {test_mAP75}, mAP90: {test_mAP90}")
    
if __name__ == '__main__':
    test_prediction()