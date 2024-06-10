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
    cellboxes_to_boxes,
    cellboxes_to_boxes_test,
    find_the_best_model,
    load_checkpoint,
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


def load_model(path_opt_model=None, boolean_optim=False):
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
    elif config.BACKBONE == 'bedmodel':
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

    if not boolean_optim:
        load_checkpoint(torch.load(os.path.join(config.DRIVE_PATH,f'{config.BACKBONE}/{config.TOTAL_PATH}/model/{best_model}')), model, optimizer)
        print(f"Load: {os.path.join(config.DRIVE_PATH,f'{config.BACKBONE}/{config.TOTAL_PATH}/model/{best_model}')}")
    else:
        load_checkpoint(torch.load(os.path.join(config.DRIVE_PATH,f'{config.BACKBONE}/{config.TOTAL_PATH}/model_opt/{path_opt_model}/YOLO_opt.pth.tar')), model, optimizer)
    model.eval()
    return model

def do_prediction_tile(model, image_tile, iou_threshold=0.5, threshold=0.4):
    image_array, boxes = config.predict_transforms(image_tile, [])
    x = image_array.unsqueeze(0).to(config.DEVICE)
    start_time = time.time()
    bboxes = cellboxes_to_boxes(model(x))
    bboxes = non_max_suppression(bboxes[0], iou_threshold=iou_threshold, threshold=threshold, box_format="midpoint")
    end_time = time.time()

    return bboxes, end_time - start_time

def plot_image_with_pil_(image, pred_boxes, true_boxes=None):
    """Edit the image to draw bounding boxes using PIL but does not show or save the image."""
    if config.DATASET == 'Airbus_256':
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
                draw.rectangle([upper_left_x, upper_left_y, lower_right_x, lower_right_y], outline='yellow', width=5)
    return image

def get_bboxes_test(model, test_loader, tile_size = 256, overlap = 0.2, threshold = 0.4, iou_threshold = 0.4):
    model.eval()
    all_pred_boxes = []
    all_true_boxes = []
    loop = tqdm(test_loader, total=len(test_loader), leave=True)
    for idx, (x, labels) in enumerate(loop):
        x = x.to(config.DEVICE)
        image = v2.ToPILImage()(x.view(3, 2560, 2560))
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
        true_bboxes = cellboxes_to_boxes_test(labels, SPLIT_SIZE = 100, NUM_BOXES=2, NUM_CLASSES=config.NUM_CLASSES)
        for nms_box in all_bboxes:
            all_pred_boxes.append([idx] + nms_box)
        for box in true_bboxes[0]:
            # many will get converted to 0 pred
            if box[1] > threshold:
                all_true_boxes.append([idx] + box)
        image_predicted = plot_image_with_pil_(image, all_bboxes, true_bboxes[0])
        os.makedirs(os.path.join(config.DRIVE_PATH, config.BACKBONE, config.TOTAL_PATH, 'test'), exist_ok=True)
        output_path = os.path.join(config.DRIVE_PATH, config.BACKBONE, config.TOTAL_PATH, 'test', f'{idx}' + '_predicted.jpg')
        image_predicted.save(output_path) 
    print(f"Inference large image, Time: {end_time - start_time} seconds, FPS: {1 / (end_time - start_time)}")
    print(f"Average inference time per tile: {sum(tile_times)/len(tile_times):.6f} seconds")
 

    return all_pred_boxes, all_true_boxes
        
    


def test_prediction(tile_size=256, overlap=0.2, threshold=0.4, iou_threshold=0.4):
    model = load_model()
    test_loader = get_test_loader()

    model.eval()
    
    pred_boxes, target_boxes = get_bboxes_test(model, test_loader, tile_size, overlap, threshold, iou_threshold)
    test_mAP50, test_mAP75, test_mAP90 = mean_average_precision(pred_boxes, target_boxes, iou_thresholds=[0.5, 0.75, 0.9], box_format="midpoint", num_classes=config.NUM_CLASSES, mode='Test')

    print(f"Tile size: {tile_size}, Overlap: {overlap}")
    print(f"mAP50: {test_mAP50}, mAP75: {test_mAP75}, mAP90: {test_mAP90}")
    
if __name__ == '__main__':
    test_prediction()