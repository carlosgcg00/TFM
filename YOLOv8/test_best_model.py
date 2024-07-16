import torch
import sys
import os
sys.path.append(os.getcwd())
from utils.utils import (
    get_test_loader,
    non_max_suppression,
    cellboxes_to_boxes_test,
    mean_average_precision
    )
from utils.plot_predictions import plot_image_with_pil_
from YOLOv8.utils_YOLOv8 import load_model
import config
import os
from PIL import Image, ImageDraw, ImageFont
import time
from tqdm import tqdm
from torchvision.transforms import v2   
from ultralytics import YOLO



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





def get_bboxes_test(model, test_loader, tile_size = 2560, overlap = 0.2, threshold = 0.4, iou_threshold = 0.4,path = 'runs/detect/train', folder_model=None):
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
    print(f"Images saved in {os.path.join(path, 'model_opt', folder_model, 'test')}")
 

    return all_pred_boxes, all_true_boxes        
    


def test_prediction(tile_size=512, overlap=0.2, threshold=0.4, iou_threshold=0.4,path = 'runs/detect/train', folder_model=None):
    model = load_model(path=path, folder_model=folder_model)
    test_loader = get_test_loader()
  
    pred_boxes, target_boxes = get_bboxes_test(model, test_loader, tile_size, overlap, threshold, iou_threshold, path = path, folder_model=folder_model)
    test_mAP50, test_mAP75, test_mAP90 = mean_average_precision(pred_boxes, target_boxes, iou_thresholds=[0.5, 0.75, 0.9], box_format="midpoint", num_classes=config.NUM_CLASSES, mode='Test')

    print(f"Tile size: {tile_size}, Overlap: {overlap}")
    print(f"mAP50: {test_mAP50}, mAP75: {test_mAP75}, mAP90: {test_mAP90}")
    
if __name__ == '__main__':
    test_prediction()