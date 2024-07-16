import os
from PIL import Image, ImageDraw, ImageFont
import time
from tqdm import tqdm
from torchvision.transforms import v2   
import sys
sys.path.append(os.getcwd())
import config
from utils.utils import (
    get_test_loader,
    non_max_suppression,
    cellboxes_to_boxes,
    mean_average_precision,
    cellboxes_to_boxes_test
    )

from utils.load_save_model import load_model
from utils.plot_predictions import plot_image_with_pil_
from predict.predict_functions import do_prediction_tile


def get_bboxes_test(model, test_loader, tile_size = 256, overlap = 0.2, threshold = 0.4, iou_threshold = 0.4, path_opt_model=None):
    model.eval()
    all_pred_boxes = []
    all_true_boxes = []
    loop = tqdm(test_loader, total=len(test_loader), leave=True)
    image_times = []
    tile_times = []
    for idx, (x, labels) in enumerate(loop):
        if path_opt_model is None: 
                os.makedirs(os.path.join(config.ROOT_DIR, config.BACKBONE, config.TOTAL_PATH, 'test'), exist_ok=True)
                output_path = os.path.join(config.ROOT_DIR, config.BACKBONE, config.TOTAL_PATH, 'test', f'{idx}' + '_predicted.jpg')
        else:
            os.makedirs(os.path.join(config.ROOT_DIR,f'{config.BACKBONE}/{config.TOTAL_PATH}/model_opt/{path_opt_model}', 'test'), exist_ok=True)
            output_path = os.path.join(config.ROOT_DIR,f'{config.BACKBONE}/{config.TOTAL_PATH}/model_opt/{path_opt_model}', 'test', f'{idx}' + '_predicted.jpg')
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
        
        image_predicted.save(output_path) 
    print(f"Inference large image, Time: {sum(image_times)/len(image_times):.3f} seconds, FPS: {1 / sum(image_times)/len(image_times)}")
    print(f"Average inference time per tile: {sum(tile_times)/len(tile_times):.6f} seconds, Time Tiles * n_tiles: {n_tiles*sum(tile_times)/len(tile_times):.3f}")
 

    return all_pred_boxes, all_true_boxes
        
    


def test_prediction(tile_size=256, overlap=0.2, threshold=0.4, iou_threshold=0.4, path_opt_model=None):
    model, _ = load_model(folder_model=path_opt_model)
    test_loader = get_test_loader()

    model.eval()
    
    pred_boxes, target_boxes = get_bboxes_test(model, test_loader, tile_size, overlap, threshold, iou_threshold, path_opt_model=path_opt_model)
    test_mAP50, test_mAP75, test_mAP90 = mean_average_precision(pred_boxes, target_boxes, iou_thresholds=[0.5, 0.75, 0.9], box_format="midpoint", num_classes=config.NUM_CLASSES, mode='Test')

    print(f"Tile size: {tile_size}, Overlap: {overlap}")
    print(f"mAP50: {test_mAP50}, mAP75: {test_mAP75}, mAP90: {test_mAP90}")
    
if __name__ == '__main__':
    test_prediction()