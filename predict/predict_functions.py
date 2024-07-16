
import os
from PIL import Image, ImageDraw, ImageFont
import time
import cv2
import sys
sys.path.append(os.getcwd())
print(sys.path)
import config
from utils.utils import (
    non_max_suppression,
    cellboxes_to_boxes,
    )

from utils.load_save_model import load_model
from utils.plot_predictions import plot_image_with_pil_


def do_prediction_tile(model, image_tile, iou_threshold=0.4, threshold=0.4, output_tile_path=None):
    image_array, boxes = config.predict_transforms(image_tile, [])
    x = image_array.unsqueeze(0).to(config.DEVICE)
    start_time = time.time()
    bboxes = cellboxes_to_boxes(model(x))
    bboxes = non_max_suppression(bboxes[0], iou_threshold=iou_threshold, threshold=threshold, box_format="midpoint")
    end_time = time.time()

    if output_tile_path:
        image_tile_predicted = plot_image_with_pil_(image_tile, bboxes)
        image_tile_predicted.save(output_tile_path)

    return bboxes, end_time - start_time
