import torch
import torch.optim as optim
from utils.utils import (
    get_loaders,
    cellboxes_to_boxes,
)
import config
import os
from tqdm import tqdm
from torch import nn
import random
import numpy as np
import time
import math
from ultralytics import YOLO



def load_model(path = 'runs/detect/train', folder_model=None):
    
    model = YOLO(os.path.join(path,'weights/best.pt'))
    _, val_loader, _ = get_loaders()
    _, _ = get_bboxes_yolov8(val_loader, model, threshold=0.5, max_batch = 1)
    if folder_model is None:
        print(f'Model loaded from {os.path.join(path,"weights/best.pt")}')
    else:
        checkpoint = torch.load(os.path.join(path, 'model_opt', folder_model, 'YOLO_opt.pth.tar'), map_location=config.DEVICE)
        print(f'Model loaded from {os.path.join(path, "model_opt", folder_model, "YOLO_opt.pth.tar")}')
        chk = checkpoint['state_dict']

        layers_of_interest=[]
        for param_indx, (name, param) in enumerate(model.named_parameters()):
            # Verificar si el módulo tiene parámetros entrenables
            if name.endswith('weight') or name.endswith('bias'):
                layers_of_interest.append(name)
        for param_indx, (name, param) in enumerate(model.named_parameters()):
            dict(model.named_parameters())[name].data.copy_(chk[name])


    return model




def get_bboxes_yolov8(
    loader=None,
    model=None,
    threshold=0.4,
    max_batch = -1
):
    all_pred_boxes = []
    all_true_boxes = []

    train_idx = 0
    loop = tqdm(loader, leave=True)
    loop.set_description(f"Eval: Valid: ")
    for batch_idx, (x, labels) in enumerate(loop):
        x = x.to(config.DEVICE)
        labels = labels.to(config.DEVICE)
        # tensor of ceros of sizes x.shape[0] and config.split_size * config.split_size
        bboxes = torch.zeros(x.shape[0], config.SPLIT_SIZE * config.SPLIT_SIZE, 6).to(config.DEVICE)

        predictions = []
        with torch.no_grad():
            prediction_yolo = model.predict(x, verbose = False)
            # print(type(prediction_yolo))
            for sample in x:
                prediction_yolo = model.predict(sample.unsqueeze(0).to(config.DEVICE), verbose = False)
                predict_sample = []
                for predict in prediction_yolo:
                    for i in range(len(predict.boxes.cls)):
                        class_prd = predict.boxes.cls[i].cpu().tolist()
                        conf = predict.boxes.conf[i].cpu().tolist()
                        x_yolo = predict.boxes.xywhn[i][0].cpu().tolist()
                        y_yolo = predict.boxes.xywhn[i][1].cpu().tolist()
                        w_yolo = predict.boxes.xywhn[i][2].cpu().tolist()
                        h_yolo = predict.boxes.xywhn[i][3].cpu().tolist()
    
                        # prediction = [class_prd, conf, x_yolo, y_yolo, w_yolo, h_yolo]
                        predict_sample.append([class_prd, conf, x_yolo, y_yolo, w_yolo, h_yolo])
                    predictions.append(predict_sample)

        batch_size = x.shape[0]  
        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        # bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
         
            all_boxes = bboxes[idx]
            # same but considering all boxes, i.e.,
        
            for nms_box in predictions[idx]:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1
        if max_batch > 0:
            if batch_idx == max_batch: #Uncomment if you want to be faster
                break



    # all pred boxes are for all images = [train_idx, class_pred, prob_score, x, y, w, h]
    # all true boxes are for all images = [train_idx, class_pred, prob_score, x, y, w, h]
    return all_pred_boxes, all_true_boxes



def cellboxes_to_boxes_yolov8n(out):

    all_bboxes = []
    for predict in out:
        for i in range(len(predict.boxes.cls)):
            class_prd = predict.boxes.cls[i].cpu().tolist()
            conf = predict.boxes.conf[i].cpu().tolist()
            x_yolo = predict.boxes.xywhn[i][0].cpu().tolist()
            y_yolo = predict.boxes.xywhn[i][1].cpu().tolist()
            w_yolo = predict.boxes.xywhn[i][2].cpu().tolist()
            h_yolo = predict.boxes.xywhn[i][3].cpu().tolist()

            prediction = [class_prd, conf, x_yolo, y_yolo, w_yolo, h_yolo]

            all_bboxes.append(prediction)

    return all_bboxes