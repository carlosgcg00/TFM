import torch
import torch.optim as optim
from utils.utils import (
    mean_average_precision_noise,
    get_loaders,
    
)
import config
import os
from utils.save_results import (
    save_results_noise,
    plot_results_noise,
)
from YOLOv8.utils_YOLOv8 import load_model, get_bboxes_yolov8
from torch import nn
import random
import numpy as np
import time
import math
from ultralytics import YOLO
from radiation.SET_activation_functions import register_hooks
from radiation.SEU_weights import SEU_simmulation



global total_layers_affected_1
total_layers_affected_1 = 0
global total_layers_affected_2
total_layers_affected_2 = 0
global max_limit_1
max_limit_1 = []
global max_limit_2
max_limit_2 = []
 
            


def test_noise(case1=False, percentage_layers_case_1 = [0], slices_case1 = 0, interval_1 = (-12, 2), percentage_tensors_1=0.1,
               case2=False, percentage_layers_case_2 = [0], slices_case2 = 0, interval_2 = (-12, 2), percentage_tensors_2=0.1,
               folder_model = None, path = 'runs/detect/train', debug=False):
    print(f'Test with case1: {case1}, case2: {case2}')
    global total_layers_affected_1
    global total_layers_affected_2
    global max_limit_1
    global max_limit_2
    
    ############################################
    #           Define the loss function       #
    ############################################

    # Load data into data loader
    _, val_loader,_ = get_loaders()

    ############################################
    #       Folder to save the results         #
    ############################################
    if folder_model is None:
        folder_save_results = os.path.join(path, f'results_noise')
    else:
        folder_save_results = os.path.join(path, 'model_opt', folder_model, f'results_noise')

    if case1 and slices_case1[1] == slices_case1[0]:
        range_slices_case1 = range(0,slices_case1[1])
    else:
        range_slices_case1 = [slices_case1[0]]
    
    if case2 and slices_case2[1] == slices_case2[0]:
        range_slices_case2 = range(0,slices_case2[1])
    else:
        range_slices_case2 = [slices_case2[0]]

    ############################################
    #         File to save the results         #
    ############################################

    result_name_file = 'results_noise'
    if case2:
        result_name_file = f'case2_{result_name_file}_slices2_{slices_case2[1]}_interval2_{interval_2[0]}_{interval_2[1]}_p2_{percentage_tensors_2}'
    if case1:
        result_name_file = f'case1_{result_name_file}_slices_1_{slices_case1[1]}_interval1_{interval_1[0]}_{interval_1[1]}_p1_{percentage_tensors_1}'
    # Eliminate previous file if exists
    result_name_file = f'{result_name_file}.txt'

    if folder_model is None:
        if os.path.exists(os.path.join(path, 'results_noise', result_name_file)):
            os.remove(os.path.join(path, 'results_noise', result_name_file))
            print(f'File {result_name_file} removed')
    else:
        if os.path.exists(os.path.join(path, 'model_opt', folder_model, 'results_noise', result_name_file)):
            os.remove(os.path.join(path, 'model_opt', folder_model, 'results_noise', result_name_file))
            print(f'File {result_name_file} removed')
            
    print(f'Folder to save results: {os.path.join(path, folder_save_results)}')
    print(f'Results will be saved in: {result_name_file}')    
    for percentage_layers_level_1 in percentage_layers_case_1:
        for percentage_layers_level_2 in percentage_layers_case_2:
            for slice_1 in range_slices_case1:
                for slice_2 in range_slices_case2:
                    torch.cuda.empty_cache()
                    model  = load_model(folder_model=folder_model, path=path)
                    print("############################################")
                    if case1:
                        print(f'Noise level 1: Slice: [{slice_1+1}, {slices_case1[1]}], Percentage: {percentage_layers_level_1}')
                    if case2:
                        print(f'Noise level 2: Slice: [{slice_2+1}, {slices_case2[1]}], Percentage: {percentage_layers_level_2}')
                    print("############################################")

                    '''
                    case1: Add NoisyLayer to the model if case1 flag is True
                    '''
                    if case1:
                        register_hooks(model, slice_1 = slice_1, 
                                       slices_case1 = slices_case1, 
                                       percentage_layers_level_1 = percentage_layers_level_1, 
                                       interval_1 = interval_1,
                                       percentage_tensors_1=percentage_tensors_1, debug=debug)
                        

                    '''
                    case2: Add noise to model weights if case2 flag is True 
                    '''
                    if case2:
                        val_mAP_50_array = []

                        for _ in range(1):
                            model = load_model(folder_model=folder_model, path=path)
                            model = SEU_simmulation(model, slices_case2, slice_2, percentage_layers_level_2, interval_2, percentage_tensors_2=percentage_tensors_2, debug=debug)
                            
                            ############################################
                            #       Evaluating for valid               #
                            ############################################
                            
                                
                            pred_boxes, target_boxes = get_bboxes_yolov8(val_loader, model, threshold=0.5, max_batch = 3)

                            val_mAP_50, val_mAP_75, val_mAP_90 = mean_average_precision_noise(pred_boxes, target_boxes, iou_thresholds=[0.5, 0.75, 0.9], box_format="midpoint", num_classes=config.NUM_CLASSES, mode='Valid')
                            print(f"Valid: \t mAP@50: {val_mAP_50:.6f}, mAP@75: {val_mAP_75:.6f}, mAP@90: {val_mAP_90:.6f}")
                            val_mAP_50_array.append(val_mAP_50)
                            # print(f'array_param_shape: {array_param_shape}')
                        val_mAP_50 = np.mean(val_mAP_50_array)

                    if case1:
                        val_mAP_50_array = []
                        loss_val_array = []
                        for _ in range(1):
                            pred_boxes, target_boxes = get_bboxes_yolov8(val_loader, model, threshold=0.5,  max_batch = 3)
                            val_mAP_50, val_mAP_75, val_mAP_90 = mean_average_precision_noise(pred_boxes, target_boxes, iou_thresholds=[0.5, 0.75, 0.9], box_format="midpoint", num_classes=config.NUM_CLASSES, mode='Valid')
                            print(f"Valid: \t mAP@50: {val_mAP_50:.6f}, mAP@75: {val_mAP_75:.6f}, mAP@90: {val_mAP_90:.6f}")
                            val_mAP_50_array.append(val_mAP_50)
                        val_mAP_50 = np.mean(val_mAP_50_array)
                    
                    print(f"Valid: \t mAP@50: {val_mAP_50:.6f}, mAP@75: {val_mAP_75:.6f}, mAP@90: {val_mAP_90:.6f}")
                    save_results_noise( total_layers_affected_1=total_layers_affected_1, slices1=slice_1, nslices1=slices_case1[1],
                                        total_layers_affected_2=total_layers_affected_2, slices2=slice_2, nslices2=slices_case2[1],
                                        mAP_50=val_mAP_50, mAP_75=val_mAP_75, mAP_90=val_mAP_90, mean_loss=0, 
                                        file_name=result_name_file, folder_save_results = folder_save_results)
                    plot_results_noise(result_name_file, folder_save_results, case1, case2, max_limit_1=max_limit_1, max_limit_2=max_limit_2)

if __name__ == "__main__":
    case1 = False
    case2 = True
    percentage_layers_case_1  = [0.5] if case1 else [0]
    percentage_layers_case_2 = [0.5] if case2 else [0]
    # Con este parámetro seleccionamos la zona en la que queremos afectar a las activacionies o pesos
    # es decir, slices = [slice_i, n_slices], con esto dividimos todas las capas en n_slices y afectamos a las capas que esten en slice_i
    '''
    |slice 0 | slice 1 | slice 2 | slice 3 | slice 4 | 
    '''
    interval_1 = (-2, 2)
    interval_2 = (-12, 2)
    slices_case1 = [3, 3] 
    slices_case2 = [3,3]
    percentage_tensors_1 = 0.0000001
    percentage_tensors_2 = 0.0000001

    test_noise(case1=case1, percentage_layers_case_1=percentage_layers_case_1, slices_case1 = slices_case1, interval_1 = interval_1, percentage_tensors_1= percentage_tensors_1,
               case2=case2, percentage_layers_case_2=percentage_layers_case_2, slices_case2 = slices_case2, interval_2 = interval_2, percentage_tensors_2= percentage_tensors_2,)

