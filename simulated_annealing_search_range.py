import torch
import torch.optim as optim
from backbone import vgg16, resnet50, efficientnet_b0
from tinyissimo_model import tinyissimoYOLO
from ext_tinyissimo_model import ext_tinyissimoYOLO
from bed_model import bedmodel
from YOLOv1 import Yolov1
import pickle
from utils import (
    get_loaders,
    get_bboxes,
    mean_average_precision,
    find_the_best_model,
    load_checkpoint,
    save_checkpoint
)
import config
import os
import random
import math
import time
import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
from loss import YoloLoss
import torch.nn as nn
from fxpmath import Fxp
from sklearn import preprocessing
import seaborn as sns
import io
import shutil
import warnings
from save_results import save_simulated_annealing_results, plot_simulated_annealing_results, save_final_state
from collections import Counter

warnings.filterwarnings("ignore", message="torch.utils._pytree._register_pytree_node is deprecated")
# Filtra advertencias de deprecación específicas
warnings.filterwarnings("ignore", category=UserWarning, module='transformers.utils.generic')
warnings.filterwarnings("ignore", category=UserWarning, module='torchvision.models._utils')

# Filtra todas las advertencias de deprecación (no recomendado)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ----------------------------USER DEFINED
simulations = 3  # Nº of simulations. Set >1 if want multiple cuantization results.
Max_steps = 100  # Steps of Simulated Annealing convergence algorithm.

interval = (
    6,
    18,
)  # Search range when simulating the quantification of the fractional part of the parameters.
max_degradation = 5  # Reference based on maximum network accuracy operating in float32 format

# ---------------------------Convergence guidance hyperparameters
# Cost function=gamma*((lower_bound-mAP50)**2) + beta*avg_bits -alpha*lower_bound
alpha = 0.5
beta = 50
gamma = 1
theta = 0.2
# ---------------------------/Convergence guidance hyperparameters
# ----------------------------/USER DEFINED

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
    for param in model.parameters():
        param.requires_grad = True
    load_checkpoint(torch.load(os.path.join(config.DRIVE_PATH,f'{config.BACKBONE}/{config.TOTAL_PATH}/model/{best_model}')), model, optimizer)
    model.eval()
    return model, optimizer

def fractional_bits_required(number, precision=10):
    # Get the fractional part of the number
    fractional_part = number - int(number)
    
    # List to store the binary representation of the fractional part
    binary_fractional_part = []
    
    # Repeat until the fractional part is 0 or we reach the desired precision
    while fractional_part != 0 and len(binary_fractional_part) < precision:
        fractional_part *= 2
        bit = int(fractional_part)
        binary_fractional_part.append(bit)
        fractional_part -= bit
    
    # The number of bits required is the length of the binary_fractional_part list
    return len(binary_fractional_part)


def save_range_bits(layer, min_value, min_max, bits_int_max, max_frac):
    os.makedirs(os.path.join(config.DRIVE_PATH, f"{config.BACKBONE}/{config.TOTAL_PATH}/model_opt"), exist_ok=True)
    with open(os.path.join(config.DRIVE_PATH, f"{config.BACKBONE}/{config.TOTAL_PATH}/model_opt/range_bits.txt"), 'a+') as file:
        # Move the cursor to the start of the file to check if it's empty
        file.write(f"{layer}, Bits: ({bits_int_max}, {max_frac}), Range: [{min_value}, {min_max}]\n")

def counter_fractional_bits():
    torch.cuda.empty_cache()

    model, optimizer = load_model()
    all_frac_bits_max = []


    # if exists remove it
    if os.path.exists(os.path.join(config.DRIVE_PATH, f"{config.BACKBONE}/{config.TOTAL_PATH}/model_opt/range_bits.txt")):
        os.remove(os.path.join(config.DRIVE_PATH, f"{config.BACKBONE}/{config.TOTAL_PATH}/model_opt/range_bits.txt"))

    for name, param in model.named_parameters():

        if 'weight' in name or 'bias' in name:
            max_val = param.max().item()
            
            max_int = int(max_val)
                
            bits_int_max = len(bin(abs(max_int))[2:]) + 1 # bit de signo      

            fractional_part_max = fractional_bits_required(abs(max_val) - abs(max_int), 32)
            
            all_frac_bits_max.append(fractional_part_max)

            max_value = 2**(bits_int_max-1) - 1 + 2**(-fractional_part_max)
            min_value = -2**(bits_int_max-1) + 2**(-fractional_part_max)

            save_range_bits(name, min_value, max_value, bits_int_max, fractional_part_max)


       

if __name__ == '__main__':
    counter_fractional_bits()
    