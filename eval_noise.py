import torch
import torch.optim as optim
from backbone import resnet50, vgg16, efficientnet_b0
from utils import (
    mean_average_precision_noise,
    load_checkpoint,
    get_loaders,
    get_bboxes_noise,
    find_the_best_model
)
from loss import YoloLoss
from tinyissimo_model import tinyissimoYOLO
from ext_tinyissimo_model import ext_tinyissimoYOLO
from YOLOv1 import Yolov1
from bed_model import bedmodel
import config
import os
from save_results import (
    save_results_noise,
    plot_results_noise,
)
from torch import nn
import random
import numpy as np

# seed = 123
# torch.manual_seed(seed)
debug = 0

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

def convert_fractional_to_binary(number, precision=32):
    fractional_part = number - int(number)
    binary_fractional_part = []
    while fractional_part != 0 and len(binary_fractional_part) < precision:
        fractional_part *= 2
        bit = int(fractional_part)
        binary_fractional_part.append(abs(bit))
        fractional_part -= bit
    return binary_fractional_part

def fractional_bits_required(number, precision=10):
  
    # The number of bits required is the length of the binary_fractional_part list
    return len(convert_fractional_to_binary(number, precision=precision))

def convert_fractional_binary_to_fractional(int_value, binary_fractional_part):
    fractional_part = abs(int_value)
    for i in range(len(binary_fractional_part)):
        fractional_part += abs(binary_fractional_part[i]) * 2**(-(i+1))
    return fractional_part

def convert_int_binary_to_int(binary_int_part):
    fractional_part = 0
    for i in range(len(binary_int_part)):
        fractional_part += binary_int_part[i] * 2**(i)
    return fractional_part

# Function to add noise to tensors
def add_noise_to_tensor(tensor, noise_level=0.1):
    return tensor + torch.randn(tensor.size(), device=config.DEVICE) * noise_level


def string_bit(sign_bit, int_bits, frac_bits):
    
    bits = f'{1 if sign_bit==-1 else 0}'
    for int_bit in int_bits:
        bits = f'{bits}{int_bit}'
    bits = f'{bits}.'
    for frac_bit in frac_bits:
        bits = f'{bits}{frac_bit}'
    return bits

# Class to add noise to intermediate activations
class NoisyLayer(nn.Module):
    def __init__(self, percentage_layers_level_1=0.1):
        super(NoisyLayer, self).__init__()
        self.percentage_layers_level_1 = percentage_layers_level_1


    def forward(self, x):
        n_layers_affected = 0
        if torch.rand(1) <= self.percentage_layers_level_1:
            n_layers_affected += 1
            max_val = x.max().item()
            bits_int_max = len(bin(int(max_val))[2:])+1

            exponential = int(torch.empty(1).uniform_(-10, bits_int_max-1).item())
            
            with torch.no_grad():
                if exponential == bits_int_max-1:
                    x_flatten = torch.flatten(x, start_dim = 0)
                    random_index = random.randint(0, x_flatten.numel() - 1) # Choose a random index of the parameters


                    binary_int_part = [int(bit) for bit in bin(abs(int(x_flatten[0])))[2:]]
                    binary_fractional_part = convert_fractional_to_binary(x_flatten[random_index])

                    if debug:
                        string_bits = string_bit(np.sign(x_flatten[random_index].item()), binary_int_part, binary_fractional_part)
                        print(f'\nExponential:  {-exponential} \n\tx: {x_flatten[random_index]}, Bits: {string_bits}')

                    x_flatten[random_index] = -x_flatten[random_index] # Change the sign of the parameter

                    if debug:
                        string_bits = string_bit(np.sign(x_flatten[random_index].item()), binary_int_part, binary_fractional_part)
                        print(f'\tx: {x_flatten[random_index]}, Bits: {string_bits}')

                    x = x_flatten.view(x.size())
                else:
                    if exponential < 0:
                        exponential = abs(exponential)
                        x_flatten = torch.flatten(x, start_dim = 0)
                        random_index = random.randint(0, x_flatten.numel() - 1) # Choose a random index of the parameters
                        binary_fractional_part = convert_fractional_to_binary(x_flatten[random_index])
                        signo = np.sign(x_flatten[random_index].item())

                        binary_int_part = [int(bit) for bit in bin(abs(int(x_flatten[0])))[2:]]
                        binary_int_part.reverse()
                        
                        if debug:
                            string_bits = string_bit(signo, binary_int_part, binary_fractional_part)
                            print(f'\nExponential:  {-exponential} \n\tx: {x_flatten[random_index]}, Bits: {string_bits}')
                        exponential = len(binary_fractional_part)-1 if exponential>=len(binary_fractional_part) else exponential
                        binary_fractional_part[exponential-1] = 0 if binary_fractional_part[exponential-1] else 1

                        x_flatten[random_index] = signo*abs((convert_fractional_binary_to_fractional(abs(int(x_flatten[random_index])), binary_fractional_part)))

                        if debug:
                            string_bits = string_bit(signo, binary_int_part, binary_fractional_part)
                            print(f'\tx: {x_flatten[random_index]}, Bits: {string_bits}')

                    elif exponential >=0:
                        x_flatten = torch.flatten(x, start_dim = 0)
                        random_index = random.randint(0, x_flatten.numel() - 1) # Choose a random index of the parameters

                        binary_fractional_part = convert_fractional_to_binary(x_flatten[random_index])
                        binary_int_part = [int(bit) for bit in bin(abs(int(x_flatten[0])))[2:]]
                        binary_int_part.reverse()
                        signo = np.sign(x_flatten[random_index].item())
                        
                        if debug:
                            string_bits = string_bit(signo, binary_int_part, binary_fractional_part)
                            print(f'\nExponential:  {exponential} \n\tx: {x_flatten[random_index]}, Bits: {string_bits}')

                        exponential = len(binary_int_part)-1 if exponential >= len(binary_int_part) else exponential
                        binary_int_part[exponential] = 0 if binary_int_part[exponential] else 1
                        x_flatten[random_index] = signo*abs(convert_fractional_binary_to_fractional(convert_int_binary_to_int(binary_int_part), binary_fractional_part))
                        
                        if debug:
                            string_bits = string_bit(signo, binary_int_part, binary_fractional_part)
                            print(f'\tx: {x_flatten[random_index]}, Bits: {string_bits}')

                    x = x_flatten.view(x.size())  
        
        if debug:
            print(f'Number of layers affected: {n_layers_affected}')
        return x
    
def test_noise(case1=False, percentage_layers_case_1 = [0], slices_case1 = 0, 
               case2=False, percentage_layers_case_2 = [0], slices_case2 = 0,
               folder_model = None, model_name = None):
    print(f'Test with case1: {case1}, case2: {case2}')
    

    model, optimizer = load_model()

    result_name_file = 'results_noise.txt'
    if case2:
        result_name_file = f'case2_{result_name_file}'
    if case1:
        result_name_file = f'case1_{result_name_file}'

    # Eliminate previous file if exists

    if os.path.exists(os.path.join(config.DRIVE_PATH, f'{config.BACKBONE}/{config.TOTAL_PATH}/results_noise', result_name_file)):
        print(f'File {result_name_file} exists. Deleting...')
        os.remove(os.path.join(config.DRIVE_PATH, f'{config.BACKBONE}/{config.TOTAL_PATH}/results_noise', result_name_file))

    ############################################
    #           Define the loss function       #
    ############################################
    loss_fn = YoloLoss(S=config.SPLIT_SIZE, B=config.NUM_BOXES, C=config.NUM_CLASSES)

    # Load data into data loader
    _, val_loader,_ = get_loaders()

    ############################################
    #       Load the model and the results     #
    ############################################
    if folder_model is None:
        best_model, epoch_best_model = find_the_best_model(os.path.join(config.DRIVE_PATH, f'{config.BACKBONE}/{config.TOTAL_PATH}/model'))
        load_checkpoint(torch.load(os.path.join(config.DRIVE_PATH, f'{config.BACKBONE}/{config.TOTAL_PATH}/model/{best_model}')), model, optimizer)
        print(f"Model loaded: {config.BACKBONE}/{config.TOTAL_PATH}/model/{best_model}")
        folder_save_results = os.path.join(config.DRIVE_PATH, f'{config.BACKBONE}/{config.TOTAL_PATH}/results_noise')
    else:
        load_checkpoint(torch.load(os.path.join(config.DRIVE_PATH, f'{config.BACKBONE}/{config.TOTAL_PATH}/model_opt/{folder_model}/{model_name}')), model, optimizer)
        print(f"Model loaded: {config.BACKBONE}/{config.TOTAL_PATH}/model_opt/{folder_model}/{model_name}")
        folder_save_results = os.path.join(config.DRIVE_PATH, f'{config.BACKBONE}/{config.TOTAL_PATH}/model_opt/{folder_model}/results_noise')

    if case1 and slices_case1[1] == slices_case1[0]:
        range_slices_case1 = range(slices_case1[1])
    else:
        range_slices_case1 = [slices_case1[0]]
    
    if case2 and slices_case2[1] == slices_case2[0]:
        range_slices_case2 = range(slices_case2[1])
    else:
        range_slices_case2 = [slices_case2[0]]

    for percentage_layers_level_1 in percentage_layers_case_1:
        for percentage_layers_level_2 in percentage_layers_case_2:
            for slice_1 in range_slices_case1:
                for slice_2 in range_slices_case2:
                    if case1:
                        print(f'Noise level 1: Slice: [{slice_1}, {slices_case1[1]}], Percentage: {percentage_layers_level_1}')
                    if case2:
                        print(f'Noise level 2: Slice: [{slice_2}, {slices_case2[1]}], Percentage: {percentage_layers_level_2}')

                    '''
                    case1: Add NoisyLayer to the model if case1 flag is True
                    '''
                    if case1:
                        total_modules = sum(1 for _ in model.named_modules())
                        limit_n = total_modules//slices_case1[1]
                        for indx, (name, module) in enumerate(model.named_modules()):
                            if indx >= limit_n*slice_1 and indx < limit_n*(slice_1+1) and name == 'darknet':
                                if isinstance(module, nn.ReLU) or isinstance(module, nn.LeakyReLU):  # Choose appropriate layers
                                    setattr(model, name, nn.Sequential(module, NoisyLayer(percentage_layers_level_1=percentage_layers_level_1)))

                    '''
                    case2: Add noise to model weights if case2 flag is True 
                    '''
                    if case2:
                        total_param = sum(1 for _ in model.parameters())
                        limit_n = total_param//slices_case2[1]
                        n_layers_affected = 0
                        for indx, (name, param) in enumerate(model.named_parameters()):
                            if 'weight' or 'bias' in name:
                                if indx >= limit_n*slice_2 and indx < limit_n*(slice_2+1):


                                    random_value = torch.rand(1)
                                    
                                    if indx == (limit_n*(slice_2+1)-1) and percentage_layers_level_2 != 0:
                                        random_value = 0 if n_layers_affected == 1 else random_value

                                    if random_value <= percentage_layers_level_2:
                                        n_layers_affected += 1
                                        max_val = param.max().item()
                                        bits_int_max = len(bin(int(max_val))[2:])+1 # +1 for the sign bit
                                        
                                        exponential = int(torch.empty(1).uniform_(-10, bits_int_max-1).item())

                                        # exponential = bits_int_max -1 # Test Case  if we want to see how it changes the sign
                                        # exponential = 0 # Test Case if we want to see how it changes the more significant bit
                                        # exponential = -1 #< 0 Test Case if we want to see how it change an specific bit of the fractional part
                                        with torch.no_grad():
                                            if exponential == bits_int_max-1:
                                                param_flatten = torch.flatten(param, start_dim = 0)
                                                random_index = random.randint(0, param_flatten.numel() - 1) # Choose a random index of the parameters


                                                binary_int_part = [int(bit) for bit in bin(abs(int(param_flatten[0])))[2:]]
                                                binary_fractional_part = convert_fractional_to_binary(param_flatten[random_index])

                                                if debug:
                                                    string_bits = string_bit(np.sign(param_flatten[random_index].item()), binary_int_part, binary_fractional_part)
                                                    print(f'Exponential:  {-exponential} \n\tParam: {param_flatten[random_index]}, Bits: {string_bits}')

                                                param_flatten[random_index] = -param_flatten[random_index] # Change the sign of the parameter

                                                if debug:
                                                    string_bits = string_bit(np.sign(param_flatten[random_index].item()), binary_int_part, binary_fractional_part)
                                                    print(f'\tParam: {param_flatten[random_index]}, Bits: {string_bits}')

                                                param = param_flatten.view(param.size())
                                            else:
                                                if exponential < 0:
                                                    exponential = abs(exponential)
                                                    param_flatten = torch.flatten(param, start_dim = 0)
                                                    random_index = random.randint(0, param_flatten.numel() - 1) # Choose a random index of the parameters
                                                    binary_fractional_part = convert_fractional_to_binary(param_flatten[random_index])
                                                    signo = np.sign(param_flatten[random_index].item())

                                                    binary_int_part = [int(bit) for bit in bin(abs(int(param_flatten[0])))[2:]]
                                                    binary_int_part.reverse()
                                                    
                                                    if debug:
                                                        string_bits = string_bit(signo, binary_int_part, binary_fractional_part)
                                                        print(f'Exponential:  {-exponential} \n\tParam: {param_flatten[random_index]}, Bits: {string_bits}')
                                                    exponential = len(binary_fractional_part)-1 if exponential>=len(binary_fractional_part) else exponential
                                                    binary_fractional_part[exponential-1] = 0 if binary_fractional_part[exponential-1] else 1

                                                    param_flatten[random_index] = signo*abs((convert_fractional_binary_to_fractional(abs(int(param_flatten[random_index])), binary_fractional_part)))

                                                    if debug:
                                                        string_bits = string_bit(signo, binary_int_part, binary_fractional_part)
                                                        print(f'\tParam: {param_flatten[random_index]}, Bits: {string_bits}')

                                                elif exponential >=0:
                                                    param_flatten = torch.flatten(param, start_dim = 0)
                                                    random_index = random.randint(0, param_flatten.numel() - 1) # Choose a random index of the parameters

                                                    binary_fractional_part = convert_fractional_to_binary(param_flatten[random_index])
                                                    binary_int_part = [int(bit) for bit in bin(abs(int(param_flatten[0])))[2:]]
                                                    binary_int_part.reverse()
                                                    signo = np.sign(param_flatten[random_index].item())
                                                    
                                                    if debug:
                                                        string_bits = string_bit(signo, binary_int_part, binary_fractional_part)
                                                        print(f'Exponential:  {exponential} \n\tParam: {param_flatten[random_index]}, Bits: {string_bits}')
                                                    exponential = len(binary_int_part)-1 if exponential >= len(binary_int_part) else exponential
                                                    binary_int_part[exponential] = 0 if binary_int_part[exponential] else 1
                                                    param_flatten[random_index] = signo*abs(convert_fractional_binary_to_fractional(convert_int_binary_to_int(binary_int_part), binary_fractional_part))
                                                    
                                                    if debug:
                                                        string_bits = string_bit(signo, binary_int_part, binary_fractional_part)
                                                        print(f'\tParam: {param_flatten[random_index]}, Bits: {string_bits}')

                                                param = param_flatten.view(param.size())

                                            dict(model.named_parameters())[name].data = param
                        if debug:
                            print(f'Number of layers affected: {n_layers_affected}')
                    ############################################
                    #       Evaluating for valid               #
                    ############################################
                    model.eval()
                    
                    pred_boxes, target_boxes, val_loss = get_bboxes_noise(val_loader, model, loss_fn, iou_threshold=0.5, threshold=0.5, device=config.DEVICE, mode='Valid')

                    val_mAP_50, val_mAP_75, val_mAP_90 = mean_average_precision_noise(pred_boxes, target_boxes, iou_thresholds=[0.5, 0.75, 0.9], box_format="midpoint", num_classes=config.NUM_CLASSES, mode='Valid')
                    
                    print(f"Valid: \t mAP@50: {val_mAP_50:.6f}, mAP@75: {val_mAP_75:.6f}, mAP@90: {val_mAP_90:.6f}, Mean Loss: {val_loss:.6f}")
                    save_results_noise( perc1=percentage_layers_level_1, slices1=slice_1, nslices1=slices_case1[1],
                                        perc2=percentage_layers_level_2, slices2=slice_2, nslices2=slices_case2[1],
                                        mAP_50=val_mAP_50, mAP_75=val_mAP_75, mAP_90=val_mAP_90, mean_loss=val_loss, 
                                        file_name=result_name_file, folder_save_results = folder_save_results)
                    plot_results_noise(result_name_file, folder_save_results, case1, case2)

if __name__ == "__main__":
    case1 = False
    case2 = True
    percentage_layers_case_1  = [0.05] if case1 else [0]
    percentage_layers_case_2 = [0.05] if case2 else [0]
    # Con este parámetro seleccionamos la zona en la que queremos afectar a las activacionies o pesos
    # es decir, slices = [slice_i, n_slices], con esto dividimos todas las capas en n_slices y afectamos a las capas que esten en slice_i
    '''
    |slice 0 | slice 1 | slice 2 | slice 3 | slice 4 | 
    '''
    
    slices_case1 = [0,5] 
    slices_case2 = [0,5]
    test_noise(case1=case1, percentage_layers_case_1=percentage_layers_case_1, slices_case1 = slices_case1,
               case2=case2, percentage_layers_case_2=percentage_layers_case_2, slices_case2 = slices_case2)