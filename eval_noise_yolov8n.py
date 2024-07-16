import torch
import torch.optim as optim
from utils import (
    mean_average_precision_noise,
    get_loaders,
    get_bboxes_yolov8,
)
from loss import YoloLoss
import config
import os
from save_results import (
    save_results_noise,
    plot_results_noise,
)
from torch import nn
import random
import numpy as np
import time
import math
from ultralytics import YOLO



debug = 0
global total_layers_affected_1
total_layers_affected_1 = 0
global total_layers_affected_2
total_layers_affected_2 = 0
global max_limit_1
max_limit_1 = []
global max_limit_2
max_limit_2 = []

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




def string_bit(sign_bit, int_bits, frac_bits):
    
    bits = f'{1 if sign_bit==-1 else 0}'
    for int_bit in int_bits:
        bits = f'{bits}{int_bit}'
    bits = f'{bits}.'
    for frac_bit in frac_bits:
        bits = f'{bits}{frac_bit}'
    return bits

def adding_noise_to_activations(tensor, name='', interval_1=(-2, 2), percentage_tensors_1=0.1):

    if debug:
        print('############################################')
        print(f'Name: {name}')
        print(f'Output shape: {tensor.shape}, Output flatten shape: {torch.flatten(tensor, start_dim=0).shape}')
    max_val = tensor.max().item()
    bits_int_max = interval_1[1] if interval_1[1] <= len(bin(int(max_val))[2:])+1 else len(bin(int(max_val))[2:])+1

    with torch.no_grad():
        x_flatten = torch.flatten(tensor, start_dim=0).clone()  # Crear una copia para evitar problemas de vista
        random_index_array = random.sample(range(0, x_flatten.numel() - 1), math.ceil(x_flatten.numel()*percentage_tensors_1))  # Elegir índices aleatorios
        # print(f'random_index_array: {len(random_index_array)}')
        # mitad_x_flatten = int(x_flatten.numel()//2)
        # range_sample = range(mitad_x_flatten, mitad_x_flatten + int(x_flatten.numel()*percentage_tensors_1))
        # # range_sample = range(0, int(x_flatten.numel()*percentage_tensors_1))
        # print(f'x_flatten: {x_flatten.numel()}, Percentage: {percentage_tensors_1}, Range: {len(range_sample)}')
        # random_index_array = range_sample #random.sample(range_sample, len(range_sample))
                             
        # random_index_array = random.sample(range(0, int(x_flatten.numel()/3)), math.ceil(x_flatten.numel()*percentage_tensors_1))  # Elegir índices aleatorios
        
        t1 = time.time()
        if debug:                
            print(f'Random index len: {len(random_index_array)}')
        for random_index in random_index_array:
            exponential = int(torch.empty(1).uniform_(interval_1[0], bits_int_max).item())
            if exponential >= bits_int_max-1:
                binary_fractional_part = convert_fractional_to_binary(x_flatten[random_index])
                binary_int_part = [int(bit) for bit in bin(abs(int(x_flatten[random_index])))[2:]]
                
                if debug:
                    string_bits = string_bit(np.sign(x_flatten[random_index].item()), binary_int_part, binary_fractional_part)
                    print(f'\nRandom index: {random_index} Exponential:  {exponential} \n\tx: {x_flatten[random_index]}, Bits: {string_bits}')

                if len(binary_fractional_part) != 0:
                    x_flatten[random_index] = -x_flatten[random_index]  # Cambiar el signo del parámetro

                if debug:
                    string_bits = string_bit(np.sign(x_flatten[random_index].item()), binary_int_part, binary_fractional_part)
                    print(f'\tx: {x_flatten[random_index]}, Bits: {string_bits}')

            else:
                if exponential < 0:
                    exponential = abs(exponential)
                    binary_fractional_part = convert_fractional_to_binary(x_flatten[random_index])
                    signo = np.sign(x_flatten[random_index].item())

                    binary_int_part = [int(bit) for bit in bin(abs(int(x_flatten[random_index])))[2:]]
                    binary_int_part.reverse()
                    
                    if debug:
                        string_bits = string_bit(signo, binary_int_part, binary_fractional_part)
                        print(f'\nRandom index: {random_index} Exponential:  {-exponential} \n\tx: {x_flatten[random_index]}, Bits: {string_bits}')
                    exponential = len(binary_fractional_part)-1 if exponential >= len(binary_fractional_part) else exponential
                    
                    if len(binary_fractional_part) != 0:
                        binary_fractional_part[exponential-1] = 0 if binary_fractional_part[exponential-1] else 1

                    x_flatten[random_index] = signo * abs((convert_fractional_binary_to_fractional(abs(int(x_flatten[random_index])), binary_fractional_part)))

                    if debug:
                        string_bits = string_bit(signo, binary_int_part, binary_fractional_part)
                        print(f'\tx: {x_flatten[random_index]}, Bits: {string_bits}')

                elif exponential >= 0:
                    binary_fractional_part = convert_fractional_to_binary(x_flatten[random_index])
                    binary_int_part = [int(bit) for bit in bin(abs(int(x_flatten[random_index])))[2:]]
                    binary_int_part.reverse()
                    signo = np.sign(x_flatten[random_index].item())
                    
                    if debug:
                        string_bits = string_bit(signo, binary_int_part, binary_fractional_part)
                        print(f'\nRandom index: {random_index} Exponential:  {exponential} \n\tx: {x_flatten[random_index]}, Bits: {string_bits}')

                    exponential = len(binary_int_part) - 1 if exponential >= len(binary_int_part) else exponential
                    if len(binary_int_part) != 0:
                        binary_int_part[exponential] = 0 if binary_int_part[exponential] else 1
                    x_flatten[random_index] = signo * abs(convert_fractional_binary_to_fractional(convert_int_binary_to_int(binary_int_part), binary_fractional_part))
                    
                    if debug:
                        string_bits = string_bit(signo, binary_int_part, binary_fractional_part)
                        print(f'\tx: {x_flatten[random_index]}, Bits: {string_bits}')
        t2 = time.time()
        if debug:
            print(f'Time: {t2-t1}')
            print(tensor.shape)
            print(x_flatten.view(tensor.size()).shape)
            print(torch.equal(tensor,x_flatten.view(tensor.size())))    
        return x_flatten.view(tensor.size())  # Return the modified tensor

def register_hooks(model, slice_1, slices_case1, percentage_layers_level_1, interval_1=(-12, 2), percentage_tensors_1=0.1):
    global total_layers_affected_1
    global max_limit_1
    max_limit_1 = []

    layer_array = [name for name, layer in model.named_modules() if isinstance(layer, (nn.ReLU, nn.LeakyReLU))]
    total_modules = len(layer_array)
    limit_n = math.ceil(total_modules//slices_case1[1])

    max_limit = limit_n*(slice_1+1) if limit_n*(slice_1+1) < total_modules else total_modules
    n_samples = math.ceil((max_limit - limit_n*slice_1)*percentage_layers_level_1)
    layers_random = random.sample(layer_array[limit_n*slice_1:max_limit], n_samples)
    
    for sl1 in range(slices_case1[1]):
        max_limit = limit_n*(sl1+1) - limit_n*sl1 +1
        max_limit = total_modules  - max_limit*sl1 if max_limit*(sl1+1) >= total_modules else max_limit
        max_limit_1.append(max_limit)

    total_layers_affected_1 = math.ceil(max_limit_1[slice_1]*percentage_layers_level_1)
    if debug:
        print(f'Number of layers affected: {len(layers_random)}')
    for indx, (name, layer) in enumerate(model.named_modules()):
        if isinstance(layer, (nn.ReLU, nn.LeakyReLU)):
            if name in layers_random:
                def create_hook(name):
                    def hook(module, input, output):
                        return adding_noise_to_activations(output, name=name, 
                                                            interval_1=interval_1,
                                                            percentage_tensors_1=percentage_tensors_1)
                    return hook
                layer.register_forward_hook(create_hook(name))
    

    
            


def test_noise(case1=False, percentage_layers_case_1 = [0], slices_case1 = 0, interval_1 = (-12, 2), percentage_tensors_1=0.1,
               case2=False, percentage_layers_case_2 = [0], slices_case2 = 0, interval_2 = (-12, 2), percentage_tensors_2=0.1,
               folder_model = None, model_name = None):
    print(f'Test with case1: {case1}, case2: {case2}')
    global total_layers_affected_1
    global total_layers_affected_2
    global max_limit_1
    global max_limit_2
    
    ############################################
    #           Define the loss function       #
    ############################################
    loss_fn = YoloLoss(S=config.SPLIT_SIZE, B=config.NUM_BOXES, C=config.NUM_CLASSES)

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
                    model  = load_model(folder_model=folder_model, model_name=model_name)
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
                                       percentage_tensors_1=percentage_tensors_1)
                        

                    '''
                    case2: Add noise to model weights if case2 flag is True 
                    '''
                    if case2:
                        val_mAP_50_array = []

                        for _ in range(10):
                            model = load_model(folder_model=folder_model, model_name=model_name)

                            # Layer_array contains the name of the layers that are weights or bias
                            layer_array = [name for name, layer in model.named_parameters() if 'weight' or 'bias' in name]
                            total_param = len(layer_array)

                            # limit_n is the number of layers that we will affect in each slice
                            limit_n = math.ceil(total_param//slices_case2[1])

                            # max_limit is the number of layers that we belongs in the corresponding slice
                            max_limit = limit_n*(slice_2+1) if limit_n*(slice_2+1) < total_param else total_param

                            # n_samples is the number of layers that we will affect in the corresponding slice
                            n_samples = math.ceil((max_limit - limit_n*slice_2)*percentage_layers_level_2)

                            # layers_random is the name of the layers that we will affect in the corresponding slice
                            layers_random = random.sample(layer_array[limit_n*slice_2:max_limit], n_samples)
                            max_limit_2 = []
                            for sl2 in range(slices_case2[1]):
                                max_limit = limit_n*(sl2+1) - limit_n*sl2 +1
                                max_limit = total_param  - max_limit*sl2 if max_limit*(sl2+1) >= total_param else max_limit
                                max_limit_2.append(max_limit)
                            
                            total_layers_affected_2 = math.ceil(max_limit_2[slice_2]*percentage_layers_level_2)

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
                                    bits_int_max = interval_2[1] if interval_2[1] <= len(bin(int(max_val))[2:])+1 else len(bin(int(max_val))[2:])+1 # +1 for the sign bit


                                    # exponential = bits_int_max -1 # Test Case  if we want to see how it changes the sign
                                    # exponential = 0 # Test Case if we want to see how it changes the more significant bit
                                    # exponential = -1 #< 0 Test Case if we want to see how it change an specific bit of the fractional part
                                    with torch.no_grad():
                                        # exponential is a random number between the interval_2[0] and bits_int_max
                                        exponential = int(torch.empty(1).uniform_(interval_2[0], bits_int_max).item())
                                        # exponential = -1
                                        param_flatten = torch.flatten(param, start_dim = 0)
                                        
                                        # As param_flatten is a view of the original tensor, we can modify a random index of the flatten tensor
                                        param_flatten_numel = param_flatten.numel()
                                        if param_flatten_numel >1:
                                            random_index_array = random.sample(range(0, param_flatten_numel - 1), math.ceil(param_flatten_numel*percentage_tensors_2)) # Choose a random index of the parameters
                                        else:
                                            random_index_array = [0]    
                                        # mitad_numel = int(param_flatten.numel()//2)
                                        # random_index_array = range(mitad_numel, mitad_numel + int(param_flatten.numel()*percentage_tensors_2))
                                        # random_index_array = [0]
                                        if debug:
                                            print(f'Random index: {len(random_index_array)}')
                                        for random_index in random_index_array:
                                            if exponential == bits_int_max-1:
                                                binary_int_part = [int(bit) for bit in bin(abs(int(param_flatten[0])))[2:]]
                                                binary_fractional_part = convert_fractional_to_binary(param_flatten[random_index])

                                                if debug:
                                                    string_bits = string_bit(np.sign(param_flatten[random_index].item()), binary_int_part, binary_fractional_part)
                                                    print(f'Random index: {random_index} Exponential:  {exponential} \n\tParam: {param_flatten[random_index]}, Bits: {string_bits}')

                                                if len(binary_fractional_part) != 0:
                                                    param_flatten[random_index] = -param_flatten[random_index] # Change the sign of the parameter

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
                                                    exponential = len(binary_fractional_part)-1 if exponential>=len(binary_fractional_part) else exponential
                                                    
                                                    if len(binary_fractional_part) != 0:
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
                                                        print(f'Random index: {random_index} Exponential:  {exponential} \n\tParam: {param_flatten[random_index]}, Bits: {string_bits}')
                                                    exponential = len(binary_int_part)-1 if exponential >= len(binary_int_part) else exponential
                                                    
                                                    if len(binary_int_part) != 0:
                                                        binary_int_part[exponential] = 0 if binary_int_part[exponential] else 1
                                                    param_flatten[random_index] = signo*abs(convert_fractional_binary_to_fractional(convert_int_binary_to_int(binary_int_part), binary_fractional_part))
                                                    
                                                    if debug:
                                                        string_bits = string_bit(signo, binary_int_part, binary_fractional_part)
                                                        print(f'\tParam: {param_flatten[random_index]}, Bits: {string_bits}')

                                                param = param_flatten.view(param.size())

                                                dict(model.named_parameters())[name].data = param
                                                if debug:
                                                    print(f'Number of weights affected: {n_weights_affected}')
                                ############################################
                                #       Evaluating for valid               #
                                ############################################
                            
                                
                            pred_boxes, target_boxes = get_bboxes_yolov8(val_loader, model, iou_threshold=0.5, threshold=0.5, mode = 'eval_noise')

                            val_mAP_50, val_mAP_75, val_mAP_90 = mean_average_precision_noise(pred_boxes, target_boxes, iou_thresholds=[0.5, 0.75, 0.9], box_format="midpoint", num_classes=config.NUM_CLASSES, mode='Valid')
                            print(f"Valid: \t mAP@50: {val_mAP_50:.6f}, mAP@75: {val_mAP_75:.6f}, mAP@90: {val_mAP_90:.6f}")
                            val_mAP_50_array.append(val_mAP_50)
                            # print(f'array_param_shape: {array_param_shape}')
                        val_mAP_50 = np.mean(val_mAP_50_array)

                    if case1:
                        val_mAP_50_array = []
                        loss_val_array = []
                        for _ in range(1):
                            pred_boxes, target_boxes = get_bboxes_yolov8(val_loader, model, iou_threshold=0.5, threshold=0.5, mode = 'eval_noise')
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

