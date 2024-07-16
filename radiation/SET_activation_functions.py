import torch
import sys
import os
sys.path.append(os.getcwd())
from torch import nn
import random
import numpy as np
import time
import math
from radiation.binary_functions import (convert_fractional_to_binary,
                                    convert_fractional_binary_to_fractional,
                                    convert_int_binary_to_int, string_bit)


def adding_noise_to_activations(tensor, name='', interval_1=(-2, 2), percentage_tensors_1=0.1, debug = 0):

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

def register_hooks(model, slice_1, slices_case1, percentage_layers_level_1, interval_1=(-12, 2), percentage_tensors_1=0.1, debug = 0):
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
                                                            percentage_tensors_1=percentage_tensors_1,
                                                            debug = debug)
                    return hook
                layer.register_forward_hook(create_hook(name))
    