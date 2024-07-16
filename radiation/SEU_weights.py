import torch
import sys
import os
sys.path.append(os.getcwd())
from torch import nn
import random
import numpy as np
import time
import math
from utils.load_save_model import load_model
from radiation.binary_functions import (convert_fractional_to_binary,
                                    convert_fractional_binary_to_fractional,
                                    convert_int_binary_to_int, string_bit)
                                    



def SEU_simmulation(model, slices_case2, slice_2=0, percentage_layers_level_2=0, interval_2=(-2, 2), percentage_tensors_2 = 0.1, debug=0):
    

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
    

    if debug:
        print(f'Number of layers affected: {len(layers_random)}')

    n_weights_affected = 0
    # array_param_shape = 0
    for indx, (name, param) in enumerate(model.named_parameters()):
        if name in layers_random:
            
            if debug:
                print(f'Name: {name}, Shape: {param.shape}, Flattened: {param.flatten().shape}')
            # array_param_shape += param.flatten().shape[0]
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
                random_index_array = random.sample(range(0, param_flatten.numel() - 1), math.ceil(param_flatten.numel()*percentage_tensors_2)) # Choose a random index of the parameters
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

    return model