import torch
import pickle
import os
import random
import math
import time
import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt
from fxpmath import Fxp
from sklearn import preprocessing
# import seaborn as sns
import shutil
import warnings
import sys
sys.path.append(os.getcwd())
import config

from utils.utils import (
    get_loaders,
    get_bboxes,
    mean_average_precision,
)

from utils.load_save_model import load_model, save_checkpoint
from utils.binary_functions import fractional_bits_required
from utils.save_results import (
    save_simulated_annealing_results, 
    plot_simulated_annealing_results, 
    save_acceptance_prob,
    save_final_state,  
    save_history_state
    )
utils_YOLO_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../YOLO_functions'))
if utils_YOLO_path not in sys.path:
    sys.path.append(utils_YOLO_path)
from YOLO_functions.loss import YoloLoss


warnings.filterwarnings("ignore", message="torch.utils._pytree._register_pytree_node is deprecated")
# Filtra advertencias de deprecación específicas
warnings.filterwarnings("ignore", category=UserWarning, module='transformers.utils.generic')
warnings.filterwarnings("ignore", category=UserWarning, module='torchvision.models._utils')

# Filtra todas las advertencias de deprecación (no recomendado)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ----------------------------USER DEFINED
simulations = 5  # Nº of simulations. Set >1 if want multiple cuantization results.
Max_steps = 100  # Steps of Simulated Annealing convergence algorithm.

global interval
interval = (
    4,
    10,
)  # Search range when simulating the quantification of the fractional part of the parameters.
max_degradation = 5  # Reference based on maximum network accuracy operating in float32 format
n_slices = 3 # Number of slices to divide the pipeline model

# ---------------------------Convergence guidance hyperparameters
# Cost function=gamma*((lower_bound-mAP50)**2) + beta*avg_bits -alpha*lower_bound
alpha = 0.1  # penalization for lower_bound
beta = 20     # penalization for avg_bits
gamma = 50   # penalization for the difference btwd lower_bound y mAP50

# ---------------------------/Convergence guidance hyperparameters
# ----------------------------/USER DEFINED






final_mAP50_sim = []
final_avg_sim = []
final_MB_sim = []
subfolder = f'{simulations}_sims_max_steps_{Max_steps}_slices_{n_slices}_interval_{interval[0]}_{interval[1]}_degredation_{max_degradation}'
# If exists remove it and then create the folder
if os.path.exists(os.path.join(config.ROOT_DIR,f'{config.BACKBONE}/{config.TOTAL_PATH}/model_opt/{subfolder}')):
    shutil.rmtree(os.path.join(config.ROOT_DIR,f'{config.BACKBONE}/{config.TOTAL_PATH}/model_opt/{subfolder}'))
os.makedirs(os.path.join(config.ROOT_DIR,f'{config.BACKBONE}/{config.TOTAL_PATH}/model_opt/{subfolder}'), exist_ok=True)
loss_fn = YoloLoss(S=config.SPLIT_SIZE, B=config.NUM_BOXES, C=config.NUM_CLASSES)

for n_iter in range(simulations):
    print('##############################################################')
    print(f'              Simulation {n_iter+1} of {simulations}         ')
    print('##############################################################')
    # -------GLOBAL VARS
    last_indices = [-1]
    new_cost = -1
    weights_cost = []
    time_stamp = time.time()
    # -------------------------------Model_data_parameters. MODIFY WITH THE TOPOLOGY OF INTEREST
    num_classes = config.NUM_CLASSES
    input_shape = (config.IMAGE_SIZE, config.IMAGE_SIZE, 3)

    torch.cuda.empty_cache()

    model, optimizer = load_model()
    model_orig, optimizer_orig = load_model()


    # -------------------------------/Model_data_parameters. MODIFY WITH THE TOPOLOGY OF INTEREST
    # weight_dict = {name: param for name, param in model.named_parameters() if 'weight' or 'bias' in name}
    weight_dict = {}
    total_param = sum(1 for _ in model.parameters())

    for param_indx, (name, param) in enumerate(model_orig.named_parameters()):
        if param.requires_grad:
            if name.endswith('weight') or name.endswith('bias'):
                weight_dict[name] = param
                
                

    # sns.set(context="talk", style="darkgrid", palette="hls", font="sans-serif", font_scale=1.05)
    # FIGSIZE = (19, 8)  #: Figure size, in inches!
    # mpl.rcParams['figure.figsize'] = FIGSIZE

    factor=10 #Neccesary if the cost function involves the square of the difference
        
    # Load data into data loader
    _, val_loader, _ = get_loaders()
    model.eval()
    pred_boxes, target_boxes, val_loss = get_bboxes(0, val_loader, model, loss_fn, iou_threshold=0.5, threshold=0.5, device=config.DEVICE, mode='Annealing')
    val_mAP_50_ini, val_mAP_75, val_mAP_90 = mean_average_precision(pred_boxes, target_boxes, iou_thresholds=[0.5, 0.75, 0.9], box_format="midpoint", num_classes=config.NUM_CLASSES)

    lower_bound=(val_mAP_50_ini-(max_degradation/100))*factor

    layers_of_interest=[]
    n_int_of_layers_of_interest=[]
    n_fractional_part_of_layers_of_interest = []
    size_mb_initial = 0 # Initial size of the model in MB
    for param_indx, (name, param) in enumerate(model_orig.named_parameters()):
        # Verificar si el módulo tiene parámetros entrenables
        if param.requires_grad:
            if name.endswith('weight') or name.endswith('bias'):
                layers_of_interest.append(name)
                max_val = param.max().item()
                max_int = int(abs(max_val))
                bits_max = len(bin(abs(max_int))[2:])+1
                n_int_of_layers_of_interest.append(bits_max)
                frac_bits = fractional_bits_required(max_val, precision=32)
                n_fractional_part_of_layers_of_interest.append(frac_bits)
                size_mb_initial += param.numel() * (32 / 8)/(1024**2)    

    
    # avg_bits_ini = sum(n_fractional_part_of_layers_of_interest) / len(n_fractional_part_of_layers_of_interest)
    avg_bits_ini = 32 # Initial average bits, Pytorch default precision 
    print(f"Initial mAP50: {val_mAP_50_ini:.3f}, Initial avg bits: {32:.3f}, Initial size: {size_mb_initial:.3f} MB")
    print(f"Initial fractional part: {n_fractional_part_of_layers_of_interest}")
    
    for indx_slice in range(0, n_slices): 
        print(f'################################################')
        print(f'       Slice  {indx_slice+1} of {n_slices}      ')
        print(f'################################################')
        def f(x,alpha,beta,gamma, temp=10, step = 0):
            """ Function to minimize."""
            fxp_by_layer=[]
            size_mb = 0

            for i in range(len(x)):
                fxp_by_layer.append(Fxp(None, signed=True, n_int=n_int_of_layers_of_interest[i], n_frac=x[i]))

            for i, layer_name in enumerate(layers_of_interest):
                if layer_name in weight_dict:
                    # Obtenemos el tensor de pesos
                    param = weight_dict[layer_name]             

                    # Convertimos el tensor a numpy array
                    weight_array = param.detach().cpu().numpy()
                    
                    # Cuantizamos el array de numpy usando Fxp
                    quantized_weight = Fxp(weight_array, like=fxp_by_layer[i]).astype(float)
                    
                    # Convertimos de nuevo a tensor de PyTorch y copiamos los datos al nuevo modelo
                    quantized_weight_tensor = torch.tensor(quantized_weight, dtype=param.dtype).to(param.device)
                    dict(model.named_parameters())[layer_name].data.copy_(quantized_weight_tensor)
                    size_mb += quantized_weight_tensor.numel() * (x[i] / 8)/(1024**2)

            model.eval()
            pred_boxes, target_boxes, val_loss = get_bboxes(0, val_loader, model, loss_fn, iou_threshold=0.5, threshold=0.5, device=config.DEVICE, mode='Annealing')
            val_mAP_50, _, _ = mean_average_precision(pred_boxes, target_boxes, iou_thresholds=[0.5, 0.75, 0.9], box_format="midpoint", num_classes=config.NUM_CLASSES)
            val_mAP_50_factor = val_mAP_50*factor
            
            a,b=interval
            # Para evitar penalizar el número de bits de la parte entera
            length_x_array = int((indx_slice+1)*total_param/n_slices) - int(indx_slice*total_param/n_slices)
            x_array=np.array([sum(x[int((indx_slice*total_param/n_slices)):int((indx_slice+1)*total_param/n_slices)])/length_x_array,max(a,b)])
            avg_bits=preprocessing.normalize([x_array])[0][0]
            cost= gamma*((lower_bound-val_mAP_50_factor)**2) + beta*avg_bits -alpha*lower_bound
            print(f'Cost: {cost:.3f} | Lower bound: {lower_bound:.3f} | mAP50_factor: {val_mAP_50_factor:.3f} | Avg bits: {avg_bits:.3f} | Val loss: {val_loss:.3f}')
            

            weights_cost.append([gamma*((lower_bound-val_mAP_50_factor)**2)/cost,beta*avg_bits/cost,alpha*lower_bound/cost])
            mean_bits = sum(x) / len(x)
            print(f"Cost: {cost:.3f} | Avg bits: {mean_bits:.3f} | mAP50: {val_mAP_50:.3f} | Size: {size_mb:.4f} MB")

            save_simulated_annealing_results(step, mean_bits, val_mAP_50, val_loss, cost, temp, subfolder = f'{subfolder}/sim_{n_iter+1}')
            # If exists remove it and then create the file
            if os.path.exists(os.path.join(config.ROOT_DIR,f'{config.BACKBONE}/{config.TOTAL_PATH}/model_opt/{subfolder}/sim_{n_iter+1}/final_precision_weights.txt')):
                os.remove(os.path.join(config.ROOT_DIR,f'{config.BACKBONE}/{config.TOTAL_PATH}/model_opt/{subfolder}/sim_{n_iter+1}/final_precision_weights.txt'))
            
            save_final_state(layers_of_interest, n_int_of_layers_of_interest,  x, f'{subfolder}/sim_{n_iter+1}')

            return cost, val_mAP_50, size_mb



        def clip(x,i):#OK
            """ Force x to be in the interval."""
            a, b = interval
            x[i]=int(max(min(x[i], b), a))
            return x

        def random_start():#OK
            """ Random point in the interval."""
            a, b = interval
            start=[]
            for i in range(0,len(layers_of_interest)):
                if layers_of_interest[i] in weight_dict:
                    if indx_slice*total_param/n_slices <= i < (indx_slice+1)*total_param/n_slices:
                        start.append(int(round(a + (b - a) * rn.random_sample())))
                    else:
                        start.append(n_fractional_part_of_layers_of_interest[i])
            return start

        def cost_function(x,alpha,beta,gamma, temp=0, step = 0):
            """ Cost of x = f(x)."""
            return f(x,alpha,beta,gamma, temp = temp, step = step)

        def random_neighbour(x, T,cost,new_cost):
            """Move a little bit x, from the left or the right."""
            amplitude = int(math.ceil((max(interval) - min(interval))* 0.5 * T))
            indices = []
            if cost == new_cost or new_cost > cost:
                n_params = int((indx_slice + 1) * total_param / n_slices) - int(indx_slice * total_param / n_slices)
                
                # Ensure num_params_to_change is within a valid range
                num_params_to_change = max(1, min(math.ceil(n_params * (T / 10.0)), n_params))   
                
                if indx_slice != n_slices - 1:
                    range_params = range(math.ceil(indx_slice*total_param/n_slices), math.ceil((indx_slice+1)*total_param/n_slices))
                    num_params_to_change =  min(num_params_to_change, len(range_params))
                    indices = random.sample(range_params, num_params_to_change)
                else:
                    range_params = range(math.ceil(indx_slice*total_param/n_slices), total_param)
                    num_params_to_change =  min(num_params_to_change, len(range_params))
                    indices = random.sample(range_params, num_params_to_change)
            else:
                num_params_to_change = 0
                indices = last_indices # Utiliza el último índice perturbado

            
            for i in indices:
                if not i == -1:
                    delta = amplitude * random.randrange(-1, 2, 2)
                    if not x[i] == 0:
                        x[i] = x[i] + delta
                    x = clip(x, i)

            print(f'Num params to change: {num_params_to_change}')
            print(f'Indices: {indices}')
                
            return x, indices, num_params_to_change
        
        def acceptance_probability(cost, new_cost, temperature):
            if new_cost < cost:
                print("    - Acceptance probabilty = 1 as new_cost = {} < cost = {}...".format(new_cost, cost))
                return 1
            else:
                p = np.exp(- (new_cost - cost) / temperature)
                print("    - Acceptance probabilty = {:.3g}...".format(p))
                return p
                
        # def temperature(fraction):
        #     """ Example of temperature dicreasing as the process goes on."""
        #     return max(0.01, min(1, 1 - fraction))

        def temperature(fraction):
            """Modifica la temperatura siguiendo una función exponencial inversa"""
            T_initial = 10  # Temperatura inicial
            T_min = 0.01  # Temperatura mínima
            return T_initial * np.exp(-fraction * 5) + T_min
        
        def annealing(random_start,
                    cost_function,
                    random_neighbour,
                    acceptance,
                    temperature,
                    maxsteps=100,
                    debug=True,
                    alpha=1,
                    beta=1,
                    gamma=1):
            """ Optimize the black-box function 'cost_function' with the simulated annealing algorithm."""
            state = random_start()
            print(f'State: {state}')
            cost, val_mAP, size_mb = cost_function(state,alpha,beta,gamma, step = indx_slice*maxsteps)
            costs = [cost]
            states=[state[:]]

            for step in range(indx_slice*maxsteps, (indx_slice+1)*maxsteps):
                print(f'################### Step: {step+1} ###################')                
                torch.cuda.empty_cache()
                # fraction = step / float(maxsteps)
                if indx_slice == 0:
                    fraction = step / float(maxsteps)
                else:
                    fraction = (step - indx_slice*maxsteps) / float(maxsteps)
                T = temperature(fraction)
                global last_indices
                global new_cost
                global interval
                global n_fractional_part_of_layers_of_interest
                new_state, indices, num_params_to_change = random_neighbour(state[:], T,cost,new_cost)
                save_history_state(new_state, f'{subfolder}/sim_{n_iter+1}')
                new_cost, new_val_mAP, size_mb = cost_function(new_state,alpha,beta,gamma, temp = T, step = step)

                states.append(state[:])
                costs.append(cost)

                mean_state = sum(state)/len(state)
                mean_new_state = sum(new_state)/len(new_state)
                # print(f'Step: {step}, State: {state}')
                if debug: print(f"Sim: [{n_iter+1}/{simulations}], Step [{step+1}/{(indx_slice+1)*maxsteps}] : T = {T:.3f}, mean bits state = {mean_state:.3f}, cost = {cost:.3f}, mean bits new state = {mean_new_state:.3f}, new_cost = {new_cost:.3f} ...")
                
                if acceptance_probability(cost, new_cost, T) > rn.random():
                    state, cost = new_state, new_cost
                    val_mAP = new_val_mAP
                    last_indices = indices # Guarda uno de los índices perturbados
                    save_acceptance_prob(step, new_val_mAP, 1, num_params_to_change, f'{subfolder}/sim_{n_iter+1}')
                    print("  ==> Accept it!")
                    print(f'Step: {step+1}, New State: {state}')
                else:
                    save_acceptance_prob(step, new_val_mAP, 0, num_params_to_change, f'{subfolder}/sim_{n_iter+1}')
                    print("  ==> Reject it...")
                    print(f'Step: {step+1}, New State: {state}')
                
                plot_simulated_annealing_results(gamma, beta, alpha, f'{subfolder}/sim_{n_iter+1}', val_mAP_50_ini, avg_bits_ini, size_mb_initial, size_mb, lower_bound/factor)
            n_fractional_part_of_layers_of_interest = state
            return state, [cost, val_mAP, size_mb], states, costs


        state, cost_result, states, costs = annealing(random_start, cost_function, random_neighbour, acceptance_probability, temperature, maxsteps=Max_steps, debug=True, alpha=alpha, beta=beta, gamma=gamma)
        c = cost_result[0]
        val_mAP = cost_result[1]
        final_size = cost_result[2]
        states.append(state[:])
        states = states[2:]
        costs.append(c)
        costs = costs[2:]
    # ------------------
    # -----COMPUTE ACCURACY AFTER SIMULATED ANNEALING
    fxp_by_layer = []
    for i in range(len(state)):
        fxp_by_layer.append(Fxp(None, signed=True, n_int=n_int_of_layers_of_interest[i], n_frac=state[i]))

    for i, layer_name in enumerate(layers_of_interest):
        if layer_name.endswith('bias'):
            param = dict(model.named_parameters())[layer_name]
            quantized_bias = Fxp(param.data.cpu().numpy(), like=fxp_by_layer[i]).astype(float)
            param.data.copy_(torch.tensor(quantized_bias).to(param.device))

        if layer_name.endswith('weight'):
            param = dict(model.named_parameters())[layer_name]
            quantized_weight = Fxp(param.data.cpu().numpy(), like=fxp_by_layer[i]).astype(float)
            param.data.copy_(torch.tensor(quantized_weight).to(param.device))

    model.eval()
    pred_boxes, target_boxes, val_loss = get_bboxes(0, val_loader, model, loss_fn, iou_threshold=0.5, threshold=0.5, device=config.DEVICE, mode='Annealing')
    val_mAP_50, val_mAP_75, val_mAP_90 = mean_average_precision(pred_boxes, target_boxes, iou_thresholds=[0.5, 0.75, 0.9], box_format="midpoint", num_classes=config.NUM_CLASSES)
    
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }    
    save_checkpoint(checkpoint, filename=os.path.join(config.ROOT_DIR, f"{config.BACKBONE}/{config.TOTAL_PATH}/model_opt/{subfolder}/sim_{n_iter+1}/YOLO_opt.pth.tar"))    

    # If exists remove it and then create the file
    if os.path.exists(os.path.join(config.ROOT_DIR,f'{config.BACKBONE}/{config.TOTAL_PATH}/model_opt/{subfolder}/sim_{n_iter+1}/final_precision_weights.txt')):
        os.remove(os.path.join(config.ROOT_DIR,f'{config.BACKBONE}/{config.TOTAL_PATH}/model_opt/{subfolder}/sim_{n_iter+1}/final_precision_weights.txt'))    
    save_final_state(layers_of_interest, n_int_of_layers_of_interest, state, f'{subfolder}/sim_{n_iter+1}')
    save_history_state(state, f'{subfolder}/sim_{n_iter+1}')
    print(f"Original size: {size_mb_initial:.3f} MB | Final size: {final_size:.3f} MB")

    def see_annealing(states, costs):
        plt.figure(figsize=(19, 8))
        plt.suptitle("Evolution of states and costs of the simulated annealing")
        plt.subplot(121)
        plt.plot(np.mean(states,1), 'r')
        plt.title(f"States      Final state Avg bits: {sum(state)/len(state):.3f}")
        plt.xlabel('Step')
        plt.ylabel('Avg nº of bits')
        plt.subplot(122)
        plt.plot(costs, 'b', label=f"\u03B1:{alpha:.2f}, \u03B2:{beta:.2f}, \u03B3:{gamma:.2f}\nLower bound: {lower_bound/factor:.3f}\nFinal mAP50: {val_mAP_50:.3f}\nError: {((lower_bound/factor)-val_mAP_50):.3f} ")
        plt.title(f"Costs           Final cost: {c:.3f}")
        plt.xlabel('Step')
        plt.ylabel('Cost')
        plt.legend()
        plt.savefig(os.path.join(config.ROOT_DIR,f'{config.BACKBONE}/{config.TOTAL_PATH}/model_opt/{subfolder}/sim_{n_iter+1}/sim_{Max_steps}_steps_{time_stamp}.pdf'), format="pdf", bbox_inches="tight")
        plt.close()

    def see_weights_cost(weights_cost):
        alpha_cost = [item[2] for item in weights_cost]
        beta_cost = [item[1] for item in weights_cost]
        gamma_cost = [item[0] for item in weights_cost]

        plt.figure(figsize=(19, 8))
        plt.plot(alpha_cost, label="\u03B2*lower_bound")
        plt.plot(beta_cost, label="\u03B1*avg_bits")
        plt.plot(gamma_cost, label="\u03B3*(lower_bound-actual_mAP50)")
        plt.legend()
        plt.savefig(os.path.join(config.ROOT_DIR,f'{config.BACKBONE}/{config.TOTAL_PATH}/model_opt/{subfolder}/sim_{n_iter+1}/sim_weights_{Max_steps}_steps_{time_stamp}.pdf'), format="pdf", bbox_inches="tight")
        plt.close()

    see_annealing(states, costs)
    see_weights_cost(weights_cost)
    final_mAP50_sim.append(val_mAP_50)
    final_avg_sim.append(sum(state) / len(state))   
    final_MB_sim.append(final_size)             

    if len(final_avg_sim) > 1:
    # -------------- avg_bits
        mean_avg = sum(final_avg_sim) / len(final_avg_sim)
        std_avg = np.std(final_avg_sim)
        
        s_avg = np.random.normal(mean_avg, std_avg, 1000)
        count, bins, ignored = plt.hist(s_avg, 50, density=True, color='r', alpha=0.3)
        plt.plot(bins, 1 / (std_avg * np.sqrt(2 * np.pi)) * np.exp(-(bins - mean_avg) ** 2 / (2 * std_avg ** 2)), linewidth=2, color='r', label=f"μ:{mean_avg:.3f}, \u03c3:{std_avg:.3f} \n\u03B1:{alpha:.2f}, \u03B2:{beta:.2f}, \u03B3:{gamma:.2f}\n")
        plt.title(f"Distribution of the final avg bits with {simulations} simulations ")
        plt.legend()
        plt.savefig(os.path.join(config.ROOT_DIR,f'{config.BACKBONE}/{config.TOTAL_PATH}/model_opt/{subfolder}/distribution_avg_{Max_steps}_steps_{time_stamp}.pdf'), format="pdf", bbox_inches="tight")
        
        plt.close()
        # --------------accuracy
        mean_mAP50 = sum(final_mAP50_sim) / len(final_mAP50_sim)
        std_mAP50 = np.std(final_mAP50_sim)
        mean_mAP50 = mean_mAP50.item() if isinstance(mean_mAP50, torch.Tensor) else mean_mAP50
        std_mAP50 = std_mAP50.item() if isinstance(std_mAP50, torch.Tensor) else std_mAP50        
        s_mAP50 = np.random.normal(mean_mAP50, std_mAP50, 1000)
        count, bins, ignored = plt.hist(s_mAP50, 50, density=True, color='b', alpha=0.3)
        print(type(bins), type(mean_mAP50), type(std_mAP50))
        plt.plot(bins, 1 / (std_mAP50 * np.sqrt(2 * np.pi)) * np.exp(-(bins - mean_mAP50) ** 2 / (2 * std_mAP50 ** 2)), linewidth=2, color='b', label=f"μ:{mean_mAP50:.3f}, \u03c3:{std_mAP50:.3f} \n\u03B1:{alpha:.2f}, \u03B2:{beta:.2f}, \u03B3:{gamma:.2f}\n")
        plt.axvline(x=lower_bound / factor, label=f"Lower bound: {lower_bound / factor:.3f}", color='k', linestyle='dashed', linewidth=1)
        plt.title(f"Distribution of the final mAP50 with {simulations} simulations ")
        plt.legend()
        plt.savefig(os.path.join(config.ROOT_DIR,f'{config.BACKBONE}/{config.TOTAL_PATH}/model_opt/{subfolder}/distribution_mAP50_{Max_steps}_steps_{time_stamp}.pdf'), format="pdf", bbox_inches="tight")
        plt.close()



        mean_MB = sum(final_MB_sim) / len(final_MB_sim)
        std_MB = np.std(final_MB_sim)

        s_MB = np.random.normal(mean_MB, std_MB, 1000)
        count, bins, ignored = plt.hist(s_MB, 50, density=True, color='g', alpha=0.3)
        plt.plot(bins, 1 / (std_MB * np.sqrt(2 * np.pi)) * np.exp(-(bins - mean_MB) ** 2 / (2 * std_MB ** 2)), linewidth=2, color='g', label=f"μ:{mean_MB:.3f}, \u03c3:{std_MB:.3f} \n\u03B1:{alpha:.2f}, \u03B2:{beta:.2f}, \u03B3:{gamma:.2f}\n")
        plt.title(f"Distribution of the final Occupation of the parameters [MB] with {5} simulations ")
        plt.legend()
        plt.close()        
        # -------------- accuracy
        with open(os.path.join(config.ROOT_DIR,f'{config.BACKBONE}/{config.TOTAL_PATH}/model_opt/{subfolder}/final_mAP50_sim_{simulations}_sims_max_steps_{Max_steps}.pkl'), 'wb') as file:
            # A new file will be created
            pickle.dump(final_mAP50_sim, file)
            file.close()
        
        with open(os.path.join(config.ROOT_DIR,f'{config.BACKBONE}/{config.TOTAL_PATH}/model_opt/{subfolder}/final_avg_sim_{simulations}_sims_max_steps_{Max_steps}.pkl'), 'wb') as file:
            # A new file will be created
            pickle.dump(final_avg_sim, file)
            file.close()

        with open(os.path.join(config.ROOT_DIR,f'{config.BACKBONE}/{config.TOTAL_PATH}/model_opt/{subfolder}/final_MB_sim_{simulations}_sims_max_steps_{Max_steps}.pkl'), 'wb') as file:
            # A new file will be created
            pickle.dump(final_MB_sim, file)
            file.close()