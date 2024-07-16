import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import sys
# Custom imports
sys.path.append(os.getcwd())
import config

def save_results(num_epochs, mean_loss, mAP_50, mAP_75, mAP_90, learning_rate, mode = 'train'):
    # Construct the message to be saved
    message = f"{num_epochs} {mAP_50:.6f} {mAP_75:.6f} {mAP_90:.6f} {mean_loss:.6f} {learning_rate}\n"

    os.makedirs(os.path.join(config.ROOT_DIR, f"{config.BACKBONE}/{config.TOTAL_PATH}/results"), exist_ok=True)
    # Open the file in append mode if it exists, or in write mode if it doesn't exist
    with open(os.path.join(config.ROOT_DIR, f"{config.BACKBONE}/{config.TOTAL_PATH}/results/results_{mode}.txt"), "a+") as file:
        # Move the cursor to the start of the file to check if it's empty
        file.seek(0)
        # If the file is empty, write a header line
        if not file.readline():
            file.write("Epochs, mAP@50, mAP@75, mAP@90, Mean Loss, Learning Rate\n")
        # Append the current r
        # esults
        file.write(message)



def plot_results():
    # Initialize lists to store the data
    num_epochs_list_train = []
    mean_loss_list_train = []
    mAP_50_list_train = []
    mAP_75_list_train = []
    mAP_90_list_train = []
    learning_rate_list_train = []
    num_epochs_list_valid = []
    mean_loss_list_valid = []
    mAP_50_list_valid = []
    mAP_75_list_valid = []
    mAP_90_list_valid = []
    learning_rate_list_valid = []

    
    # Read the data from the file
    with open(os.path.join(config.ROOT_DIR, f'{config.BACKBONE}/{config.TOTAL_PATH}/results/results_train.txt')) as f:
        i = 0
        for line in f.readlines():
            if i!=0:
                # Split the line into its components
                num_epochs_train, mAP_50_train, mAP_75_train, mAP_90_train, mean_loss_train, learning_rate_train = line.split(' ')
                # Append the data to the lists, converting them to the appropriate types
                num_epochs_list_train.append(int(num_epochs_train))
                mean_loss_list_train.append(float(mean_loss_train))
                mAP_50_list_train.append(float(mAP_50_train))
                mAP_75_list_train.append(float(mAP_75_train))
                mAP_90_list_train.append(float(mAP_90_train))
                learning_rate_list_train.append(float(learning_rate_train))

            i += 1

    # Read the data from the file
    with open(os.path.join(config.ROOT_DIR, f'{config.BACKBONE}/{config.TOTAL_PATH}/results/results_valid.txt')) as f:
        i = 0
        for line in f.readlines():
            if i!=0:
                # Split the line into its components
                num_epochs_valid, mAP_50_valid, mAP_75_valid, mAP_90_valid, mean_loss_valid, learning_rate_valid = line.split(' ')
                # Append the data to the lists, converting them to the appropriate types
                num_epochs_list_valid.append(int(num_epochs_valid))
                mean_loss_list_valid.append(float(mean_loss_valid))
                mAP_50_list_valid.append(float(mAP_50_valid))
                mAP_75_list_valid.append(float(mAP_75_valid))
                mAP_90_list_valid.append(float(mAP_90_valid))
                learning_rate_list_valid.append(float(learning_rate_valid))

            i += 1
   
   

    # get max mAP index and value
    max_mAP_index = np.argmax(mAP_50_list_valid)
    max_mAP_value = mAP_50_list_valid[max_mAP_index]
    # get loss value for the best mAP
    min_loss_value = mean_loss_list_valid[max_mAP_index]

    # Plot num_epochs vs. mAP@50
    plt.figure(figsize=(10, 6))
    plt.plot(num_epochs_list_train, mAP_50_list_train, linestyle='-', color='b')
    plt.plot(num_epochs_list_train, mAP_50_list_valid, linestyle='-', color='r')
    plt.plot(num_epochs_list_valid[max_mAP_index], max_mAP_value, 'go')
    plt.title("mAP@50")
    plt.legend(["Train", "Valid", f"Best mAP@50: {max_mAP_value:.2f}"])
    plt.xlabel("Number of Epochs")
    plt.ylabel("Mean Average Precision")
    plt.grid(True)
    plt.savefig(os.path.join(config.ROOT_DIR,f"{config.BACKBONE}/{config.TOTAL_PATH}/results/epochs_vs_mAP50.png"))
    plt.close()

    # Plot num_epochs vs. mean_loss
    plt.figure(figsize=(10, 6))
    plt.plot(num_epochs_list_train, mean_loss_list_train, linestyle='-', color='b')
    plt.plot(num_epochs_list_train, mean_loss_list_valid, linestyle='-', color='r')
    plt.plot(num_epochs_list_valid[max_mAP_index], min_loss_value, 'go')
    plt.title("Eval Mean Loss")
    plt.legend(["Train", "Valid", f"Loss for Best mAP@50: {min_loss_value:.2f}"])
    plt.xlabel("Number of Epochs")
    plt.ylabel("Mean Loss")
    # plt.yscale('log')
    plt.grid(True)
    plt.savefig(os.path.join(config.ROOT_DIR, f"{config.BACKBONE}/{config.TOTAL_PATH}/results/epochs_vs_mean_loss.png"))
    plt.close()  # Close the plot to ensure it doesn't interfere with the next one
    

    # Plot num_epochs vs. mAP@50, mAP@75 and mAP@90 only val
    plt.figure(figsize=(10, 6))
    plt.plot(num_epochs_list_valid, mAP_50_list_valid, linestyle='-', color='b')
    plt.plot(num_epochs_list_valid, mAP_75_list_valid, linestyle='-', color='r')
    plt.plot(num_epochs_list_valid, mAP_90_list_valid, linestyle='-', color='g')
    plt.title("Validation mAP@50, mAP@75 and mAP@90")
    plt.legend(["mAP@50", "mAP@75", "mAP@90"])
    plt.xlabel("Number of Epochs")
    plt.ylabel("Mean Average Precision")
    plt.grid(True)
    plt.savefig(os.path.join(config.ROOT_DIR, f"{config.BACKBONE}/{config.TOTAL_PATH}/results/epochs_vs_mAP50_75_90.png"))
    plt.close() 
    
    
    # Plot num_epochs vs. log(learning_rate)
    plt.figure(figsize=(10, 6))
    plt.plot(num_epochs_list_train, learning_rate_list_train, marker='o', linestyle='-', color='b')
    plt.title("Learning Rate")
    plt.yscale('log')
    plt.xlabel("Number of Epochs")
    plt.ylabel("Learning Rate")
    plt.grid(True)
    plt.savefig(os.path.join(config.ROOT_DIR,f"{config.BACKBONE}/{config.TOTAL_PATH}/results/epochs_vs_learning_rate.png"))
    plt.close()

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(14, 6)) # 


    # Plot num_epochs vs. mAP@50
    axs[0].plot(num_epochs_list_train, mAP_50_list_train, linestyle='-', color='b')
    axs[0].plot(num_epochs_list_valid, mAP_50_list_valid, linestyle='-', color='r')
    axs[0].plot(num_epochs_list_valid[max_mAP_index], max_mAP_value, 'go')
    axs[0].set_title("mAP@50")
    axs[0].legend(["Train", "Valid", f"Best mAP@50: {max_mAP_value:.2f}"])
    axs[0].set_xlabel("Number of Epochs")
    axs[0].set_ylabel("Mean Average Precision")
    axs[0].grid(True)

    # Plot num_epochs vs. mean_loss
    axs[1].plot(num_epochs_list_train, mean_loss_list_train, linestyle='-', color='b')
    axs[1].plot(num_epochs_list_valid, mean_loss_list_valid, linestyle='-', color='r')
    axs[1].plot(num_epochs_list_valid[max_mAP_index], min_loss_value, 'go')
    axs[1].set_title("Eval Mean Loss")
    axs[1].legend(["Train", "Valid", f"Loss for Best mAP@50: {min_loss_value:.2f}"])
    axs[1].set_xlabel("Number of Epochs")
    axs[1].set_ylabel("Mean Loss")
    # axs[1].set_yscale('log')
    axs[1].grid(True)
   
    # Adjust layout
    plt.tight_layout()
    plt.savefig(os.path.join(config.ROOT_DIR, f"{config.BACKBONE}/{config.TOTAL_PATH}/results/summary_metrics.png"))
    plt.close()

     
    

def save_log(num_epochs, mean_loss, mean_loss_coord, mean_loss_obj, mean_loss_noobj, mean_loss_class, mode = 'train', eval_mode = 'train'):
    # Construct the message to be saved
    message = f"{num_epochs} {mean_loss:.3f} {mean_loss_coord:.3f} {mean_loss_obj:.3f} {mean_loss_noobj:.3f} {mean_loss_class:.3f}\n"


    os.makedirs(os.path.join(config.ROOT_DIR, f"{config.BACKBONE}/{config.TOTAL_PATH}/results"), exist_ok=True)
    # Open the file in append mode if it exists, or in write mode if it doesn't exist
    if mode == 'train':
        file_name = f"results_loss_training.txt"
    elif mode == 'eval':
        file_name = f"results_loss_eval_{eval_mode}.txt"
    with open(os.path.join(config.ROOT_DIR, f"{config.BACKBONE}/{config.TOTAL_PATH}/results/{file_name}"), "a+") as file:
        # Move the cursor to the start of the file to check if it's empty
        file.seek(0)
        # If the file is empty, write a header line
        if not file.readline():
            file.write("Epochs, Mean Loss, Mean Loss Coord, Mean Loss Obj, Mean Loss Noobj, Mean Loss Class\n")
        # Append the current results
        file.write(message)


def plot_loss(mode = 'train', eval_mode = 'train'):
    # Initialize lists to store the data
    num_epochs_list = []
    mean_loss_list = []
    mean_loss_coord_list = []
    mean_loss_obj_list = []
    mean_loss_noobj_list = []
    mean_loss_class_list = []

    
    # Read the data from the file
    if mode == 'train':
        file_name = f"results_loss_training.txt"
    elif mode == 'eval':
        file_name = f"results_loss_eval_{eval_mode}.txt"
    with open(os.path.join(config.ROOT_DIR, f'{config.BACKBONE}/{config.TOTAL_PATH}/results/{file_name}')) as f:
        i = 0
        for line in f.readlines():
            if i!=0:
                # Split the line into its components
                num_epochs, mean_loss, mean_loss_coord, mean_loss_obj, mean_loss_noobj, mean_loss_class = line.split(' ')
                # Append the data to the lists, converting them to the appropriate types
                num_epochs_list.append(int(num_epochs))
                mean_loss_list.append(float(mean_loss))
                mean_loss_coord_list.append(float(mean_loss_coord))
                mean_loss_obj_list.append(float(mean_loss_obj))
                mean_loss_noobj_list.append(float(mean_loss_noobj))
                mean_loss_class_list.append(float(mean_loss_class))

            i += 1

    # Plot num_epochs vs. mean_loss
    plt.figure(figsize=(7, 6))
    plt.plot(num_epochs_list, mean_loss_list, marker='o', linestyle='-', color='b')
    plt.title("Mean Loss")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Mean Loss")
    plt.grid(True)
    if mode == 'train':
        file_name = f"epochs_vs_mean_loss_training.png"
    elif mode == 'eval':
        file_name = f"epochs_vs_mean_loss_eval_{eval_mode}.png"
    plt.savefig(os.path.join(config.ROOT_DIR, f"{config.BACKBONE}/{config.TOTAL_PATH}/results/{file_name}"))
    plt.close()  # Close the plot to ensure it doesn't interfere with the next one

    # Plot num_epochs vs. mean_loss
    plt.figure(figsize=(7, 6))
    plt.plot(num_epochs_list, mean_loss_list, linestyle='-', color='b', label='Total Loss')
    plt.plot(num_epochs_list, mean_loss_coord_list,linestyle='-', color='r', label='Coord Loss')
    plt.plot(num_epochs_list, mean_loss_obj_list, linestyle='-', color='g', label='Object Loss')
    plt.plot(num_epochs_list, mean_loss_noobj_list, linestyle='-', color='y', label='No Object Loss')
    plt.plot(num_epochs_list, mean_loss_class_list, linestyle='-', color='m', label='Class Loss')
    plt.title("All losses")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    if mode == 'train':
        file_name = f"epochs_vs_all_losses_training.png"
    elif mode == 'eval':
        file_name = f"epochs_vs_all_losses_eval_{eval_mode}.png"
    plt.savefig(os.path.join(config.ROOT_DIR, f"{config.BACKBONE}/{config.TOTAL_PATH}/results/{file_name}"))
    plt.close()

def read_last_line(filename=os.path.join(config.ROOT_DIR,f'{config.BACKBONE}/{config.TOTAL_PATH}/results/results_train.txt')):
    with open(filename, 'r') as file:
        lines = file.readlines()
        if lines:  # Check if the list is not empty
            epoch = int(lines[-1].split(' ')[0])
            return epoch # .strip() removes any trailing newline characters
        else:
            return 0



def plot_AUC(precisions, recalls, pr_auc, epoch):
    import torch
    # Ordenar los valores para la visualización correcta
    sorted_indices = torch.argsort(recalls)
    sorted_recalls = recalls[sorted_indices]
    sorted_precisions = precisions[sorted_indices]

    # Cálculo del PR AUC usando la regla trapezoidal
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_recalls, sorted_precisions, marker='o', linestyle='-', color='blue', label=f'PR AUC: {pr_auc:.2f}, Epoch: {epoch}')
    plt.fill_between(sorted_recalls, 0, sorted_precisions, color='blue', alpha=0.2, label='Area under curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve with Area Under the Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(config.ROOT_DIR, f"{config.BACKBONE}/{config.TOTAL_PATH}/results/AUC_best_epoch.png"))
    plt.close()  # Close the plot to ensure it doesn't interfere with the next one    

def save_results_noise(total_layers_affected_1, slices1, nslices1,
                       total_layers_affected_2, slices2, nslices2,
                       mean_loss, mAP_50, mAP_75, mAP_90, file_name='case_1.txt', folder_save_results = 'results_noise'):
    # Construct the message to be saved
    message = f'{total_layers_affected_1} {slices1} {nslices1} {total_layers_affected_2} {slices2} {nslices2} {mAP_50:.6f} {mAP_75:.6f} {mAP_90:.6f} {mean_loss:.6f}\n'

    os.makedirs(folder_save_results, exist_ok=True)
    # Open the file in append mode if it exists, or in write mode if it doesn't exist
    with open(os.path.join(folder_save_results, file_name), "a+") as file:
        # Move the cursor to the start of the file to check if it's empty
        file.seek(0)
        # If the file is empty, write a header line
        if not file.readline():
            file.write("LayersAffected1, Slice1, nslices1, LayersAffected2, Slice2, nslices2, mAP@50, mAP@75, mAP@90, Mean Loss\n")
        # Append the current results
        file.write(message)

def plot_results_noise(result_name_file, folder_save_results, case1, case2, max_limit_1, max_limit_2):
    layers_affected_1 = []
    slices1 = []
    nslices1 = []
    layers_affected_2 = []
    slices2 = []
    nslices2 = []
    mAP_50 = []
    mAP_75 = []
    mAP_90 = []
    mean_loss = []

    with open(os.path.join(folder_save_results, result_name_file)) as f:
        i = 0
        for line in f.readlines():
            if i!=0:
                # noise1, noise2, mAP50, mAP75, mAP90, loss = line.split(' ')
                line.split()
                n_affected_1, slice1, nslice1, n_affected_2, slice2, nslice2, mAP50, mAP75, mAP90, loss = line.split(' ')
                layers_affected_1.append(float(n_affected_1))
                slices1.append(int(slice1))
                nslices1.append(int(nslice1))
                layers_affected_2.append(float(n_affected_2))
                slices2.append(int(slice2))
                nslices2.append(int(nslice2))
                mAP_50.append(float(mAP50))
                mAP_75.append(float(mAP75))
                mAP_90.append(float(mAP90))
                mean_loss.append(float(loss))

            i += 1
    
    if case1 and not case2:
        unique_slices1 = sorted(list(set(slices1)))

        if sum(mean_loss) != 0:
            # Create subplots
            fig, axs = plt.subplots(1, 2, figsize=(14, 6))

            # Plot Noise Level 2 vs mAP@50 for different percentages levels
            txt_legend = []

            for s1 in unique_slices1:
                indices = [i for i in range(len(slices1)) if slices1[i] == s1]
                axs[0].plot([layers_affected_1[i] for i in indices], [mAP_50[i] for i in indices], linestyle='-', label=f'slice1={s1}', marker='o')
                txt_legend.append(f'Slice: {s1+1}/{len(unique_slices1)}, Num activation functions: {max_limit_1[s1]}')
            axs[0].set_title("SEU Effect, mAP analysis for different percentages levels")
            axs[0].set_xlabel("Number of activation function affected")
            axs[0].set_ylabel("Mean Average Precision")
            if min(mAP_50) > 0.5:
                axs[0].set_ylim(min(mAP_50)*0.9, max(mAP_50)*1.1)
            axs[0].legend(txt_legend, loc='upper right')
            axs[0].grid(True)

            # Plot Noise Level 1 vs Mean Loss for different percentages levels
            txt_legend = []
            for s1 in unique_slices1:
                indices = [i for i in range(len(slices1)) if slices1[i] == s1]
                axs[1].plot([layers_affected_1[i] for i in indices], [mean_loss[i] for i in indices], linestyle='-', label=f'slice1={s1}', marker='o')
                txt_legend.append(f'Slice: {s1+1}/{len(unique_slices1)}, Num activation functions: {max_limit_1[s1]}')
            axs[1].set_title("SEU Effect, Loss analysis for different percentages levels")
            axs[1].set_xlabel("Number of activation function affected")
            axs[1].set_ylabel("Mean Loss")
            axs[1].legend(txt_legend, loc = 'upper left')# loc arriba izqda
            axs[1].grid(True)

            # Adjust layout
            plt.tight_layout()
            plot_file = result_name_file.replace('.txt', '.png')
            plt.savefig(os.path.join(folder_save_results,f"{plot_file}"))
            plt.close()
        else:
            #only plot map
            plt.figure(figsize=(10, 6))
            for s1 in unique_slices1:
                indices = [i for i in range(len(slices1)) if slices1[i] == s1]
                plt.plot([layers_affected_1[i] for i in indices], [mAP_50[i] for i in indices], linestyle='-', label=f'slice1={s1}', marker='o')
            plt.title("SEU Effect, mAP analysis for different percentages levels")
            plt.xlabel("Number of activation function affected")
            plt.ylabel("Mean Average Precision")
            plt.ylim(-0.05, max(mAP_50)*1.1)
            plt.legend()
            plt.grid(True)
            plot_file = result_name_file.replace('.txt', '.png')
            plt.savefig(os.path.join(folder_save_results,f"{plot_file}"))
            plt.close()
        

    if case2 and not case1:
        unique_slices2 = sorted(list(set(slices2)))

        if sum(mean_loss) != 0:
            # Create subplots
            fig, axs = plt.subplots(1, 2, figsize=(14, 6))

            # Plot Noise Level 2 vs mAP@50 for different percentages levels
            txt_legend = []
            for s2 in unique_slices2:
                indices = [i for i in range(len(slices2)) if slices2[i] == s2]
                axs[0].plot([layers_affected_2[i] for i in indices], [mAP_50[i] for i in indices], linestyle='-', label=f'slice2={s2}', marker='o')
                txt_legend.append(f'Slice: {s2+1}/{len(unique_slices2)}, Layers affected: {max_limit_2[s2]}')
            axs[0].set_title("SEL Effect, mAP analysis for different percentages levels")
            axs[0].set_xlabel("Number of layers affected")
            axs[0].set_ylabel("Mean Average Precision")
            axs[0].set_ylim(-0.05, max(mAP_50)*1.1)
            axs[0].legend(txt_legend, loc='upper right')
            axs[0].grid(True)

            # Plot Noise Level 3 vs Mean Loss for different percentages levels
            txt_legend = []
            for s2 in unique_slices2:
                indices = [i for i in range(len(slices2)) if slices2[i] == s2]
                axs[1].plot([layers_affected_2[i] for i in indices], [mean_loss[i] for i in indices], linestyle='-', label=f'slice2={s2}', marker='o')
                txt_legend.append(f'Slice: {s2+1}/{len(unique_slices2)}, Layers affected: {max_limit_2[s2]}')
            axs[1].set_title("SEL Effect, Loss analysis for different percentages levels")
            axs[1].set_xlabel("Number of layers affected")
            axs[1].set_ylabel("Mean Loss")
            axs[1].legend(txt_legend, loc='upper left')
            axs[1].grid(True)

            # Adjust layout
            plt.tight_layout()
            plot_file = result_name_file.replace('.txt', '.png')
            plt.savefig(os.path.join(folder_save_results,f"{plot_file}"))
            plt.close()
        else:
            #only plot map
            plt.figure(figsize=(10, 6))
            for s2 in unique_slices2:
                indices = [i for i in range(len(slices2)) if slices2[i] == s2]
                plt.plot([layers_affected_2[i] for i in indices], [mAP_50[i] for i in indices], linestyle='-', label=f'slice2={s2}', marker='o')
            plt.title("SEL Effect, mAP analysis for different percentages levels")
            plt.xlabel("Number of layers affected")
            plt.ylabel("Mean Average Precision")
            plt.ylim(-0.05, max(mAP_50)*1.1)
            plt.legend()
            plt.grid(True)
            plot_file = result_name_file.replace('.txt', '.png')
            plt.savefig(os.path.join(folder_save_results,f"{plot_file}"))
            plt.close()
        



   


def save_simulated_annealing_results(step, avg_bits, mAP50, loss, cost_annealing, temp, subfolder, file_name='simulated_annealing.txt', path_yolov8n = None):
    # Construct the message to be saved
    message = f"{step} {avg_bits:.2f} {mAP50:.6f} {loss:.6f} {cost_annealing:.6f} {temp:.6f}\n"

    if path_yolov8n is None:
        os.makedirs(os.path.join(config.ROOT_DIR, f"{config.BACKBONE}/{config.TOTAL_PATH}/model_opt/{subfolder}"), exist_ok=True)
        save_file = os.path.join(config.ROOT_DIR, f"{config.BACKBONE}/{config.TOTAL_PATH}/model_opt/{subfolder}/{file_name}")
    else:
        os.makedirs(os.path.join(path_yolov8n, f"model_opt/{subfolder}"), exist_ok=True)
        save_file = os.path.join(path_yolov8n, f"model_opt/{subfolder}/{file_name}")
    # Open the file in append mode if it exists, or in write mode if it doesn't exist
    with open(save_file, "a+") as file:
        # Move the cursor to the start of the file to check if it's empty
        file.seek(0)
        # If the file is empty, write a header line
        if not file.readline():
            file.write("Step, Average Bits, mAP@50, Loss, Cost Annealing, Temperature\n")
        # Append the current results
        file.write(message)




def save_acceptance_prob(step, val_mAP, acceptance, num_params_to_change, subfolder, file_name='acceptance_prob.txt', path_yolov8n = None):
    # Construct the message to be saved
    message = f"{step} {val_mAP:6f} {acceptance} {num_params_to_change}\n"

    if path_yolov8n is None:
        os.makedirs(os.path.join(config.ROOT_DIR, f"{config.BACKBONE}/{config.TOTAL_PATH}/model_opt/{subfolder}"), exist_ok=True)
        save_file = os.path.join(config.ROOT_DIR, f"{config.BACKBONE}/{config.TOTAL_PATH}/model_opt/{subfolder}/{file_name}")
    else:
        os.makedirs(os.path.join(path_yolov8n, f"model_opt/{subfolder}"), exist_ok=True)
        save_file = os.path.join(path_yolov8n, f"model_opt/{subfolder}/{file_name}")    # Open the file in append mode if it exists, or in write mode if it doesn't exist
    with open(save_file, "a+") as file:
        # Move the cursor to the start of the file to check if it's empty
        file.seek(0)
        # If the file is empty, write a header line
        if not file.readline():
            file.write("Step, mAP@50, Acceptance Probability, Num Params to Change\n")
        # Append the current results
        file.write(message)

def plot_simulated_annealing_results(gamma, beta, alpha, subfolder, val_map_ini, avg_bits_ini, initial_mb, size_mb, lower_bound, result_name_file = 'simulated_annealing.txt', path_yolov8n = None):
    steps = []
    avg_bits = []
    mAP_50 = []
    mean_loss = []
    mean_cost_annealing = []
    mean_temp = []
    step_acceptances = []
    map_acceptance = []
    acceptance_boolean_array = []
    num_params_to_change = []

    if path_yolov8n is None:
        results_file = os.path.join(config.ROOT_DIR, f"{config.BACKBONE}/{config.TOTAL_PATH}/model_opt/{subfolder}/{result_name_file}")
        save_figs = os.path.join(config.ROOT_DIR, f"{config.BACKBONE}/{config.TOTAL_PATH}/model_opt/{subfolder}")
    else:
        results_file = os.path.join(path_yolov8n, f"model_opt/{subfolder}/{result_name_file}")
        save_figs = os.path.join(path_yolov8n, f"model_opt/{subfolder}")

    with open(results_file) as f:
        i = 0
        for line in f.readlines():
            if i!=0:
                step, bits, mAP50, loss, cost_annealing, temp = line.split(' ')
                steps.append(int(step))
                avg_bits.append(float(bits))
                mAP_50.append(float(mAP50))
                mean_loss.append(float(loss))
                mean_cost_annealing.append(float(cost_annealing))
                mean_temp.append(float(temp))

            i += 1
    # Verify if exists acceptance probability
    if path_yolov8n is None:
        acceptance_file = os.path.join(config.ROOT_DIR, f"{config.BACKBONE}/{config.TOTAL_PATH}/model_opt/{subfolder}/acceptance_prob.txt")
    else:
        acceptance_file = os.path.join(path_yolov8n, f"model_opt/{subfolder}/acceptance_prob.txt")

    if os.path.exists(acceptance_file):
        with open(acceptance_file) as f:
            i = 0
            for line in f.readlines():
                if i!=0:
                    step_acceptance, map50, acceptance, params_to_change = line.split(' ')
                    # if acceptance == '0' or int(acceptance)==0:
                    step_acceptances.append(int(step_acceptance))
                    map_acceptance.append(float(map50))
                    acceptance_boolean_array.append(int(acceptance))
                    num_params_to_change.append(int(params_to_change))
                i += 1

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(steps)), mAP_50, linestyle='-', color='b')
    for i in range(0,len(step_acceptances)):
        if int(acceptance_boolean_array[i]) == 0:
            plt.plot(step_acceptances[i]+1, map_acceptance[i], 'x', color='r')
        else:
            plt.plot(step_acceptances[i]+1, map_acceptance[i], 'o', color='g')
    # Lower bound is a float convert to numpy array
    threshold_map = np.ones(len(steps))*np.array(lower_bound)
    plt.plot(range(len(steps)), threshold_map, linestyle='--', color='black')
    plt.title("mAP@50")
    plt.xlabel("Steps")
    plt.ylabel("Mean Average Precision")

    # Crear líneas ficticias para la leyenda
    legend_elements = [
        Line2D([0], [0], color='b', lw=2, label=f'Initial mAP@50: {val_map_ini:.3f} Full Precision Avg Bits: {avg_bits_ini:.2f} \nFinal mAP@50: {mAP_50[-1]:.3f} Avg Bits: {avg_bits[-1]:.2f} \n {initial_mb:1.3f} MB -> {size_mb:1.3f} MB'),
        Line2D([0], [0], marker='o', color='green',  markersize=10, label='Accepted'),
        Line2D([0], [0], marker='x', color='red',  markersize=10, label='Rejected'),
        Line2D([0], [0], linestyle='--', color='black', label=f'Threshold mAP@50: {lower_bound:.3f}'),
    ]

    # Añadir la leyenda
    plt.legend(handles=legend_elements, loc='lower right', fontsize=12)
    plt.grid(True)
    plt.savefig(os.path.join(save_figs, f"steps_vs_mAP50.png"))
    plt.close()

    if sum(mean_loss) != 0:
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(steps)), mean_loss, linestyle='-', color='b')
        plt.title("Steps Bits vs Loss")
        plt.xlabel("Steps")
        plt.ylabel("Mean Loss")
        plt.yscale('log')
        plt.grid(True)
        plt.savefig(os.path.join(save_figs, f"steps_vs_loss.png"))
        plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(steps)), mean_cost_annealing, linestyle='-', color='b')
    plt.title("Cost Annealing \n"+ r"Cost function=$\gamma*(lower_{bound}-mAP@50)^2 + \beta*avg_{bits} -\alpha*lower_{bound}$")
    plt.xlabel("Steps")
    plt.ylabel("Mean Cost Annealing")
    plt.legend([rf"$\gamma$={gamma}, $\beta$={beta}, $\alpha$={alpha}"])
    plt.grid(True)
    plt.savefig(os.path.join(save_figs, f"steps_vs_cost_annealing.png"))
    plt.close()

    # Crear un colormap que vaya de rojo a azul
    cmap = plt.get_cmap('coolwarm')

    # Normalizar los valores de temperatura para el colormap
    norm = plt.Normalize(vmin=min(mean_temp), vmax=max(mean_temp))

    plt.figure(figsize=(10, 6))
    for i in range(len(steps)-1):
        plt.plot(steps[i:i+2], mean_temp[i:i+2], linestyle='-', color=cmap(norm(mean_temp[i])))    
    plt.title("Temperature")
    plt.xlabel("Steps")
    plt.ylabel("Mean Temperature")
    plt.grid(True)
    plt.savefig(os.path.join(save_figs, f"steps_vs_temp.png"))
    plt.close()

    # plot avg bits
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(steps)), avg_bits, linestyle='-', color='b')
    plt.title("Average Bits")
    plt.xlabel("Steps")
    plt.ylabel("Average Bits")
    plt.grid(True)
    plt.savefig(os.path.join(save_figs, f"steps_vs_avg_bits.png"))
    plt.close()

    # plot of a histogram of the number of parameters to change
    plt.figure(figsize=(10, 6))
    plt.bar(step_acceptances, num_params_to_change, edgecolor='blue')
    plt.title("Histogram of the number of parameters to change")
    plt.xlabel("Steps")
    plt.ylabel("Number of parameters to change")
    plt.grid(True)
    plt.savefig(os.path.join(save_figs, f"steps_vs_num_params_to_change.png"))
    plt.close()

    
    fig, axs = plt.subplots(5, 1, figsize=(10, 10)) #
    # Plot num_epochs vs. mAP@50
    axs[0].plot(range(len(steps)), mAP_50, linestyle='-', color='b')
    # Include threshold
    threshold_map = np.ones(len(steps))*np.array(lower_bound)
    axs[0].plot(range(len(steps)), threshold_map, linestyle='--', color='black')
    axs[0].set_title("mAP@50")
    axs[0].set_xlabel("Steps")
    axs[0].set_ylabel("Mean Average Precision")
    if min(mAP_50) > 0.5:
        axs[0].set_ylim(min(mAP_50)*0.9, 1)
    # Legend
    axs[0].legend([f'Initial mAP@50: {val_map_ini:.3f} Full Precision Avg Bits: {avg_bits_ini:.2f} \nFinal mAP@50: {mAP_50[-1]:.3f} Avg Bits: {avg_bits[-1]:.2f} \n {initial_mb:1.3f} MB -> {size_mb:1.3f} MB', f'Threshold mAP@50: {lower_bound:.3f}'], loc='lower right', fontsize=8)
    axs[0].grid(True)

    # Plot num_epochs vs. mean_loss_annealing
    axs[1].plot(range(len(steps)), mean_cost_annealing, linestyle='-', color='b')
    axs[1].set_title("Cost Annealing \n" + r"Cost function=" + f"{gamma}" + r"*$(lower_{bound}-mAP@50)^2$+"+ f"{beta}*"+ r"$avg_{bits}$-"+ f"{alpha}*"+r"$lower_{bound}$")
    axs[1].set_xlabel("Steps")
    axs[1].set_ylabel("Mean Cost Annealing")
    axs[1].grid(True)


    # Plot Avg Bits
    axs[2].plot(range(len(steps)), avg_bits, linestyle='-', color='b')
    axs[2].set_title("Average Bits")
    axs[2].set_xlabel("Steps")
    axs[2].set_ylabel("Average Bits")
    axs[2].grid(True)

    # Plot num_temperature
    for i in range(len(steps)-1):
        axs[3].plot(steps[i:i+2], mean_temp[i:i+2], linestyle='-', color=cmap(norm(mean_temp[i])))
    axs[3].set_title("Temperature")
    axs[3].set_xlabel("Steps")
    axs[3].set_ylabel("Mean Temperature")
    axs[3].grid(True)


    # Plot num_hist num params to change
    axs[4].bar(step_acceptances, num_params_to_change, edgecolor='blue')
    axs[4].set_title("Histogram of the number of parameters to change")
    axs[4].set_xlabel("Steps")
    axs[4].set_ylabel("Number of parameters to change")
    axs[4].grid(True)


    # Adjust layout
    plt.tight_layout()
    plt.savefig(os.path.join(save_figs, f"summary_metrics.png"))
    plt.close()  # Close the plot to ensure it doesn't interfere with the next one

def save_final_state(layers_of_interest, n_int, state_frac, subfolder, file_name='final_precision_weights.txt', path_yolov8n=None):
    # Construct the message to be saved
    message = ""

    for i in range(len(layers_of_interest)):
        max_value = 2**(n_int[i]-1) - 1 + 2**(-state_frac[i])
        min_value = -2**(n_int[i]-1) + 2**(-state_frac[i])        
        message += f"{layers_of_interest[i]}: ({n_int[i]},{state_frac[i]}), Range: [{min_value}, {max_value}]\n"
    
    if path_yolov8n is None:
        os.makedirs(os.path.join(config.ROOT_DIR, f"{config.BACKBONE}/{config.TOTAL_PATH}/model_opt/{subfolder}"), exist_ok=True)
        save_file = os.path.join(config.ROOT_DIR, f"{config.BACKBONE}/{config.TOTAL_PATH}/model_opt/{subfolder}/{file_name}")
    else:
        os.makedirs(os.path.join(path_yolov8n, f"model_opt/{subfolder}"), exist_ok=True)
        save_file = os.path.join(path_yolov8n, f"model_opt/{subfolder}/{file_name}")

    # Open the file in append mode if it exists, or in write mode if it doesn't exist
    with open((save_file), "a+") as file:
        # Move the cursor to the start of the file to check if it's empty
        file.seek(0)
        # If the file is empty, write a header line
        if not file.readline():
            file.write("Layers state_frac\n")
        # Append the current results
        file.write(message)


def save_history_state(state_frac, subfolder, file_name='history_precision_weights.txt', path_yolov8n=None):
    # Construct the message to be saved
    message = ""

    if path_yolov8n is None:
        os.makedirs(os.path.join(config.ROOT_DIR, f"{config.BACKBONE}/{config.TOTAL_PATH}/model_opt/{subfolder}"), exist_ok=True)
        save_file = os.path.join(config.ROOT_DIR, f"{config.BACKBONE}/{config.TOTAL_PATH}/model_opt/{subfolder}/{file_name}")
    else:
        os.makedirs(os.path.join(path_yolov8n, f"model_opt/{subfolder}"), exist_ok=True)
        save_file = os.path.join(path_yolov8n, f"model_opt/{subfolder}/{file_name}")

    # Open the file in append mode if it exists, or in write mode if it doesn't exist
    with open((save_file), "a+") as file:        # Move the cursor to the start of the file to check if it's empty
        txt = ''
        for i in state_frac:
            txt += f'|{i}'.ljust(3)
        
        txt += '\n'
        file.write(txt)

