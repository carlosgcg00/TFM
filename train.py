import csv
import torch
import torch.optim as optim
from tqdm import tqdm
from backbone import resnet50, vgg16, efficientnet_b0
from pathlib import Path
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
    get_loaders,
    warmup_scheduler,
    update_lr,
    update_lr_no_pretrained,
    get_lr,
    seed_everything,
)
from loss import YoloLoss
from tinyissimo_model import tinyissimoYOLO
from ext_tinyissimo_model import ext_tinyissimoYOLO
from YOLOv1 import Yolov1
from bed_model import bedmodel
import config
from torch.nn.utils import clip_grad_norm_
import os
import shutil
from save_results import (
    save_results,
    save_log,
    plot_loss,
    plot_results,
    read_last_line
)
from summary_training import save_summary
import io
import warnings


# seed = 123
# torch.manual_seed(seed)

seed_everything()
   

def train_fn(train_loader, model, optimizer, scheduler, loss_fn, initial_epoch, epoch):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []
    mean_loss_box_coordinates = []
    mean_loss_object_loss = []
    mean_loss_no_object_loss = []
    mean_loss_class = []
    loop.set_description(f"Epoch [{epoch}/{config.EPOCHS  + initial_epoch -1}]")
    for batch_idx, (x, y) in enumerate(loop):
        if not config.LR_SCHEDULER == 'None':
            if config.LR_SCHEDULER == 'LearningRateScheduler':
                #Update learning rate.
                update_lr(optimizer, epoch)
                lr = get_lr(optimizer)
            if config.PRETRAINED and epoch < config.WARM_UP:
                warmup_scheduler(optimizer, epoch)
                lr = get_lr(optimizer)
            if not config.PRETRAINED and epoch < config.WARM_UP:
                warmup_scheduler(optimizer, epoch)
                lr = get_lr(optimizer)
            if config.LR_SCHEDULER == 'SchedulerNoPretrainedModels':
                update_lr_no_pretrained(optimizer, epoch)
                lr = get_lr(optimizer)


        x, y = x.to(config.DEVICE), y.to(config.DEVICE)
        out = model(x)
        loss, loss_box_coordinates, loss_object_loss, loss_no_object_loss, loss_class = loss_fn(out, y)
        # L1 regularization, Lasso regularization
        l1_lambda = config.L1_LAMBDA  # hyperparameter for L1 regularization
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        
        loss = loss + l1_lambda * l1_norm

        mean_loss.append(loss.item())
        mean_loss_box_coordinates.append(loss_box_coordinates.item())
        mean_loss_object_loss.append(loss_object_loss.item())
        mean_loss_no_object_loss.append(loss_no_object_loss.item())
        mean_loss_class.append(loss_class.item())
             
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)  # Limita la norma de los gradientes
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())
    # Update the learning rate
    if not config.LR_SCHEDULER == 'None' :
        if config.LR_SCHEDULER != 'LearningRateScheduler' and config.LR_SCHEDULER != 'SchedulerNoPretrainedModels' and epoch >= config.WARM_UP:
            scheduler.step(sum(mean_loss)/len(mean_loss))   
            lr = scheduler.get_last_lr()[0]
    if config.LR_SCHEDULER == 'None':
        lr = optimizer.param_groups[0]['lr']
    # print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")
    # Resumen loss
    mean_loss_out = sum(mean_loss)/len(mean_loss)
    mean_loss_box_coordinates_out = sum(mean_loss_box_coordinates)/len(mean_loss_box_coordinates)
    mean_loss_object_loss_out = sum(mean_loss_object_loss)/len(mean_loss_object_loss)
    mean_loss_no_object_loss_out = sum(mean_loss_no_object_loss)/len(mean_loss_no_object_loss)
    mean_loss_class_out = sum(mean_loss_class)/len(mean_loss_class)

    print("---------------------------------------------------")
    print('-----------Loss Summary during training------------')
    print("Total Loss".ljust(12) + "|" + 
          "Loss Coord".ljust(12) + "|" + 
          "Conf Loss".ljust(12) + "|" + 
          "No Obj Loss".ljust(12) + "|" +
            "Class Loss".ljust(12) + "|") 
    print(f'{mean_loss_out:.3f}'.ljust(12) + "|" +
          f'{mean_loss_box_coordinates_out:.3f}'.ljust(12) + "|" +
          f'{mean_loss_object_loss_out:.3f}'.ljust(12) + "|" +
          f'{mean_loss_no_object_loss_out:.3f}'.ljust(12) + "|"+
          f'{mean_loss_class_out:.3f}'.ljust(12) + "|")  
    print("---------------------------------------------------")

    save_log(epoch, mean_loss_out, mean_loss_box_coordinates_out, mean_loss_object_loss_out, mean_loss_no_object_loss_out, mean_loss_class_out, mode='train')
    plot_loss()
    return lr

def main():
    save_summary()
    #Choose model from backbone.py
    if config.BACKBONE == 'resnet50':
        model = resnet50(split_size=config.SPLIT_SIZE, num_boxes=config.NUM_BOXES, num_classes=config.NUM_CLASSES, pretrained = True).to(config.DEVICE)
    elif config.BACKBONE == 'vgg16':
        model = vgg16(split_size=config.SPLIT_SIZE, num_boxes=config.NUM_BOXES, num_classes=config.NUM_CLASSES, pretrained = True).to(config.DEVICE)
    elif config.BACKBONE == 'efficientnet':
        model = efficientnet_b0(split_size=config.SPLIT_SIZE, num_boxes=config.NUM_BOXES, num_classes=config.NUM_CLASSES).to(config.DEVICE)
    elif config.BACKBONE == 'tinyissimoYOLO':
        model = tinyissimoYOLO(split_size=config.SPLIT_SIZE, num_boxes=config.NUM_BOXES, num_classes=config.NUM_CLASSES).to(config.DEVICE)
    elif config.BACKBONE == 'ext_tinyissimoYOLO':
        model = ext_tinyissimoYOLO(split_size=config.SPLIT_SIZE, num_boxes=config.NUM_BOXES, num_classes=config.NUM_CLASSES).to(config.DEVICE)
    elif config.BACKBONE == 'bedmodel':
        model = bedmodel(split_size=config.SPLIT_SIZE, num_boxes=config.NUM_BOXES, num_classes=config.NUM_CLASSES).to(config.DEVICE)
    elif config.BACKBONE == 'Yolov1':
        model = Yolov1(split_size=config.SPLIT_SIZE, num_boxes=config.NUM_BOXES, num_classes=config.NUM_CLASSES).to(config.DEVICE)

    # Print the number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    # Print the size of the model before training
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_bytes = buffer.tell()
    size_mb = size_bytes / (1024 ** 2)
    print(f'{config.BACKBONE} | Parameters: {total_params} | Size: {size_mb:.2f} MB')


    ############################################
    #           Define the optimizer           #
    ############################################
    if config.OPTIMIZER == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config.INIT_lr, weight_decay=config.WEIGHT_DECAY, momentum=0.7)
    elif config.OPTIMIZER == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config.INIT_lr, weight_decay=config.WEIGHT_DECAY)
    elif config.OPTIMIZER == 'NAdam':
        optimizer = optim.NAdam(model.parameters(), lr=config.INIT_lr, weight_decay=config.WEIGHT_DECAY)  
    
    ############################################
    #           Define the scheduler           #
    ############################################
    if config.LR_SCHEDULER == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5) # Reduce the Learning Rate when a metric has stopped improving
    elif config.LR_SCHEDULER == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0.00005)
    elif config.LR_SCHEDULER == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    elif config.LR_SCHEDULER == 'ExponentialLR':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    elif config.LR_SCHEDULER == 'None' or config.LR_SCHEDULER == 'LearningRateScheduler' or config.LR_SCHEDULER == 'SchedulerNoPretrainedModels':
        scheduler = None

    
    ############################################
    #           Define the loss function       #
    ############################################
    loss_fn = YoloLoss(S=config.SPLIT_SIZE, B=config.NUM_BOXES, C=config.NUM_CLASSES)

    #Load data into data loader
    train_loader, val_loader, train_eval_loader = get_loaders()



    ############################################
    #       Load the model and the results     #
    ############################################
    if config.LOAD_MODEL:
        # Read the file results.txt to see the initial epoch
        initial_epoch = read_last_line() 
        initial_epoch = int(initial_epoch/1)*1
    else:
        initial_epoch = 0
        # # If exists results folder, remove it
        if os.path.exists(os.path.join(config.DRIVE_PATH, f"{config.BACKBONE}/{config.TOTAL_PATH}/results")):
            shutil.rmtree(os.path.join(config.DRIVE_PATH, f"{config.BACKBONE}/{config.TOTAL_PATH}/results"), )
        # if exists model folder, remove it
        if os.path.exists(os.path.join(config.DRIVE_PATH, f"{config.BACKBONE}/{config.TOTAL_PATH}/model")):
            shutil.rmtree(os.path.join(config.DRIVE_PATH, f"{config.BACKBONE}/{config.TOTAL_PATH}/model"), )
        
        # Create the results folder
        os.makedirs(os.path.join(config.DRIVE_PATH, f"{config.BACKBONE}/{config.TOTAL_PATH}/results"), exist_ok=True)
        # Create the model folder
        os.makedirs(os.path.join(config.DRIVE_PATH, f"{config.BACKBONE}/{config.TOTAL_PATH}/model"), exist_ok=True)

    if config.LOAD_MODEL:
        load_checkpoint(torch.load(os.path.join(config.DRIVE_PATH,f'{config.BACKBONE}/{config.TOTAL_PATH}/model/YOLO_epoch_{initial_epoch}.pth.tar')), model, optimizer)
        initial_epoch +=1




    #Create empty list to store accuracy values to plot later
    best_mAP = 0
    best_epoch = 0

    #Start counting time
    import time
    start_full_time = time.time()

    #Start training Loop
    for epoch in range(initial_epoch, initial_epoch+config.EPOCHS):
        start_time = time.time() 

        ############################################
        #         Active gradient calculation      #
        ############################################
        if epoch >= config.WARM_UP:
            for param in model.parameters():
                param.requires_grad = True


        ############################################
        #         Training the model               #
        ############################################ 
        model.train()       
        last_lr = train_fn(train_loader, model, optimizer, scheduler, loss_fn, initial_epoch,epoch)

        ############################################
        #       Evaluating for training            #
        ############################################
        model.eval()
        pred_boxes, target_boxes, train_loss = get_bboxes(epoch, train_eval_loader, model, loss_fn,iou_threshold=0.5, threshold=0.5, device=config.DEVICE, mode='Train')
        train_mAP_50, train_mAP_75, train_mAP_90 = mean_average_precision(pred_boxes, target_boxes, iou_thresholds=[0.5, 0.75, 0.9], box_format="midpoint", num_classes=config.NUM_CLASSES, best_mAP=best_mAP, epoch=epoch, mode='Train')
        print(f"Train: \t mAP@50: {train_mAP_50:.6f}, mAP@75: {train_mAP_75:.6f}, mAP@90: {train_mAP_90:.6f}, Mean Loss: {train_loss:.6f}")
        save_results(epoch, train_loss, train_mAP_50, train_mAP_75, train_mAP_90, learning_rate = last_lr, mode='train')
        

        ############################################
        #       Evaluating for valid               #
        ############################################
        model.eval()
        pred_boxes, target_boxes, val_loss = get_bboxes(epoch, val_loader, model, loss_fn, iou_threshold=0.5, threshold=0.5, device=config.DEVICE, mode='Valid')
        val_mAP_50, val_mAP_75, val_mAP_90 = mean_average_precision(pred_boxes, target_boxes, iou_thresholds=[0.5, 0.75, 0.9], box_format="midpoint", num_classes=config.NUM_CLASSES, best_mAP=best_mAP, epoch=epoch, mode='Valid')
        print(f"Valid: \t mAP@50: {val_mAP_50:.6f}, mAP@75: {val_mAP_75:.6f}, mAP@90: {val_mAP_90:.6f}, Mean Loss: {val_loss:.6f}")
        save_results(epoch, val_loss, val_mAP_50, val_mAP_75, val_mAP_90, learning_rate = last_lr, mode='valid')
        plot_results()

        torch.cuda.empty_cache()
        print('----------------------------------------')
        print(f"Epoch: {epoch}" + " |" f"Train mAP: {train_mAP_50:.6f}" + " |" + f"Validation mAP: {val_mAP_50:.6f}")
        print('----------------------------------------')
        
        
        ############################################
        #  Saving the best model and the best loss #
        ############################################
        if config.SAVE_MODEL:
            if epoch % 1 == 0 and epoch > 0:  # Guardar cada 2 épocas
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                if os.path.exists(os.path.join(config.DRIVE_PATH, f"{config.BACKBONE}/{config.TOTAL_PATH}/model/YOLO_epoch_{epoch-1}.pth.tar")):
                    os.remove(os.path.join(config.DRIVE_PATH, f"{config.BACKBONE}/{config.TOTAL_PATH}/model/YOLO_epoch_{epoch-1}.pth.tar"))
                save_checkpoint(checkpoint, filename=os.path.join(config.DRIVE_PATH, f"{config.BACKBONE}/{config.TOTAL_PATH}/model/YOLO_epoch_{epoch}.pth.tar"))
        
            if epoch > 0:
                # We want to save the best model and avoid overfitting
                if val_mAP_50 > best_mAP and train_mAP_50-val_mAP_50 < 0.15:
                    checkpoint = {
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }
                    if os.path.exists(os.path.join(config.DRIVE_PATH, f"{config.BACKBONE}/{config.TOTAL_PATH}/model/{best_epoch}_YOLO_best.pth.tar")):
                        os.remove(os.path.join(config.DRIVE_PATH, f"{config.BACKBONE}/{config.TOTAL_PATH}/model/{best_epoch}_YOLO_best.pth.tar"))
                    best_mAP = val_mAP_50
                    best_epoch = epoch
                    save_checkpoint(checkpoint, filename=os.path.join(config.DRIVE_PATH, f"{config.BACKBONE}/{config.TOTAL_PATH}/model/{best_epoch}_YOLO_best.pth.tar"))

                    
           
        

        #Train network
        torch.cuda.empty_cache()

        #Demarcation
        print(f"time: {time.time() - start_time}")
        print("----------------------------------------")
        print("----------------------------------------")

    print(f"Total_Time: {time.time() - start_full_time}")   
    
    # Print main characteristics of the best model
    load_checkpoint(torch.load(os.path.join(os.path.join(config.DRIVE_PATH, f"{config.BACKBONE}/{config.TOTAL_PATH}/model/{best_epoch}_YOLO_best.pth.tar")), model, optimizer))
    # Print the number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    # Print the size of the model after training
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_bytes = buffer.tell()
    size_mb = size_bytes / (1024 ** 2)
    print(f'{config.BACKBONE} | Parameters: {total_params} | Size: {size_mb:.2f} MB')    
    print(f'Best epoch: {best_epoch} | Best mAP: {best_mAP:.4f}')

if __name__ == "__main__":
    main()