import os
import sys
import torch
import torch.optim as optim
from tqdm import tqdm
import io
import time
import shutil
from torch.nn.utils import clip_grad_norm_

# Custom imports
sys.path.append(os.getcwd())
from utils.utils import (
    mean_average_precision,
    get_bboxes,
    get_loaders,
    warmup_scheduler,
    update_lr,
    update_lr_no_pretrained,
    get_lr,
)
from utils.load_save_model import (
    save_checkpoint,
    load_checkpoint,
    charge_model
)
from utils.save_results import (
    save_results,
    save_log,
    plot_loss,
    plot_results,
    read_last_line
)
from utils.summary_training import save_summary
import config
from YOLO_functions.loss import YoloLoss

# Set random seed for reproducibility
seed = 123
torch.manual_seed(seed)

def train_fn(train_loader, model, optimizer, scheduler, loss_fn, initial_epoch, epoch):
    """
    Train the model for one epoch.
    """
    loop = tqdm(train_loader, leave=True)
    mean_loss, mean_loss_box_coordinates, mean_loss_object_loss, mean_loss_no_object_loss, mean_loss_class = [], [], [], [], []
    loop.set_description(f"Epoch [{epoch}/{config.EPOCHS + initial_epoch - 1}]")

    for batch_idx, (x, y) in enumerate(loop):
        # Update learning rate
        if config.LR_SCHEDULER != 'None':
            if config.LR_SCHEDULER == 'LearningRateScheduler':
                update_lr(optimizer, epoch)
            if (config.PRETRAINED or not config.PRETRAINED) and epoch < config.WARM_UP:
                warmup_scheduler(optimizer, epoch)
            if config.LR_SCHEDULER == 'SchedulerNoPretrainedModels':
                update_lr_no_pretrained(optimizer, epoch)
            lr = get_lr(optimizer)
        else:
            lr = optimizer.param_groups[0]['lr']

        # Move data to the configured device
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)

        # Forward pass and loss calculation
        out = model(x)
        loss, loss_box_coordinates, loss_object_loss, loss_no_object_loss, loss_class = loss_fn(out, y)
        
        # L1 regularization (Lasso)
        l1_lambda = config.L1_LAMBDA
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        loss += l1_lambda * l1_norm

        # Record losses
        mean_loss.append(loss.item())
        mean_loss_box_coordinates.append(loss_box_coordinates.item())
        mean_loss_object_loss.append(loss_object_loss.item())
        mean_loss_no_object_loss.append(loss_no_object_loss.item())
        mean_loss_class.append(loss_class.item())
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Update progress bar
        loop.set_postfix(loss=loss.item())

    # Update the learning rate scheduler if applicable
    if config.LR_SCHEDULER not in ['None', 'LearningRateScheduler', 'SchedulerNoPretrainedModels'] and epoch >= config.WARM_UP:
        scheduler.step(sum(mean_loss) / len(mean_loss))

    # Calculate mean losses
    mean_loss_out = sum(mean_loss) / len(mean_loss)
    mean_loss_box_coordinates_out = sum(mean_loss_box_coordinates) / len(mean_loss_box_coordinates)
    mean_loss_object_loss_out = sum(mean_loss_object_loss) / len(mean_loss_object_loss)
    mean_loss_no_object_loss_out = sum(mean_loss_no_object_loss) / len(mean_loss_no_object_loss)
    mean_loss_class_out = sum(mean_loss_class) / len(mean_loss_class)

    # Print loss summary
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
          f'{mean_loss_no_object_loss_out:.3f}'.ljust(12) + "|" +
          f'{mean_loss_class_out:.3f}'.ljust(12) + "|")  
    print("---------------------------------------------------")

    save_log(epoch, mean_loss_out, mean_loss_box_coordinates_out, mean_loss_object_loss_out, mean_loss_no_object_loss_out, mean_loss_class_out, mode='train')
    plot_loss()
    return lr

def main():
    """
    Main function to train and evaluate the model.
    """
    save_summary()
    model, optimizer = charge_model()

    # Print the number of parameters and size of the model
    total_params = sum(p.numel() for p in model.parameters())
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_bytes = buffer.tell()
    size_mb = size_bytes / (1024 ** 2)
    print(f'{config.BACKBONE} | Parameters: {total_params} | Size: {size_mb:.2f} MB')

    # Define the scheduler
    scheduler = None
    if config.LR_SCHEDULER == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5)
    elif config.LR_SCHEDULER == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0.00005)
    elif config.LR_SCHEDULER == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    elif config.LR_SCHEDULER == 'ExponentialLR':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # Define the loss function
    loss_fn = YoloLoss(S=config.SPLIT_SIZE, B=config.NUM_BOXES, C=config.NUM_CLASSES)

    # Load data into data loader
    train_loader, val_loader, train_eval_loader = get_loaders()

    # Load the model and the results
    initial_epoch = 0
    if config.LOAD_MODEL:
        initial_epoch = read_last_line()
        initial_epoch = int(initial_epoch / 1) * 1
        load_checkpoint(torch.load(os.path.join(config.ROOT_DIR, f'{config.BACKBONE}/{config.TOTAL_PATH}/model/YOLO_epoch_{initial_epoch}.pth.tar')), model, optimizer)
        initial_epoch += 1
    else:
        for folder in ['results', 'model']:
            folder_path = os.path.join(config.ROOT_DIR, f"{config.BACKBONE}/{config.TOTAL_PATH}/{folder}")
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
            os.makedirs(folder_path, exist_ok=True)

    # Training loop
    best_mAP, best_epoch = 0, 0
    start_full_time = time.time()

    for epoch in range(initial_epoch, initial_epoch + config.EPOCHS):
        start_time = time.time()

        # Enable gradient calculation
        if epoch >= config.WARM_UP:
            for param in model.parameters():
                param.requires_grad = True

        # Training the model
        model.train()
        last_lr = train_fn(train_loader, model, optimizer, scheduler, loss_fn, initial_epoch, epoch)

        # Evaluate on the training set
        model.eval()
        pred_boxes, target_boxes, train_loss = get_bboxes(epoch, train_eval_loader, model, loss_fn, iou_threshold=0.5, threshold=0.5, device=config.DEVICE, mode='Train')
        train_mAP_50, train_mAP_75, train_mAP_90 = mean_average_precision(pred_boxes, target_boxes, iou_thresholds=[0.5, 0.75, 0.9], box_format="midpoint", num_classes=config.NUM_CLASSES, best_mAP=best_mAP, epoch=epoch, mode='Train')
        print(f"Train: \t mAP@50: {train_mAP_50:.6f}, mAP@75: {train_mAP_75:.6f}, mAP@90: {train_mAP_90:.6f}, Mean Loss: {train_loss:.6f}")
        save_results(epoch, train_loss, train_mAP_50, train_mAP_75, train_mAP_90, learning_rate=last_lr, mode='train')

        # Evaluate on the validation set
        model.eval()
        pred_boxes, target_boxes, val_loss = get_bboxes(epoch, val_loader, model, loss_fn, iou_threshold=0.5, threshold=0.5, device=config.DEVICE, mode='Valid')
        val_mAP_50, val_mAP_75, val_mAP_90 = mean_average_precision(pred_boxes, target_boxes, iou_thresholds=[0.5, 0.75, 0.9], box_format="midpoint", num_classes=config.NUM_CLASSES, best_mAP=best_mAP, epoch=epoch, mode='Valid')
        print(f"Valid: \t mAP@50: {val_mAP_50:.6f}, mAP@75: {val_mAP_75:.6f}, mAP@90: {val_mAP_90:.6f}, Mean Loss: {val_loss:.6f}")
        save_results(epoch, val_loss, val_mAP_50, val_mAP_75, val_mAP_90, learning_rate=last_lr, mode='valid')
        plot_results()

        torch.cuda.empty_cache()
        print('----------------------------------------')
        print(f"Epoch: {epoch} | Train mAP: {train_mAP_50:.6f} | Validation mAP: {val_mAP_50:.6f}")
        print('----------------------------------------')

        # Save the best model
        if config.SAVE_MODEL:
            if epoch % 1 == 0 and epoch > 0:
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                prev_model_path = os.path.join(config.ROOT_DIR, f"{config.BACKBONE}/{config.TOTAL_PATH}/model/YOLO_epoch_{epoch - 1}.pth.tar")
                if os.path.exists(prev_model_path):
                    os.remove(prev_model_path)
                save_checkpoint(checkpoint, filename=os.path.join(config.ROOT_DIR, f"{config.BACKBONE}/{config.TOTAL_PATH}/model/YOLO_epoch_{epoch}.pth.tar"))

            if epoch > 0 and val_mAP_50 > best_mAP and train_mAP_50 - val_mAP_50 < 0.15:
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                best_model_path = os.path.join(config.ROOT_DIR, f"{config.BACKBONE}/{config.TOTAL_PATH}/model/{best_epoch}_YOLO_best.pth.tar")
                if os.path.exists(best_model_path):
                    os.remove(best_model_path)
                best_mAP = val_mAP_50
                best_epoch = epoch
                save_checkpoint(checkpoint, filename=os.path.join(config.ROOT_DIR, f"{config.BACKBONE}/{config.TOTAL_PATH}/model/{best_epoch}_YOLO_best.pth.tar"))

        torch.cuda.empty_cache()
        print(f"Epoch {epoch} time: {time.time() - start_time}")
        print("----------------------------------------")
        print("----------------------------------------")

    print(f"Total training time: {time.time() - start_full_time}")

    # Print main characteristics of the best model
    load_checkpoint(torch.load(os.path.join(config.ROOT_DIR, f"{config.BACKBONE}/{config.TOTAL_PATH}/model/{best_epoch}_YOLO_best.pth.tar")), model, optimizer)
    total_params = sum(p.numel() for p in model.parameters())
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_bytes = buffer.tell()
    size_mb = size_bytes / (1024 ** 2)
    print(f'{config.BACKBONE} | Parameters: {total_params} | Size: {size_mb:.2f} MB')
    print(f'Best epoch: {best_epoch} | Best mAP: {best_mAP:.4f}')

if __name__ == "__main__":
    main()
