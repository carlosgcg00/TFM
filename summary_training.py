import os
import config
from utils import extract_image_name
def save_summary():
    num_train_images = len(os.listdir(os.path.join(config.IMG_DIR, "train")))
    num_val_images = len(os.listdir(os.path.join(config.IMG_DIR, "valid")))
    num_test_images = len(os.listdir(os.path.join(config.IMG_DIR, "test")))
    total_images = num_train_images + num_val_images + num_test_images
      

    # Recolectando los datos para guardar
    summary = f"""
    ######################################
    #              DATASET               #
    ######################################
    Dataset: {config.DATASET}
    Number of Classes: {config.NUM_CLASSES}
    Number of Train Images: {num_train_images} {num_train_images/total_images*100:.2f}%
    Number of Validation Images: {num_val_images} {num_val_images/total_images*100:.2f}%
    Number of Test Images: {num_test_images} {num_test_images/total_images*100:.2f}%

    ######################################
    #                MODEL               #
    ######################################
    Save Model: {config.SAVE_MODEL}
    Load Model: {config.LOAD_MODEL}
    Model Backbone: {config.BACKBONE}
    Pretrained: {config.PRETRAINED}
    Image Size: {config.IMAGE_SIZE}
    Split Size: {config.SPLIT_SIZE}
    Number of Boxes: {config.NUM_BOXES}
    Number of Classes: {config.NUM_CLASSES}

    ######################################
    #           HYPERPARAMETER           #
    ######################################
    Device: {config.DEVICE}
    Batch Size: {config.BATCH_SIZE}
    Weight Decay: {config.WEIGHT_DECAY}
    L1 Lambda: {config.L1_LAMBDA}
    Epochs: {config.EPOCHS}
    Number of Workers: {config.NUM_WORKERS}
    Optimizer: {config.OPTIMIZER}
    Learning Rate Scheduler: {config.LR_SCHEDULER}
    Initial Learning Rate: {config.INIT_lr}
    Warm Up: {config.WARM_UP}
    """

    # first if the directory does not exist, create it
    os.makedirs(os.path.join(config.DRIVE_PATH, f"{config.BACKBONE}/{config.TOTAL_PATH}"), exist_ok=True)
    # Path to save the summary file
    summary_path = os.path.join(config.DRIVE_PATH, f"{config.BACKBONE}/{config.TOTAL_PATH}/model_summary.txt")

    # Writing to the file
    with open(summary_path, "w") as file:
        file.write(summary)

    # Printing a brief and formatted summary to the console
    print("\nModel Configuration Summary:")
    print(f"  Dataset: {config.DATASET}")
    print(f"  Backbone: {config.BACKBONE} ({'Pretrained' if config.PRETRAINED else 'Not Pretrained'})")
    print(f"  Batch Size: {config.BATCH_SIZE}")
    print(f"  Optimizer: {config.OPTIMIZER}")
    print(f"  Learning Rate Scheduler: {config.LR_SCHEDULER}")
    print(f"  Number of Classes: {config.NUM_CLASSES}")
    print(f"  Image Size: {config.IMAGE_SIZE}")

    print(f"Summary saved to {summary_path}")