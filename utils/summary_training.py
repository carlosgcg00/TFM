import os
import sys

# Add config path to the system path
# Custom imports
sys.path.append(os.getcwd())
import config

def save_summary():
    """
    Save a summary of the model configuration and dataset statistics to a file.
    """
    try:
        # Count the number of images in each dataset split
        num_train_images = len(os.listdir(os.path.join(config.IMG_DIR, "train")))
        num_val_images = len(os.listdir(os.path.join(config.IMG_DIR, "valid")))
        num_test_images = len(os.listdir(os.path.join(config.IMG_DIR, "test")))
        total_images = num_train_images + num_val_images + num_test_images

        # Collect data for the summary
        summary = f"""
        ######################################
        #              DATASET               #
        ######################################
        Dataset: {config.DATASET}
        Number of Classes: {config.NUM_CLASSES}
        Number of Train Images: {num_train_images} ({num_train_images/total_images*100:.2f}%)
        Number of Validation Images: {num_val_images} ({num_val_images/total_images*100:.2f}%)
        Number of Test Images: {num_test_images} ({num_test_images/total_images*100:.2f}%)

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

        # Create the directory if it does not exist
        summary_dir = os.path.join(config.ROOT_DIR, f"{config.BACKBONE}/{config.TOTAL_PATH}")
        os.makedirs(summary_dir, exist_ok=True)

        # Path to save the summary file
        summary_path = os.path.join(summary_dir, "model_summary.txt")

        # Write the summary to the file
        with open(summary_path, "w") as file:
            file.write(summary.strip())

        # Print a brief and formatted summary to the console
        print("\nModel Configuration Summary:")
        print(f"  Dataset: {config.DATASET}")
        print(f"  Backbone: {config.BACKBONE} ({'Pretrained' if config.PRETRAINED else 'Not Pretrained'})")
        print(f"  Batch Size: {config.BATCH_SIZE}")
        print(f"  Optimizer: {config.OPTIMIZER}")
        print(f"  Learning Rate Scheduler: {config.LR_SCHEDULER}")
        print(f"  Number of Classes: {config.NUM_CLASSES}")
        print(f"  Image Size: {config.IMAGE_SIZE}")
        print(f"Summary saved to {summary_path}")

    except Exception as e:
        print(f"An error occurred while saving the summary: {e}")

if __name__ == "__main__":
    save_summary()
