import cv2
import torch
import os
import datatrans
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
import warnings

warnings.filterwarnings("ignore", message="torch.utils._pytree._register_pytree_node is deprecated")
# Filtra advertencias de deprecación específicas
warnings.filterwarnings("ignore", category=UserWarning, module='transformers.utils.generic')
warnings.filterwarnings("ignore", category=UserWarning, module='torchvision.models._utils')

# Filtra todas las advertencias de deprecación (no recomendado)
warnings.filterwarnings("ignore", category=DeprecationWarning)

######################################
#              DATASET               #
######################################
# DRIVE_PATH = "/home/jovyan/Carlos_Gonzalez/YOLO"
DRIVE_PATH = os.getcwd()
DATASET = 'Airbus_256'  # PASCAL (20 classes)
                    # WiderFace (1 class) 
                    # PASCAL_3classes (3 classes) (for TinyisssimoYOLO like in the paper)
                    # reducePASCAL (1 class) (To force overfitting)
                    # Airbus_512 (1 class)
                    # Airbus_256 (1 class) (256x256 images)
                    # reduceAirbus (1 class) (To force overfitting)
IMG_DIR = os.path.join(DRIVE_PATH, 'Datasets',DATASET + "/images")
LABEL_DIR = os.path.join(DRIVE_PATH, 'Datasets', DATASET + "/labels")

######################################
#           HYPERPARAMETER           #
######################################
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32 # 64
WEIGHT_DECAY = 5e-4 # 5e-4 in original paper
L1_LAMBDA = 0.001 # To avoid overfitting
EPOCHS = 40
NUM_WORKERS = 4
OPTIMIZER = 'Adam'  # 'Adam'
                    # 'SGD'
                    # 'NAdam'

# LR_SCHEDULER = 'CosineAnnealingLR'  # 'ReduceLROnPlateau'
LR_SCHEDULER = 'None'  # 'ReduceLROnPlateau'
                                        # 'StepLR'
                                        # 'CosineAnnealingLR'
                                        # 'ExponentialLR'
                                        # 'LearningRateScheduler'
                                        # 'SchedulerNoPretrainedModels'
                                        # 'None' (No scheduler)

######################################
#                MODEL               #
######################################
SAVE_MODEL = True #True
LOAD_MODEL = False

######################################
#              BACKBONE              #
######################################
BACKBONE = 'tinyissimoYOLO'  # 'resnet50'
                    # 'vgg16'
                    # 'efficientnet'
                    # 'tinyissimoYOLO'
                    # 'ext_tinyissimoYOLO'
                    # 'bedmodel'
                    # 'Yolov1'

if BACKBONE == 'resnet50' or BACKBONE == 'vgg16'  or BACKBONE == 'efficientnet':
    PRETRAINED = True
    # warm up if the model is pretrained
    WARM_UP = 5
else:
    PRETRAINED = False
    # warm up if the model is no pretrained
    WARM_UP = 10
    
######################################
#                MODEL               #
######################################
# S: Split size, B: Number of boxes, C: Number of classes
# Select the number of classes based on the dataset
if DATASET == 'PASCAL':
    NUM_CLASSES = 20
elif DATASET == 'WiderFace' or DATASET == 'Airbus' or DATASET == 'reduceAirbus' or DATASET == 'Airbus_256':
    NUM_CLASSES = 1
elif DATASET == 'PASCAL_3classes' or DATASET == 'reducePASCAL':
    NUM_CLASSES = 3

# Select the learning rate based on the learning rate scheduler
if LR_SCHEDULER == 'ReduceLROnPlateau' or LR_SCHEDULER == 'StepLR' or LR_SCHEDULER == 'CosineAnnealingLR' or LR_SCHEDULER == 'ExponentialLR':
    INIT_lr = 2e-5
    BASE_lr = 5e-4
elif LR_SCHEDULER == 'LearningRateScheduler':
    INIT_lr = 2e-4
    BASE_lr = 8e-4
elif LR_SCHEDULER == 'SchedulerNoPretrainedModels':
    INIT_lr = 2e-7
    BASE_lr = 5e-4
elif LR_SCHEDULER == 'None':
    INIT_lr = 5e-4
    BASE_lr = 5e-4

if DATASET == 'reducePASCAL' or DATASET == 'reduceAirbus':
    BATCH_SIZE = 1

# Select the number of classes based on the backbone
if BACKBONE == 'resnet50' or BACKBONE == 'vgg16' or BACKBONE == 'Yolov1':
    IMAGE_SIZE = 448 # 448 for resnet50, 448 for vgg16, 224 for efficientnet
elif BACKBONE == 'efficientnet' or BACKBONE == 'bedmodel':
    IMAGE_SIZE = 224
elif BACKBONE == 'tinyissimoYOLO':
    IMAGE_SIZE = 88
elif BACKBONE == 'ext_tinyissimoYOLO':
    IMAGE_SIZE = 352


# Select the split size and number of boxes based on the backbone
if BACKBONE == 'tinyissimoYOLO' or BACKBONE == 'ext_tinyissimoYOLO':
    SPLIT_SIZE = 4
    NUM_BOXES = 2
else:
    SPLIT_SIZE = 7
    NUM_BOXES = 2

######################################
#                PATH                #
######################################
TOTAL_PATH = f"{OPTIMIZER}_{LR_SCHEDULER}_{DATASET}_BATCH_{BATCH_SIZE}_LR_{INIT_lr}"

######################################
#           TRANSFORMATIONS          #
######################################

train_transforms = datatrans.Compose([
    datatrans.Resize(size=(IMAGE_SIZE, IMAGE_SIZE)),
    # v2.RandomResizedCrop(size=(IMAGE_SIZE, IMAGE_SIZE), antialias=True),
    datatrans.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.4, hue=0.1, p=0.3),
    datatrans.RandomHorizontalFlip(p=0.5),
    datatrans.RandomVerticalFlip(p=0.3),
    datatrans.GaussianBlur(p=0.2),
    datatrans.toTensor(),
])

valid_transforms = datatrans.Compose([
    datatrans.Resize(size=(IMAGE_SIZE, IMAGE_SIZE)),
    datatrans.toTensor(),    
])

predict_transforms = datatrans.Compose([
    datatrans.Resize(size=(IMAGE_SIZE, IMAGE_SIZE)),
    datatrans.toTensor(),    
])


test_transform = datatrans.Compose([
    datatrans.toTensor()
])


    


# This is only for the convolutional layers
architecture_tinyissimo_YOLO = [
    # Tuple: (kernel_size, num_filters, stride, padding)
    # ReLU activation is assumed after each convolutional layer
    [(3,16,1,1),2],
    "M",            # Max pooling with 2x2 filters, stride 2 (implied)
    (3, 16, 1, 1),  # Another conv. layer with 3x3 filters, 16 filters, stride 1, padding 1
    (3, 32, 1, 1),  # Another conv. layer with 3x3 filters, 16 filters, stride 1, padding 1
    "M",            # Max pooling again
    (3, 32, 1, 1),  # Conv. layer, 3x3 filters, 32 filters, stride 1, padding 1
    (3, 64, 1, 1),  # Conv. layer, 3x3 filters, 32 filters, stride 1, padding 1
    "M",            # Max pooling
    [(3, 64, 1, 1),2],  # Conv. layer, 3x3 filters, 64 filters, stride 1, padding 1
    "M",            # Max pooling
    (3, 128, 1, 1), # Conv. layer, 3x3 filters, 128 filters, stride 1, padding 1
    (3, 128, 1, 3), # Conv. layer, 3x3 filters, 128 filters, stride 1, padding 1
    "M",            # Max pooling
]

# This is only for the convolutional layers
architecture_ext_tinyissimo_YOLO = [
    # Tuple: (kernel_size, num_filters, stride, padding)
    # ReLU activation is assumed after each convolutional layer
    (3,4,1,1),     # Conv. layer with 3x3 filters, 4 filters, stride 1, padding 1
    (3,4,1,1),     # Conv. layer with 3x3 filters, 4 filters, stride 1, padding 1
    "M",            # Max pooling with 2x2 filters, stride 2 (implied)
    (3, 4, 1, 1),  # Another conv. layer with 3x3 filters, 8 filters, stride 1, padding 1
    (3, 8, 1, 1),  # Another conv. layer with 3x3 filters, 16 filters, stride 1, padding 1
    "M",            # Max pooling with 2x2 filters, stride 2 (implied)
    (3, 8, 1, 1),  # Another conv. layer with 3x3 filters, 8 filters, stride 1, padding 1
    (3, 16, 1, 1),  # Another conv. layer with 3x3 filters, 16 filters, stride 1, padding 1
    "M",            # Max pooling again
    (3, 16, 1, 1),  # Another conv. layer with 3x3 filters, 16 filters, stride 1, padding 1
    (3, 32, 1, 1),  # Another conv. layer with 3x3 filters, 16 filters, stride 1, padding 1
    "M",            # Max pooling again
    (3, 32, 1, 1),  # Conv. layer, 3x3 filters, 32 filters, stride 1, padding 1
    (3, 64, 1, 1),  # Conv. layer, 3x3 filters, 32 filters, stride 1, padding 1
    "M",            # Max pooling
    [(3, 64, 1, 1),2],  # Conv. layer, 3x3 filters, 64 filters, stride 1, padding 1
    "M",            # Max pooling
    (3, 128, 1, 1), # Conv. layer, 3x3 filters, 128 filters, stride 1, padding 1
    (3, 128, 1, 3), # Conv. layer, 3x3 filters, 128 filters, stride 1, padding 1
    "M",            # Max pooling
]


architecture_bed = [
    (3,64,1,1),       # Conv. 3x3, 64 filters
    "M",               # Max pooling
    (3,24,1,1),       # Conv. 3x3, 24 filters
    "M",               # Max pooling
    (3,16,1,1),       # Conv. 3x3, 24 filters
    (3,32,1,1),       # Conv. 3x3, 32 filters
    (3,32,1,1),       # Conv. 3x3, 32 filters
    (3,64,1,1),       # Conv. 3x3, 64 filters
    "M",               # Max pooling
    [(3, 32, 1, 1),(3, 64, 1, 1),3], # Conv. 3x3, 32 filters, Conv. 3x3, 64 filters, 3 times
    "M",               # Max pooling
    [(3, 32, 1, 1),(3, 64, 1, 1),2], # Conv. 3x3, 32 filters, Conv. 3x3, 64 filters, 3 times
    (3, 64, 1, 1),   # Conv. 3x3, 64 filters
    (3, 64, 1, 1),   # Conv. 3x3, 64 filters
    "M",               # Max pooling
    (3, 64, 1, 1),   # Conv. 3x3, 64 filters
    (3, 64, 1, 1),   # Conv. 3x3, 64 filters
    "M",               # Max pooling
    (1, 64, 1, 0),   # Conv. 1x1, 64 filters
    (1, 16, 1, 0),   # Conv. 1x1, 16 filters
    (1, 16, 1, 0),   # Conv. 1x1, 16 filters
    (1, 15, 1, 1),   # Conv. 1x1, 15 filters
    (1, 15, 1, 1),   # Conv. 1x1, 15 filters
]


architecture_YOLOv1 = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

AIRBUS_LABELS = ['Airplane']
PASCAL_LABELS = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
reduce_PASCAL_LABELS = ['car', 'person', 'chair']
colors = ['blue', 'green', 'red', 'yellow', 'purple', 'orange', 'cyan', 'magenta', 'lime', 'pink', 'teal', 'lavender', 'brown', 'beige', 'maroon', 'mint', 'olive', 'coral', 'navy', 'grey', 'white', 'black'] 
