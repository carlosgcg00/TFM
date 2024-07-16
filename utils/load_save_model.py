import os
import sys
import torch
import torch.optim as optim
import re
import sys

# Custom imports
sys.path.append(os.getcwd())
import config



def charge_model():
    """
    Load the specified model and initialize it.

    :return: Initialized model and optimizer
    """
     

    if config.BACKBONE == 'resnet50':
        from models.resnet50_YOLO import resnet50
        model = resnet50(split_size=config.SPLIT_SIZE, num_boxes=config.NUM_BOXES, num_classes=config.NUM_CLASSES, pretrained=True).to(config.DEVICE)
    elif config.BACKBONE == 'vgg16':
        from models.vgg16_YOLO import vgg16
        model = vgg16(split_size=config.SPLIT_SIZE, num_boxes=config.NUM_BOXES, num_classes=config.NUM_CLASSES, pretrained=True).to(config.DEVICE)
    elif config.BACKBONE == 'efficientnet':
        from models.efficientnet_b0_YOLO import efficientnet_b0
        model = efficientnet_b0(split_size=config.SPLIT_SIZE, num_boxes=config.NUM_BOXES, num_classes=config.NUM_CLASSES).to(config.DEVICE)
    elif config.BACKBONE == 'tinyissimoYOLO':
        from models.tinyissimoYOLO import TinyissimoYOLO
        model = TinyissimoYOLO(split_size=config.SPLIT_SIZE, num_boxes=config.NUM_BOXES, num_classes=config.NUM_CLASSES).to(config.DEVICE)
    elif config.BACKBONE == 'bed_model':
        from models.bed_YOLO import bedmodel
        model = bedmodel(split_size=config.SPLIT_SIZE, num_boxes=config.NUM_BOXES, num_classes=config.NUM_CLASSES).to(config.DEVICE)
    elif config.BACKBONE == 'Yolov1':
        from models.YOLOv1 import Yolov1
        model = Yolov1(split_size=config.SPLIT_SIZE, num_boxes=config.NUM_BOXES, num_classes=config.NUM_CLASSES).to(config.DEVICE)
    else:
        raise ValueError(f"Unsupported backbone model: {config.BACKBONE}")

    if config.OPTIMIZER == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config.INIT_lr, weight_decay=config.WEIGHT_DECAY, momentum=0.7)
    elif config.OPTIMIZER == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config.INIT_lr, weight_decay=config.WEIGHT_DECAY)
    elif config.OPTIMIZER == 'NAdam':
        optimizer = optim.NAdam(model.parameters(), lr=config.INIT_lr, weight_decay=config.WEIGHT_DECAY)
    else:
        raise ValueError(f"Unsupported optimizer: {config.OPTIMIZER}")

    return model, optimizer

def find_the_best_model(folder_path):
    """
    Find the best model based on the highest index in the filename.

    :param folder_path: Path to the folder containing model checkpoints
    :return: Filename of the best model and its index
    """
    pattern = re.compile(r'(\d+)_YOLO_best\.pth\.tar$')
    
    largest_index = -1
    largest_file = None
    
    for filename in os.listdir(folder_path):
        match = pattern.search(filename)
        if match:
            index = int(match.group(1))
            if index > largest_index:
                largest_index = index
                largest_file = filename
    
    return largest_file, largest_index

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """
    Save the model checkpoint.

    :param state: State dictionary to save
    :param filename: Path to the file where the state will be saved
    """
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    """
    Load the model checkpoint.

    :param checkpoint: Path to the checkpoint file
    :param model: Model to load the state into
    :param optimizer: Optimizer to load the state into
    """
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

def load_model(folder_model=None):
    """
    Load the best model and its optimizer state.

    :param folder_model: Folder containing the model (if quantized)
    :param model_name: Name of the model to load
    :return: Loaded model and optimizer
    """
    model, optimizer = charge_model()

    for param in model.parameters():
        param.requires_grad = True

    if folder_model is None:
        best_model, epoch_best_model = find_the_best_model(os.path.join(config.ROOT_DIR, f'{config.BACKBONE}/{config.TOTAL_PATH}/model'))
        if best_model is None:
            raise FileNotFoundError("No model checkpoint found")
        load_checkpoint(torch.load(os.path.join(config.ROOT_DIR, f'{config.BACKBONE}/{config.TOTAL_PATH}/model/{best_model}')), model, optimizer)
        print(f"Model loaded: {config.BACKBONE}/{config.TOTAL_PATH}/model/{best_model}")
    else:
        model_path = os.path.join(config.ROOT_DIR, f'{config.BACKBONE}/{config.TOTAL_PATH}/model_opt/{folder_model}/YOLO_opt.pth.tar')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} does not exist")
        load_checkpoint(torch.load(model_path), model, optimizer)
        print(f"Model loaded: {config.BACKBONE}/{config.TOTAL_PATH}/model_opt/{folder_model}/YOLO_opt.pth.tar")

    model.eval()
    return model, optimizer
