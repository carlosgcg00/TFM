# Import dependencies
import torch
import torchvision
import torch.nn as nn
import os
import sys
import io

# Custom imports
# Custom imports
sys.path.append(os.getcwd())
import config
'''
Build a YOLO v1 model using a pre-trained EfficientNet model from torchvision.
'''


def efficientnet_b0(split_size, num_boxes, num_classes, pretrained=True):
    """
    Modifies a pre-trained EfficientNet B0 model to build a variant of the YOLO v1 model.

    :param split_size: Grid size for object detection (S)
    :param num_boxes: Number of bounding boxes per grid cell (B)
    :param num_classes: Number of object classes (C)
    :param pretrained: Boolean indicating whether to use a pre-trained EfficientNet B0 model
    :return: Modified model for YOLO v1
    """
    S, B, C = split_size, num_boxes, num_classes

    # Import EfficientNet B0 from torchvision
    if pretrained:
        model = torchvision.models.efficientnet_b0(pretrained=True)
  
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
  
    # Remove FC layers
    modules = list(model.children())[:-2]
    model = nn.Sequential(*modules)

    # Add custom Adaptive Average Pooling
    model.avgpool = nn.AdaptiveAvgPool2d((split_size, split_size))

    # Calculate the input features to the first fully connected layer
    num_features = 1280 * S * S  # 1280 is the number of output channels before the classifier

    # Flatten the output and add two fully connected layers
    model.fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(num_features, 496),
        nn.Dropout(0.5),
        nn.LeakyReLU(0.1),
        nn.Linear(496, S * S * (C + B * 5)),
    )

    return model

def print_info(S=7, B=2, C=1):
    """
    Prints model summary and logs model architecture to TensorBoard.

    :param S: Grid size for object detection (default is 7)
    :param B: Number of bounding boxes per grid cell (default is 2)
    :param C: Number of object classes (default is 1)
    """
    batch_size = 2
    num_channels = 3  # RGB
    from torchsummary import summary
    from torch.utils.tensorboard import SummaryWriter
    import shutil

    # Remove previous TensorBoard logs
    log_dir = os.path.join(config.ROOT_DIR,'Architecture/efficientnet_b0')
    if shutil.os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir)
    
    # Initialize model
    model = efficientnet_b0(split_size=S, num_boxes=B, num_classes=C).to(config.DEVICE)
    size_img = 224

    # Input tensor
    x = torch.randn((batch_size, num_channels, size_img, size_img)).to(config.DEVICE)
    writer.add_graph(model, x, use_strict_trace=False)

    # Print model summary
    print(summary(model, (num_channels, size_img, size_img)))
    print(model(x).shape)
    writer.close()

# Test case
def test(S=7, B=2, C=20):
    """
    Runs a test case to validate the modified EfficientNet B0 model for YOLO v1.

    :param S: Grid size for object detection (default is 7)
    :param B: Number of bounding boxes per grid cell (default is 2)
    :param C: Number of object classes (default is 20)
    """
    batch_size = 2
    num_channels = 3  # RGB
    size_img = 448
    x_YOLO = torch.randn((batch_size, num_channels, size_img, size_img))
    x_efficientnet = torch.randn((batch_size, num_channels, 224, 224))

    from torch.utils.tensorboard import SummaryWriter    

    # EfficientNet B0
    model_efficientnet = efficientnet_b0(split_size=S, num_boxes=B, num_classes=C)
    writer = SummaryWriter(os.path.join(config.ROOT_DIR, 'Architecture/efficientnet_b0'))
    writer.add_graph(model_efficientnet, x_efficientnet, use_strict_trace=False)

    # Calculate total number of parameters
    total_params = sum(p.numel() for p in model_efficientnet.parameters())
    print(f"Total parameters in EfficientNet model: {total_params}")

    # Save the model to a buffer instead of a file on disk
    buffer = io.BytesIO()
    torch.save(model_efficientnet.state_dict(), buffer)
    size_bytes = buffer.tell()
    size_mb = size_bytes / (1024 ** 2)
    print(f'Model size: {size_mb:.2f} MB')

if __name__ == "__main__":
    test()
