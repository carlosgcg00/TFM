import torch
import torch.nn as nn
import os
import sys
import io

# Custom imports
sys.path.append(os.getcwd())
import config

'''
Adapt BED model to YOLO  model.
oBject detection model for Edge Devices.
Link paper: https://ar5iv.labs.arxiv.org/html/2202.07503
'''
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



class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, initialize=True, **kwargs):
        """
        Initializes a convolutional block with Conv2D, BatchNorm, and LeakyReLU layers.

        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param initialize: Boolean indicating whether to initialize weights
        :param kwargs: Additional arguments for Conv2D layer
        """
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)
        if initialize:
            self.initialize_weights()

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)

class BedModel(nn.Module):
    def __init__(self, in_channels=3, initialize=False, **kwargs):
        """
        Initializes the custom YOLO v1-like model with convolutional and fully connected layers.

        :param in_channels: Number of input channels (default is 3 for RGB)
        :param initialize: Boolean indicating whether to initialize weights
        :param kwargs: Additional arguments for the fully connected layers
        """
        super(BedModel, self).__init__()
        self.architecture = architecture_bed
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        x = torch.flatten(x, start_dim=1)
        return self.fcs(x)
    
    def _create_conv_layers(self, architecture):
        """
        Creates convolutional layers based on the provided architecture.

        :param architecture: List of tuples and lists defining the architecture
        :return: nn.Sequential model containing convolutional layers
        """
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:  # Tuple: (kernel_size, num_filters, stride, padding)
                layers += [CNNBlock(in_channels, out_channels=x[1], kernel_size=x[0], stride=x[2], padding=x[3])]
                in_channels = x[1]
            elif type(x) == str:  # Maxpooling layer "M"
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(in_channels, conv1[1], kernel_size=conv1[0], stride=conv1[2], padding=conv1[3]),
                        CNNBlock(conv1[1], conv2[1], kernel_size=conv2[0], stride=conv2[2], padding=conv2[3])
                    ]
                    in_channels = conv2[1]

        return nn.Sequential(*layers)
                
    def _create_fcs(self, split_size, num_boxes, num_classes):
        """
        Creates fully connected layers for the model.

        :param split_size: Grid size for object detection
        :param num_boxes: Number of bounding boxes per grid cell
        :param num_classes: Number of object classes
        :return: nn.Sequential model containing fully connected layers
        """
        S, B, C = split_size, num_boxes, num_classes
        num_input_features = 15 * S * S

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_input_features, 256),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(256, S * S * (C + B * 5)),
        )    

def print_info(S=7, B=2, C=20):
    """
    Prints model summary and logs model architecture to TensorBoard.

    :param S: Grid size for object detection (default is 7)
    :param B: Number of bounding boxes per grid cell (default is 2)
    :param C: Number of object classes (default is 20)
    :param return_model: Boolean indicating whether to return the model instance
    """
    batch_size = 2
    num_channels = 3  # RGB
    from torchsummary import summary
    from torch.utils.tensorboard import SummaryWriter
    import shutil

    # Remove previous TensorBoard logs
    log_dir = os.path.join(config.ROOT_DIR,f'Architecture/bed_YOLO')
    if shutil.os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir)
    
    # Initialize model
    model = BedModel(split_size=S, num_boxes=B, num_classes=C).to(config.DEVICE)
    size_img = 224
    
    # Input tensor
    x = torch.randn((batch_size, num_channels, size_img, size_img)).to(config.DEVICE)
    writer.add_graph(model, x, use_strict_trace=False)
    writer.close()

    print(summary(model, (num_channels, size_img, size_img)))

# Test Case
def test(S=7, B=2, C=3):
    """
    Runs a test case to validate the custom YOLO v1-like model.

    :param S: Grid size for object detection (default is 7)
    :param B: Number of bounding boxes per grid cell (default is 2)
    :param C: Number of object classes (default is 3)
    """
    from torch.utils.tensorboard import SummaryWriter    
    from torchsummary import summary

    writer = SummaryWriter(os.path.join(config.ROOT_DIR, 'Architecture/bed_YOLO'))

    # Specify the key argument
    model = BedModel(split_size=S, num_boxes=B, num_classes=C)
    batch_size = 2
    num_channels = 3  # RGB
    size_img = 224
    x = torch.randn((batch_size, num_channels, size_img, size_img))

    print(summary(model, (num_channels, size_img, size_img)))

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters in the BedModel: {total_params}")
    
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)

    # Calculate the size in bytes and convert to megabytes
    size_bytes = buffer.tell()
    size_mb = size_bytes / (1024 ** 2)

    print(f'Model size: {size_mb:.2f} MB')
    writer.add_graph(model, x, use_strict_trace=False)

    m = model(x)
    print(m.shape)  # (2, 1470) = (batch_size, S*S*(C+B*5))

if __name__ == "__main__":
    test()
