import torch
import torch.nn as nn
import io
import os
import sys

# Custom imports
sys.path.append(os.getcwd())
import config
'''
TinyissimoYOO model for object detection, reduce version of YOLOv1.
Link Paper: https://arxiv.org/pdf/2306.00001.pdf
'''
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


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, initialize=False, **kwargs):
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

class TinyissimoYOLO(nn.Module):
    def __init__(self, in_channels=3, initialize=False, **kwargs):
        """
        Initializes the custom YOLO v1-like model with convolutional and fully connected layers.

        :param in_channels: Number of input channels (default is 3 for RGB)
        :param initialize: Boolean indicating whether to initialize weights
        :param kwargs: Additional arguments for the fully connected layers
        """
        super(TinyissimoYOLO, self).__init__()
        self.architecture = architecture_tinyissimo_YOLO
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
            elif type(x) == list:  # List: [(kernel_size, num_filters, stride, padding), num_repeats]
                conv = x[0]
                num_repeats = x[1]

                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            out_channels=conv[1],
                            kernel_size=conv[0],
                            stride=conv[2],
                            padding=conv[3],
                        )
                    ]
                    in_channels = conv[1]

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

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * S * S, 256),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(256, S * S * (C + B * 5)),
        )

def print_info(S=4, B=2, C=20, return_model=False):
    """
    Prints model summary and logs model architecture to TensorBoard.

    :param model_name: Name of the model
    :param S: Grid size for object detection (default is 4)
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
    log_dir = os.path.join(config.ROOT_DIR,f'Architecture/TinyissimoYOLO')
    if shutil.os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir)

    # Initialize model
    model = TinyissimoYOLO(split_size=S, num_boxes=B, num_classes=C).to(config.DEVICE)
    size_img = 88

    # Input tensor
    x = torch.randn((batch_size, num_channels, size_img, size_img)).to(config.DEVICE)
    writer.add_graph(model, x, use_strict_trace=False)

    print(summary(model, (num_channels, size_img, size_img)))
    writer.close()

# Test Case
def test(S=4, B=2, C=3):
    """
    Runs a test case to validate the custom YOLO v1-like model.

    :param S: Grid size for object detection (default is 4)
    :param B: Number of bounding boxes per grid cell (default is 2)
    :param C: Number of object classes (default is 3)
    """
    from torch.utils.tensorboard import SummaryWriter    

    writer = SummaryWriter(os.path.join(config.ROOT_DIR, 'Architecture/TinyissimoYOLO_1'))
    model = TinyissimoYOLO(split_size=S, num_boxes=B, num_classes=C)
    batch_size = 2
    num_channels = 3  # RGB
    size_img = 88
    x = torch.randn((batch_size, num_channels, size_img, size_img))
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters in the TinyissimoYOLO model: {total_params}")
    
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)

    # Calculate the size in bytes and convert to megabytes
    size_bytes = buffer.tell()
    size_mb = size_bytes / (1024 ** 2)

    print(f'Model size: {size_mb:.2f} MB')
    writer.add_graph(model, x, use_strict_trace=False)

    m = model(x)
    print(m.shape)  # (2, S*S*(C+B*5)) = (batch_size, S*S*(C+B*5))

if __name__ == "__main__":
    print_info()
