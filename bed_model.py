import torch
import torch.nn as nn
import config
import io


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, initialize= True, **kwargs):
        # super() function is used to give access to methods and properties of a parent or sibling class.
        super(CNNBlock, self).__init__() # To initialize the parent class
        # nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding) 
        # Conv2d applies a 2D convolution over an input signal composed of several input planes.
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        # nn.BatchNorm2d(num_features)
        # BatchNorm2d applies Batch Normalization over a 4D input (a mini-batch of 2D inputs with additional channel dimension)
        self.batchnorm = nn.BatchNorm2d(out_channels) 
        # nn.LeakyReLU(negative_slope=0.01, inplace=False)
        # LeakyReLU applies the element-wise function Leaky ReLU(x) = max(0,    x) + negative_slope * min(0, x)
        self.leakyrelu = nn.LeakyReLU(0.1)
        # Forward pass of the CNNBlock
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
  
    
class bedmodel(nn.Module):
    def __init__(self, in_channels=3, initialize = False,**kwargs):
        # 3 channels due to the RGB images
        super(bedmodel, self).__init__()
        self.architecture = config.architecture_bed
        self.in_channels = in_channels
        # The CNN are darknet, is how is it call to theese CNN
        self.darknet = self._create_conv_layers(self.architecture)
        # Then we have the fully connected layers
        self.fcs = self._create_fcs(**kwargs)


    def forward(self, x):
        x = self.darknet(x)
        # We flatten the tensor to pass it to the fully connected layers
        # flatten is used to reshape the tensor
        # print(x.shape)
        x = torch.flatten(x, start_dim=1)
        # print(x.shape)
        return self.fcs(x)
    
    def _create_conv_layers(self, architecture):
        '''
        This function is going to create the convolutional layers
        self.architecture is a list of tuples and lists
        architecture is a list of tuples and lists
        '''
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple: # Tuple: (kernel_size, num_filters, stride, padding)
                layers += [
                    CNNBlock(
                    in_channels, out_channels=x[1], kernel_size=x[0], stride=x[2], padding=x[3]
                    )
                ]
                in_channels = x[1] # The in_channels for the next layer is the out_channels of the previous layer
            elif type(x) == str: # Maxpooling layer "M"
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    layers += [
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]
                    in_channels = conv2[1]

        return nn.Sequential(*layers)
                

    def _create_fcs(self, split_size, num_boxes, num_classes):
        '''
        This function is going to create the fully connected layers
        split_size is the size of the feature map
        num_boxes is the number of boxes per cell
        num_classes is the number of classes of the dataset
        '''
        S, B, C = split_size, num_boxes, num_classes
        
        num_input_features = 15 * S * S

        return nn.Sequential(
            nn.Flatten(), # Flatten the tensor to reshape it into a single vector per image in the batch
            nn.Linear(num_input_features, 256), 
            nn.Dropout(0.5), # Dropout with a probability of 0.5 to prevent overfitting
            nn.LeakyReLU(0.1), # LeakyReLU activation with a negative slope of 0.1
            nn.Linear(256, S * S * (C + B * 5)), # Output layer to predict the required tensor shape for bounding boxes and class probabilities
            # The output size S * S * (C + B * 5) represents each cell of the grid having predictions for num_classes and num_boxes each with 5 params (x, y, w, h, confidence)
        )    
    
def print_info(model_name = 'bed_model', S = 7, B = 2, C = 20, return_model = False):
    batch_size = 2
    num_channels = 3 # RGB
    from torchsummary import summary
    from torch.utils.tensorboard import SummaryWriter
    import shutil
    if shutil.os.path.exists(f'Architecture/{model_name}'):
        shutil.rmtree(f'Architecture/{model_name}')
    writer = SummaryWriter(f'Architecture/{model_name}')
    model = bedmodel(split_size=S, num_boxes=B, num_classes=C).to(config.DEVICE)
    size_img = 224
    
    # Input tensor
    x = torch.randn((batch_size, num_channels, size_img, size_img)).to(config.DEVICE)
    writer.add_graph(model, x, use_strict_trace=False)
    writer.close()

    print(summary(model, (num_channels, size_img, size_img)))


# ¡¡Test Case!!
def test(S = 7, B = 2, C = 3):
    from torch.utils.tensorboard import SummaryWriter    
    from torchsummary import summary
    writer = SummaryWriter('Architecture/bedmodel')
    # Specify the keyargument
    model = bedmodel(split_size=S, num_boxes=B, num_classes=C)
    batch_size = 2
    num_channels = 3 # RGB
    size_img = 224
    x = torch.randn((batch_size, num_channels, size_img, size_img))
    # print(model)
    print(summary(model, (num_channels, size_img, size_img)))


    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total de parámetros en el modelo Tinyissimo: {total_params}")
    
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)

    # Calcular el tamaño en bytes y convertir a megabytes
    size_bytes = buffer.tell()
    size_mb = size_bytes / (1024 ** 2)

    print(f'Tamaño del modelo Tinyissimo: {size_mb:.2f} MB')
    writer.add_graph(model, x, use_strict_trace=False)

    m = model(x)
    
    print(m.shape) # (2, 1470) = (batch_size, S*S*(C+B*5))

if __name__ == "__main__":
    test()
