import torch
import torch.nn as nn
import config
import io


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, initialize= False, **kwargs):
        # super() function is used to give access to methods and properties of a parent or sibling class.
        super(CNNBlock, self).__init__() # To initialize the parent class
        # nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding) 
        # Conv2d applies a 2D convolution over an input signal composed of several input planes.
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        # nn.BatchNorm2d(num_features)
        # BatchNorm2d applies Batch Normalization over a 4D input (a mini-batch of 2D inputs with additional channel dimension)
        self.batchnorm = nn.BatchNorm2d(out_channels) 
        # nn.LeakyReLU(negative_slope=0.01, inplace=False)
        # LeakyReLU applies the element-wise function Leaky ReLU(x) = max(0, x) + negative_slope * min(0, x)
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
  
    
class tinyissimoYOLO(nn.Module):
    def __init__(self, in_channels=3, initialize = False,**kwargs):
        # 3 channels due to the RGB images
        super(tinyissimoYOLO, self).__init__()
        self.architecture = config.architecture_tinyissimo_YOLO
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
            elif type(x) == list: # List: [(kernel_size, num_filters, stride, padding), (kernel_size, num_filters, stride, padding), num_repeats]
                conv1 = x[0] # tupple
                num_repeats = x[1] # Integer

                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            out_channels=conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    # the in_channels for the next layer is the out_channels of the second convolutional layer
                    in_channels = conv1[1] # como esto se ejecuta num_repeat veces, el in_channels de la siguiente iteracion/capa es el out_channels de la segunda capa convolucional de esta iteracion/capa
        return nn.Sequential(*layers) # It is going to unpack the list of layers and pass them as arguments to the nn.Sequential
                

    def _create_fcs(self, split_size, num_boxes, num_classes):
        '''
        This function is going to create the fully connected layers
        split_size is the size of the feature map
        num_boxes is the number of boxes per cell
        num_classes is the number of classes of the dataset
        '''
        S, B, C = split_size, num_boxes, num_classes
        
        return nn.Sequential(
            nn.Flatten(), # It is going to flatten the tensor, i.e., it is going to reshape the tensor
            nn.Linear(128 * S * S, 256), 
            nn.Dropout(0.55), # toDO Dropout is used to prevent overfitting, we should put here p = 0.5
            nn.LeakyReLU(0.1), # LeakyReLU applies the element-wise function Leaky ReLU(x) = max(0, x) + negative_slope * min(0, x)
            nn.Linear(256, S * S * (C + B * 5)), # S * S * (C + B * 5) is the number of nodes in the fully connected layer
            # S * S is the size of the feature map, C is the number of classes, B is the number of boxes and 5 is the number of parameters for each box
            # (S,S,30) where C+B*5 = 30
        )    
    
    
def print_info(model_name = 'tinyissimoYOLO', S = 4, B = 2, C = 20, return_model = False):
    batch_size = 2
    num_channels = 3 # RGB
    from torchsummary import summary
    from torch.utils.tensorboard import SummaryWriter
    import shutil
    if shutil.os.path.exists(f'Architecture/{model_name}'):
        shutil.rmtree(f'Architecture/{model_name}')    
    writer = SummaryWriter(f'Architecture/{model_name}')
    model = tinyissimoYOLO(split_size=S, num_boxes=B, num_classes=C).to(config.DEVICE)
    size_img = 88
    
    # Input tensor
    x = torch.randn((batch_size, num_channels, size_img, size_img)).to(config.DEVICE)
    writer.add_graph(model, x, use_strict_trace=False)
    
    print(summary(model, (num_channels, size_img, size_img)))
    wirter.close()

# ¡¡Test Case!!
def test(S = 4, B = 2, C = 3):
    from torch.utils.tensorboard import SummaryWriter    
    writer = SummaryWriter('Architecture/TinyissimoYOLO_1')
    # Specify the keyargument
    model = tinyissimoYOLO(split_size=S, num_boxes=B, num_classes=C)
    batch_size = 2
    num_channels = 3 # RGB
    size_img = 88
    x = torch.randn((batch_size, num_channels, size_img, size_img))
    print(model)

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
    print_info()
