#Import dependencies
import torch
import torchvision
import torch.nn as nn
import io
import config


def resnet50(split_size, num_boxes, num_classes, pretrained=True):
    S, B, C = split_size, num_boxes, num_classes

    #Import Resnet50 from pytorch
    model = torchvision.models.resnet50(pretrained = pretrained)
  
    #Enable backprop for all layers
    for param in model.parameters():
        param.requires_grad = False
  
    #Remove FC layers
    modules=list(model.children())[:-2]
    model=nn.Sequential(*modules)

    #Add custom CNN layer
    model.avgpool = nn.AdaptiveAvgPool2d((split_size,split_size))
    model = nn.Sequential(model, 
               nn.Conv2d(2048, 1024, kernel_size=(1,1)))

    #Add custom FC layers      
    model.fc=nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, S * S * (C + B * 5)),)

    return model


def vgg16(split_size, num_boxes, num_classes, pretrained=True):
    S, B, C = split_size, num_boxes, num_classes
    #Import Resnet50 from pytorch
    model = torchvision.models.vgg16(pretrained = pretrained)

    modules=list(model.children())[:-2]
    model=nn.Sequential(*modules)
    
    #Enable backprop for all layers
    for param in model.parameters():
        param.requires_grad = False

    #Add custom pooling layer
    model.avgpool = nn.AdaptiveAvgPool2d((split_size,split_size))

    #Add custom FC layers 
    model.fc=nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * S * S, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, S * S * (C + B * 5)),)

    return model

def efficientnet_b0(split_size, num_boxes, num_classes, pretrained=True):
    S, B, C = split_size, num_boxes, num_classes

    #Import Resnet50 from pytorch
    if pretrained:
        model = torchvision.models.efficientnet_b0(pretrained)
  
    #Enable backprop for all layers
    for param in model.parameters():
        param.requires_grad = False
  
    #Remove FC layers
    modules=list(model.children())[:-2]
    model=nn.Sequential(*modules)

    # Add custom Adaptive Average Pooling
    model.avgpool = nn.AdaptiveAvgPool2d((split_size, split_size))

    # Flatten the output and add two fully connected layers
    # Calculate the input features to the first fully connected layer
    num_features = 1280 * S * S  # 1280 is the number of output channels before the classifier
    model.fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(num_features, 496),
        nn.Dropout(0.5),
        nn.LeakyReLU(0.1),
        nn.Linear(496, S * S * (C + B * 5)),
    )

    return model


def print_info(model_name = 'resnet50', S = 7, B = 2, C = 1):
    batch_size = 2
    num_channels = 3 # RGB
    from torchsummary import summary
    from torch.utils.tensorboard import SummaryWriter
    import shutil
    if shutil.os.path.exists(f'Architecture/{model_name}'):
        shutil.rmtree(f'Architecture/{model_name}')
    writer = SummaryWriter(f'Architecture/{model_name}')
    if model_name == 'resnet50':
        model = resnet50(split_size=S, num_boxes=B, num_classes=C).to(config.DEVICE)
        size_img = 448
    elif model_name == 'vgg16':
        model = vgg16(split_size=S, num_boxes=B, num_classes=C).to(config.DEVICE)
        size_img = 448
    elif model_name == 'efficientnet':
        model = efficientnet_b0(split_size=S, num_boxes=B, num_classes=C).to(config.DEVICE)
        size_img = 224
    
    # Input tensor
    x = torch.randn((batch_size, num_channels, size_img, size_img)).to(config.DEVICE)
    writer.add_graph(model, x, use_strict_trace=False)

    print(summary(model, (num_channels, size_img, size_img)))
    print(model(x).shape)
    writer.close()

# ¡¡Test Case!!
def test(S = 7, B = 2, C = 20):
    batch_size = 2
    num_channels = 3 # RGB
    size_img = 448
    x_YOLO = torch.randn((batch_size, num_channels, size_img, size_img))
    x_efficientnet = torch.randn((batch_size, num_channels, 224, 224))

    from torch.utils.tensorboard import SummaryWriter    
    writer = SummaryWriter('Architecture/resnet50')


    # Resnet50
    model_resnet50 = resnet50(split_size=S, num_boxes=B, num_classes=C, pretrained = True)
    writer = SummaryWriter('Architecture/resnet50')
    writer.add_graph(model_resnet50, x_YOLO, use_strict_trace=False)

    # VGG16
    model_vgg16 = vgg16(split_size=S, num_boxes=B, num_classes=C, pretrained = True)
    writer = SummaryWriter('Architecture/vgg16')
    writer.add_graph(model_vgg16, x_YOLO, use_strict_trace=False)

    # EfficientNet B0
    model_efficientnet = efficientnet_b0(split_size=S, num_boxes=B, num_classes=C)
    writer = SummaryWriter('Architecture/efficientnet')
    writer.add_graph(model_efficientnet, x_efficientnet, use_strict_trace=False)


    total_params = sum(p.numel() for p in model_resnet50.parameters())
    print(f"Total de parámetros en el modelo ResNet50: {total_params}")
    total_params = sum(p.numel() for p in model_vgg16.parameters())
    print(f"Total de parámetros en el modelo VGG16: {total_params}")
    total_params = sum(p.numel() for p in model_efficientnet.parameters())
    print(f"Total de parámetros en el modelo EfficientNet: {total_params}")

    # Guardar el modelo RESNET en un buffer en lugar de un archivo en el disco
    buffer = io.BytesIO()
    torch.save(model_resnet50.state_dict(), buffer)
    size_bytes = buffer.tell()
    size_mb = size_bytes / (1024 ** 2)
    print(f'Tamaño del modelo ResNet: {size_mb:.2f} MB')

    # Guardar el modelo VGG16 en un buffer en lugar de un archivo en el disco
    buffer = io.BytesIO()
    torch.save(model_vgg16.state_dict(), buffer)
    size_bytes = buffer.tell()
    size_mb = size_bytes / (1024 ** 2)
    print(f'Tamaño del modelo VGG16: {size_mb:.2f} MB')

    # Guardar el modelo EfficientNet en un buffer en lugar de un archivo en el disco
    buffer = io.BytesIO()
    torch.save(model_efficientnet.state_dict(), buffer)
    size_bytes = buffer.tell()
    size_mb = size_bytes / (1024 ** 2)
    print(f'Tamaño del modelo EfficientNet: {size_mb:.2f} MB')


if __name__ == "__main__":
    test()