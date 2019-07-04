import torch
import torchvision


def get_model(pretrained=False):
    if pretrained:
        model = torch.hub.load('pytorch/vision',
                               'deeplabv3_resnet101',
                               pretrained=pretrained)
        model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1))
        model.aux_classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1))
    else:
        model = torchvision.models.segmentation.deeplabv3_resnet101(
            pretrained=False, num_classes=2)
    for param in model.parameters():
        param.requires_grad = True
    return model
