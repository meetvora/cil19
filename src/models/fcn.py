import torch
import torchvision


def get_model(pretrained=False):
    model = torchvision.models.segmentation.fcn_resnet101(
        pretrained=pretrained)
    model.classifier[4] = torch.nn.Conv2d(512, 2, kernel_size=(1, 1))
    for param in model.parameters():
        param.requires_grad = True
    return model
