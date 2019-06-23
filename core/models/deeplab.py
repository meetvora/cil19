import torchvision

model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=2)