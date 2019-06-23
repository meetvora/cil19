import torchvision

model = torchvision.models.segmentation.fcn_resnet101(pretrained=False, num_classes=2)