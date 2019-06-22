import torch
_model = torch.hub.load('pytorch/vision',
                       'deeplabv3_resnet101',
                       pretrained=True)

def resize(model: torch.nn.Module, n_class: int):
	model.classifier[4] = torch.nn.Conv2d(256, n_class, kernel_size=(1, 1))
	model.aux_classifier[4] = torch.nn.Conv2d(256, n_class, kernel_size=(1, 1))
	return model
	
model = resize(_model, 2)