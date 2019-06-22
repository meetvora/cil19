import torch

import config
from main import train, evaluate
from utils.visualize import Visualizer

if config.USE_GPU:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


def main():
    model = config.MODEL
    dataset = config.DATASET(task=config.TRAIN, normalize=True)
    visualizer = Visualizer(dataset.get_colormap(), is_save=False)
    evaluate(model=model, dataset=dataset, visualizer=visualizer)


if __name__ == "__main__":
    main()
