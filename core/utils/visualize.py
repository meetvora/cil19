import os
import numpy as np
import torch
import ipdb
from PIL import Image
from typing import Dict, List
import matplotlib.pyplot as plt


class Visualizer(object):
    def __init__(self, mapping: Dict, is_save: bool = False):
        self.mapping = mapping
        self.is_save = is_save

        palette = torch.tensor([2**25 - 1, 2**15 - 1, 2**21 - 1])
        colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
        self.colors = (colors % 255).cpu().numpy().astype("uint8")

    def __call__(self,
                 mask: torch.Tensor,
                 raw_image: torch.Tensor,
                 filename: str = ""):
        fig = plt.figure(figsize=(9, 4))

        mask = Image.fromarray(mask.byte().cpu().numpy())
        mask.putpalette(self.colors)
        if self.is_save and filename:
            plt.savefig(f"{filename}.png")
        fig.add_subplot(1, 2, 2)
        plt.imshow(mask)

        image = Image.open(raw_image)
        fig.add_subplot(1, 2, 1)
        plt.imshow(image)

        plt.show()
