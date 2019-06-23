import abc
import torch
import os
import numpy as np
import ipdb
import math
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as transf
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import List, Dict

from utils.type import Task


class _MaskedDataset(Dataset):
    """Special abstract Dataset where images and masks are stored in separate dir with same names"""

    def __init__(self, task: Task, normalize: bool, image_transforms: List, pair_transforms: List[str]) -> None:
        """User should call set_paths() before evoking super's __init__"""
        preprocess = [transf.ToTensor()]
        if normalize:
            preprocess.append(
                transf.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]))
        self.normalize = transf.Compose(preprocess)
        self.image_transforms = transf.Compose(image_transforms) if image_transforms else None
        self.pair_transforms = pair_transforms

        if task.setting == "train":
            assert len(self.image_paths) == len(
                self.mask_paths
            ), f"Every image should have corresponding label mask. Number of images\
            ({len(self.image_paths)}) does not match number of masks ({len(self.mask_paths)})."

            self.image_paths.sort()
            self.mask_paths.sort()

        for attr in task.attributes:
            if hasattr(task, attr):
                setattr(self, attr, getattr(task, attr))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict:
        raw_path = self.image_paths[idx]
        image = Image.open(raw_path)

        if self.setting == "test":
            return {'image': self.normalize(image),
                    'raw_path': raw_path}

        mask = Image.open(self.mask_paths[idx])

        if self.image_transforms is not None:
            image = self.image_transforms(image)

        p = np.random.rand(1)
        if p >= 0.5:
            image, mask = self._transform(image, mask)

        image = self.normalize(image)
        mask = TF.to_tensor(mask)

        return {
            'image': image,
            'mask': self.mask_postprocess(mask),
            'raw_path': raw_path
        }

    def _transform(self, image, mask):
        def coin_flip():
            return np.random.rand(1) >= 0.5

        if "crop" in self.pair_transforms and coin_flip():
            resize = transf.Resize(size=(450, 450))
            image, mask = resize(image), resize(mask)
            i, j, h, w = transf.RandomCrop.get_params(
                image, output_size=(400, 400))
            image, mask = TF.crop(image, i, j, h, w), TF.crop(mask, i, j, h, w)

        if "hflip" in self.pair_transforms and coin_flip():
            image, mask = TF.hflip(image), TF.hflip(mask)

        if "vflip" in self.pair_transforms and coin_flip():
            image, mask = TF.vflip(image), TF.vflip(mask)

        if "rotate" in self.pair_transforms and coin_flip():
            angle = (np.random.rand(1) - 0.5) * 30
            image, mask = TF.rotate(image, angle), TF.rotate(mask, angle)

        return image, mask

    @abc.abstractmethod
    def set_paths(self, task: Task) -> None:
        """Method to load correct `image_paths` and `mask_paths`"""
        return

    @abc.abstractmethod
    def mask_postprocess(self, mask: torch.Tensor) -> torch.Tensor:
        """Middleware method to process mask files before returning to trainer"""
        return

    @abc.abstractmethod
    def get_colormap(self) -> Dict:
        """Returns label to color mapping for visualization"""
        return

    def get_num_class(self) -> int:
        return self.__getitem__(0)['mask'].max().item() + 1


class AerialDataset(_MaskedDataset):
    def __init__(self, task: Task, normalize: bool = True, image_transforms: List = [], pair_transforms: List = []):
        self.set_paths(task)
        super(AerialDataset, self).__init__(task, normalize, image_transforms, pair_transforms)

    def set_paths(self, task):
        IMAGE_DIR = os.path.join(task.ROOT_DIR, task.IMAGE_DIR)
        self.image_paths = np.asarray([
            os.path.join(IMAGE_DIR, filename)
            for filename in os.listdir(IMAGE_DIR)
        ])

        if hasattr(task, 'MASK_DIR'):
            MASK_DIR = os.path.join(task.ROOT_DIR, task.MASK_DIR)
            self.mask_paths = np.asarray([
                os.path.join(MASK_DIR, filename)
                for filename in os.listdir(MASK_DIR)
            ])

    def mask_postprocess(self, mask: torch.Tensor) -> torch.Tensor:
        return torch.where(mask > 0.75, torch.ones_like(mask), torch.zeros_like(mask)).long()

    def get_colormap(self) -> Dict:
        return {0: (0, 0, 0), 1: (255, 255, 255)}
