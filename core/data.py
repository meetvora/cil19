import abc
import torch
import os
import numpy as np
import ipdb
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import transforms as transf
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import List, Dict

from utils.type import Task
from utils.labels import cityscape, mapillary


class _MaskedDataset(Dataset):
    """Special abstract Dataset where images and masks are stored in separate dir with same names"""

    def __init__(self, task: Task, normalize: bool, transforms: List, mask_transforms: List) -> None:
        """User should call set_paths() before evoking super's __init__"""
        preprocess = [transf.ToTensor()]
        if normalize:
            preprocess.append(
                transf.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]))
        self.normalize = transf.Compose(preprocess)
        self.transforms = None
        self.mask_transforms = None

        if task.setting == "train":
            assert len(self.image_paths) == len(
                self.mask_paths
            ), f"Every image should have corresponding label mask. Number of images\
            ({len(self.image_paths)}) does not match number of masks ({len(self.mask_paths)})."

            self.image_paths.sort()
            self.mask_paths.sort()

            if transforms:
                self.transforms = transf.Compose(transforms)
                for i, flag in enumerate(mask_transforms):
                    if flag == 1:
                        self.mask_transforms.append(transforms[i])
                self.mask_transforms = transf.Compose(self.mask_transforms)

        for attr in task.attributes:
            if hasattr(task, attr):
                setattr(self, attr, getattr(task, attr))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict:
        raw_path = self.image_paths[idx]
        image = Image.open(raw_path)

        mask = Image.open(self.mask_paths[idx])
        mask = np.asarray(mask)
        
        p = np.random.rand(1)
        if self.transforms is not None and p >= 0.5:
            image = self.transforms(image)
            mask = self.mask_transforms(mask)

        image = self.normalize(image)
        return {
            'image': image,
            'mask': self.mask_postprocess(mask),
            'raw_path': raw_path
        }

    @abc.abstractmethod
    def set_paths(self, task: Task) -> None:
        """Method to load correct `image_paths` and `mask_paths`"""
        return

    @abc.abstractmethod
    def mask_postprocess(self, mask: np.ndarray) -> np.ndarray:
        """Middleware method to process mask files before returning to trainer"""
        return

    @abc.abstractmethod
    def get_colormap(self) -> Dict:
        """Returns label to color mapping for visualization"""
        return

    def get_num_class(self) -> int:
        mask = self.__getitem__(0)['mask']
        return mask.max() + 1


class CityScapeDataset(_MaskedDataset):
    def __init__(self,
                 task: Task,
                 normalize: bool = True,
                 transforms: List = [],
                 use_trainId: bool = True) -> None:

        self.set_paths(task)
        super(CityScapeDataset, self).__init__(task, normalize, transforms)
        void_classes = [
            label.id for label in cityscape.labels if label.category == "void"
        ]
        self.max_void_label = max(void_classes)
        self.mask_postprocess = self._translate_to_trainId if use_trainId else self._remove_void_labels

    def set_paths(self, task):
        validator = lambda u: "._" not in u
        mask_validator = lambda u: "_labelIds.png" in u and validator(u)

        IMAGE_DIR = os.path.join(task.ROOT_DIR, task.IMAGE_DIR)
        cities = [city for city in os.listdir(IMAGE_DIR) if validator(city)]
        self.image_paths = list()
        for city in cities:
            localpath = os.path.join(IMAGE_DIR, city)
            self.image_paths += [
                os.path.join(localpath, filename)
                for filename in os.listdir(localpath) if validator(filename)
            ]
        self.image_paths = np.asarray(self.image_paths).flatten()

        MASK_DIR = os.path.join(task.ROOT_DIR, task.MASK_DIR)
        self.mask_paths = list()
        for city in cities:
            localpath = os.path.join(MASK_DIR, city)
            self.mask_paths += [
                os.path.join(localpath, filename)
                for filename in os.listdir(localpath)
                if mask_validator(filename)
            ]
        self.mask_paths = np.asarray(self.mask_paths).flatten()

    def _remove_void_labels(self, mask: np.ndarray) -> np.ndarray:
        """reduces all 'void' category labels to single class '0'"""
        mask = np.where(mask <= self.max_void_label, self.max_void_label, mask)
        mask = mask - self.max_void_label
        return mask

    def _translate_to_trainId(self, mask: np.ndarray) -> np.ndarray:
        """maps 35 classes to 19"""
        for id, trainId in cityscape.id2trainId.items():
            mask = np.where(mask == id, trainId, mask)
        mask = np.where((mask == 255) | (mask == -1), 19, mask)
        return mask

    def get_colormap(self):
        return {label.trainId: label.color for label in cityscape.labels}


class MapillaryDataset(_MaskedDataset):
    def __init__(self,
                 task: Task,
                 normalize: bool = True,
                 transforms: List = []) -> None:
        self.set_paths(task)
        super(MapillaryDataset, self).__init__(task, normalize, transforms)

    def set_paths(self, task: Task) -> None:
        IMAGE_DIR = os.path.join(task.ROOT_DIR, task.IMAGE_DIR)
        self.image_paths = np.asarray([
            os.path.join(IMAGE_DIR, filename)
            for filename in os.listdir(IMAGE_DIR)
        ])

        MASK_DIR = os.path.join(task.ROOT_DIR, task.MASK_DIR)
        self.mask_paths = np.asarray([
            os.path.join(MASK_DIR, filename)
            for filename in os.listdir(MASK_DIR)
        ])

    def mask_postprocess(self, mask: np.ndarray) -> np.ndarray:
        return mask

    def get_colormap(self) -> Dict:
        return mapillary.labels2color

class AerialDataset(_MaskedDataset):
    def __init__(self, task: Task, normalize: bool = True, transforms: List = [], mask_transforms: List = []):
        self.set_paths(task)
        super(AerialDataset, self).__init__(task, normalize, transforms, mask_transforms)

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

    def mask_postprocess(self, mask: np.ndarray) -> np.ndarray:
        return np.where(mask > 0, 1, 0)

    def get_colormap(self) -> Dict:
        return {0: (0, 0, 0), 1: (255, 255, 255)}
