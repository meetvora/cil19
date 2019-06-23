import os
import numpy as np
import logging
import ipdb
import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset, DataLoader

import config
from data import Dataset
from loss import Loss
from metrics import ConfusionMatrix
from utils import beautify, device
from utils.visualize import Visualizer

if config.USE_GPU:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

logging.basicConfig(**config.__LOG_PARAMS__)
logger = logging.getLogger(__name__)


def train(model: nn.Module, dataset: Dataset,
          validate_data: Dataset = None) -> None:
    loader = DataLoader(dataset,
                        batch_size=dataset.BATCH_SIZE,
                        shuffle=dataset.BATCH_SIZE)

    optimizer = getattr(torch.optim,
                        config.TRAIN.OPTIMIZER)(model.parameters(),
                                                **config.TRAIN.OPTIM_PARAMS)
    overall_iter = 0
    evaluation = ConfusionMatrix(dataset.get_num_class())

    model.train()
    for epoch in range(config.TRAIN.NUM_EPOCHS):
        total_loss = 0
        for batch_idx, samples in enumerate(loader):
            images, target = device([samples['image'], samples['mask']],
                                    gpu=config.USE_GPU)
            outputs = model(images)['out']
            output_mask = outputs.argmax(1)

            batch_loss = Loss.cross_entropy2D(outputs, target)
            total_loss += batch_loss.item()
            overall_loss = total_loss / ((batch_idx + 1))
            evaluation.update(output_mask, target)

            batch_loss.backward()
            optimizer.step()

            if batch_idx % config.PRINT_BATCH_FREQ == 0:
                metrics = evaluation()
                logger.info(f'Train Epoch: {epoch}, {batch_idx}')
                logger.info(
                    f'Batch loss: {batch_loss.item():.6f}, Overall loss: {overall_loss:.6f}'
                )
                for met in beautify(metrics[0]):
                    logger.info(f'{met}')
                logger.info(f'Classwise IoU')
                for met in beautify(metrics[1]):
                    logger.info(f'{met}')
                logger.info("\n")

            overall_iter += 1
            if config.SAVE_ITER_FREQ and overall_iter % config.SAVE_ITER_FREQ == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(config.LOG_PATH,
                                 config.NAME + f"-iter={overall_iter}"))


def evaluate(model: nn.Module,
             dataset: Dataset,
             pretrained: bool = False,
             visualizer: Visualizer = None) -> None:
    loader = DataLoader(dataset,
                        batch_size=dataset.BATCH_SIZE,
                        shuffle=dataset.SHUFFLE)
    if pretrained:
        model.load_state_dict(
            torch.load(os.path.join(config.LOG_PATH, config.NAME)))
    logger.info("[+] Evaluating model...")

    imgize = torchvision.transforms.ToPILImage()

    with torch.no_grad():
        model.eval()
        for batch_idx, samples in enumerate(loader):
            images, raw_path = device(
                [samples['image']], gpu=config.USE_GPU)[0], samples['raw_path']
            outputs = model(images)['out']
            output_mask = outputs.argmax(1)

            for mask, path in zip(output_mask, raw_path):
                _mask = torch.where(mask == 1,
                                    torch.ones_like(mask) * 255,
                                    torch.zeros_like(mask)).byte()
                _mask = imgize(_mask.cpu())
                filename = os.path.basename(path)
                _mask.save(os.path.join(config.OUT_PATH, filename))
    logger.info("[+] Done.")


def main():
    model = config.MODEL
    augmentation = [torchvision.transforms.ColorJitter(0.25, 0.25, 0.25, 0.25)]
    paired_augmentation = ["crop", "hflip", "vflip", "rotate"]

    dataset = config.DATASET(task=config.TRAIN,
                             normalize=True,
                             image_transforms=augmentation,
                             pair_transforms=paired_augmentation)
    test_dataset = config.DATASET(task=config.TEST, normalize=True)

    train(model, dataset, dataset)
    evaluate(model, test_dataset)


if __name__ == "__main__":
    main()
