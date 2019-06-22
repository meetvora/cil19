import os
import numpy as np
import torch
import logging
import ipdb
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

    optimizer = getattr(torch.optim, config.TRAIN.OPTIMIZER)(model.parameters(),
                                                       **config.TRAIN.OPTIM_PARAMS)
    # loss = Loss(config.TRAIN.LOSS)
    overall_iter = 0
    evaluation = ConfusionMatrix(dataset.get_num_class())

    for epoch in range(config.TRAIN.NUM_EPOCHS):
        total_loss = 0
        for batch_idx, samples in enumerate(loader):
            images, target = samples['image'], samples['mask']
            if config.USE_GPU:
              images, target = images.cuda(), target.cuda()
            outputs = model(images)['out']
            output_mask = outputs.argmax(1)

            batch_loss = Loss.cross_entropy2D(outputs, target.long())
            total_loss += batch_loss.item()
            overall_loss = total_loss / ((batch_idx + 1))
            evaluation.update(output_mask, target)

            batch_loss.backward()
            optimizer.step()

            if batch_idx % config.PRINT_BATCH_FREQ == 0:
                metrics = evaluation()
                logger.info(
                    f'Train Epoch: {epoch} [{batch_idx}]\Batch loss: {batch_loss.item():.6f}\
                    \tOverall loss: {overall_loss:.6f}'
                )
                logger.info(
                    f'Overall metrics: {beautify(metrics[0])}\nClasswise\n{beautify(metrics[1])}'
                )
            overall_iter += 1
            if config.SAVE_ITER_FREQ and overall_iter % config.SAVE_ITER_FREQ == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(config.LOG_PATH,
                                 config.NAME + f"-iter={overall_iter}"))
            if config.EVALUATE_ITER_FREQ and overall_iter % config.EVALUATE_ITER_FREQ == 0:
                evaluate(model, validate_data)


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

    total_loss = 0
    evaluation = ConfusionMatrix(dataset.get_num_class())

    with torch.no_grad():
        model.eval()
        for batch_idx, samples in enumerate(loader):
            (images, target), raw_path = device(
                [samples['image'], samples['mask']],
                gpu=config.USE_GPU), samples['raw_path']
            outputs = model(images)['out']
            output_mask = outputs.argmax(1)

            batch_loss = Loss.cross_entropy2D(outputs, target.long())
            total_loss += batch_loss.item()
            overall_loss = total_loss / ((batch_idx + 1))
            evaluation.update(output_mask, target)

            if batch_idx % config.PRINT_BATCH_FREQ == 0:
                metrics = evaluation()
                logger.info(
                    f'Validation loss (batchwise): {batch_loss.item():.6f}\t\
                    Overall loss: {overall_loss:.6f}')
                logger.info(
                    f'Overall metrics: {beautify(metrics[0])}\nClasswise\n{beautify(metrics[1])}'
                )

            if visualizer and batch_idx % config.VISUALIZE_ITER_FREQ == 0:
                ipdb.set_trace()
                visualizer(mask=output_mask[0], raw_image=raw_path[0])

def main():
    model = config.MODEL
    dataset = config.DATASET(task=config.TRAIN, normalize=True)
    test_dataset = config.DATASET(task=config.TEST, normalize=True)
    
    train(model, dataset, dataset)

if __name__ == "__main__":
    main()