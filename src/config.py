import torch
import os
import subprocess
import datetime
import logging
import sys

from models import deeplab, fcn
from utils.type import Task
from data import AerialDataset

PRODUCTION = True
USE_GPU = PRODUCTION

PRINT_BATCH_FREQ = 1
LOG_ITER_FREQ = 10
SAVE_ITER_FREQ = None
EVALUATE_ITER_FREQ = 100
VISUALIZE_ITER_FREQ = 10

MODEL = fcn.get_model(pretrained=True)
if USE_GPU:
    MODEL = MODEL.cuda()

DATASET = AerialDataset

TRAIN = Task({
    'setting': "train",
    'ROOT_DIR': "../data",
    'IMAGE_DIR': "training/images",
    'MASK_DIR': "training/groundtruth",
    'BATCH_SIZE': 2,
    'SHUFFLE': True,
    'OPTIMIZER': "Adam",
    'OPTIM_PARAMS': {
        'lr': 1e-4,
        'weight_decay': 5e-4,
    },
    'LOSS': "cross_entropy",
    'NUM_EPOCHS': 200
})

TEST = Task({
    'setting': "test",
    'ROOT_DIR': "../data",
    'IMAGE_DIR': "test_images",
    'BATCH_SIZE': 2,
    'SHUFFLE': False,
    'LOSS': "cross_entropy",
})

BRANCH = subprocess.check_output(["git", "rev-parse", "--abbrev-ref",
                                  "HEAD"]).strip().decode("utf-8")
NAME = "%s-%s-%s" % (BRANCH, TRAIN.OPTIMIZER, TRAIN.BATCH_SIZE)

# Logging configuration
LOG_PATH = "../log/%s" % BRANCH
LOG_NAME = os.path.join(
    LOG_PATH, f"{NAME}-{datetime.datetime.now().strftime('%d-%m--%H-%M')}.log")
OUT_PATH = "%s_%s" % (LOG_NAME[:-4], MODEL.__class__.__name__)

if not os.path.isdir(LOG_PATH):
    os.mkdir(LOG_PATH)
if not os.path.isdir(OUT_PATH):
    os.mkdir(OUT_PATH)

logFormatter = "%(asctime)s - [%(levelname)s] %(message)s"

if PRODUCTION:
    __LOG_PARAMS__ = {
        'filename': LOG_NAME,
        'filemode': 'a',
    }
else:
    __LOG_PARAMS__ = {
        'stream': sys.stdout,
    }

__LOG_PARAMS__.update({
    'format': logFormatter,
    'level': logging.INFO,
    'datefmt': '%d/%m/%Y %H:%M:%S'
})
