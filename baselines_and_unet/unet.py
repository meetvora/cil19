import cv2
import numpy as np
import argparse

from data import *
from metrics import *
from submission import *

from keras import optimizers
from keras.callbacks import Callback, ModelCheckpoint
from keras.layers import *
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

from pathlib import Path

from segmentation_models import Unet
from segmentation_models.backbones import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss, jaccard_loss
from segmentation_models.metrics import iou_score, f1_score

from sklearn.utils import class_weight, shuffle

# Parse command line arguments
parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', required=True, default='./data',
                    help='Where the data is stored.')
parser.add_argument('--out_dir', required=True, default='./unet_out',
                    help='Where the results are stored.')

ARGS = parser.parse_args()

# File path setup
data_folder = Path(ARGS.data_dir)
out_folder = Path(ARGS.out_dir)

train_images_folder = data_folder / 'training/images'
train_masks_folder = data_folder / 'training/groundtruth'

test_images_folder = data_folder / 'test_images'

out_folder.mkdir(exist_ok=True)

predictions_folder = out_folder / 'predictions'
predictions_folder.mkdir(exist_ok=True)

submission_file = out_folder / 'submission.csv'

# Load training images
train_images = load_images(train_images_folder)

# Load training masks
train_masks = load_masks(train_masks_folder)
train_masks[train_masks > 0.5] = 1.0

# Training/validation split
nr_samples = train_images.shape[0]

train_rate = 0.9
index_train = np.random.choice(nr_samples, int(
    nr_samples * train_rate), replace=False)
index_val = list(set(range(nr_samples)) - set(index_train))

train_images, train_masks = shuffle(train_images, train_masks)

val_images, val_masks = train_images[index_val], train_masks[index_val]
train_images, train_masks = train_images[index_train], train_masks[index_train]

# Reshape images to make them usable by the Unet architecture


def resize_images_masks(images, masks, shape):
    n, _, _, dim = images.shape

    new_images = np.zeros((n,) + shape + (dim,), dtype='float32')
    new_masks = np.zeros((n,) + shape, dtype='float32')

    for i in range(n):
        new_images[i] = cv2.resize(images[i], shape)
        new_masks[i] = cv2.resize(masks[i], shape)

    return new_images, np.expand_dims(new_masks, axis=3)


new_shape = (416, 416)

re_train_images, re_train_masks = resize_images_masks(
    train_images, train_masks, new_shape)
re_val_images, re_val_masks = resize_images_masks(
    val_images, val_masks, new_shape)

train_images = re_train_images
train_masks = re_train_masks

val_images = re_val_images
val_masks = re_val_masks

# Compute the appropriate class weights
weights = class_weight.compute_class_weight('balanced',
                                            np.unique(train_masks > 0.5),
                                            (train_masks > 0.5).flatten())

# Callbacks
filepath = 'weights.hdf5'
checkpoint = ModelCheckpoint(
    filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks = [checkpoint]

# Define the model
BACKBONE = 'resnet101'

model = Unet(BACKBONE, classes=1, activation='sigmoid')
model.compile('Adam', loss=weighted_binary_crossentropy(weights))

# Create generator
preprocess_input = get_preprocessing(BACKBONE)


def custom_generator(batches, trans_list):
    def coin_flip():
        return np.random.rand(1) >= 0.5

    while True:
        image_batch, mask_batch = next(batches)

        for i in range(len(image_batch)):
            if 'hflip' in trans_list and coin_flip():
                image_batch[i] = image_batch[i, :, ::-1]
                mask_batch[i] = mask_batch[i, :, ::-1]

            if 'vflip' in trans_list and coin_flip():
                image_batch[i] = image_batch[i, ::-1]
                mask_batch[i] = mask_batch[i, ::-1]

            if 'rotate' in trans_list and coin_flip():
                angle = (np.random.rand(1) - 0.5) * 2 * 30

                image_batch[i] = rotate(image_batch[i], angle)
                mask_batch[i] = rotate(mask_batch[i], angle)

        yield(image_batch, mask_batch)


train_images = preprocess_input(train_images)
val_images = preprocess_input(val_images)

flow_generator = ImageDataGenerator()
batches = flow_generator.flow(train_images, train_masks, batch_size=16)

trans_list = ['hflip', 'vflip', 'rotate']
generator = custom_generator(batches, trans_list)

# Train the model
hist = model.fit_generator(
    generator,
    steps_per_epoch=len(train_images) // 16, epochs=125,
    validation_data=(val_images, val_masks),
    callbacks=callbacks
)

# Load best weights
model.load_weights(filepath)

# Load test data
test_data = load_images(test_images_folder)

# Predict the data
test_data = preprocess_input(test_data)
test_pred = model.predict(test_data)

# Store predicted masks to files
for i, sample in enumerate(sorted(test_images_folder.iterdir())):
    io.imsave(predictions_folder / sample.name, np.squeeze(test_pred[i]))

# Create submission file
pred_files = []
for file in sorted(predictions_folder.iterdir(), key=lambda x: int(str(x).split('_')[-1].split('.')[0])):
    pred_files.append(str(file))
masks_to_submission(submission_file, *pred_files)
