import numpy as np
import argparse

from data import *
from submission import *

from keras import optimizers
from keras.callbacks import Callback, ModelCheckpoint
from keras.layers import *
from keras.models import Model

from pathlib import Path

from sklearn.utils import class_weight, shuffle

# Parse command line arguments
parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', required=True, default='./data',
                    help='Where the data is stored.')
parser.add_argument('--out_dir', required=True, default='./baseline_out',
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

# Get patches and labels per patch
patch_size = 16

val_patches, val_labels = get_patches_labels(
    val_images, val_masks, patch_size=patch_size)
train_patches, train_labels = get_patches_labels(
    train_images, train_masks, patch_size=patch_size)

# Normalize the data
patch_mean = train_patches.mean(axis=0)
patch_std = train_patches.std(axis=0)

train_patches = (train_patches - patch_mean) / patch_std
val_patches = (val_patches - patch_mean) / patch_std

# Define the model
img_input = Input(shape=(patch_size, patch_size, 3))

x = Conv2D(filters=32, kernel_size=(5, 5),
           activation='relu', padding='same')(img_input)
x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
x = BatchNormalization()(x)

x = Conv2D(filters=64, kernel_size=(5, 5),
           activation='relu', padding='same')(x)
x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
x = BatchNormalization()(x)

x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)

out_label = Dense(1, activation='sigmoid')(x)

model = Model(img_input, out_label)

# Compute the appropriate class weights
weights = class_weight.compute_class_weight('balanced',
                                            np.unique(train_labels),
                                            train_labels)

# Callbacks
filepath = 'weights.hdf5'
checkpoint = ModelCheckpoint(
    filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks = [checkpoint]

# Train the model
opt = optimizers.Adam()
model.compile(optimizer=opt, loss='binary_crossentropy')

hist = model.fit(train_patches, train_labels,
                 validation_data=(val_patches, val_labels),
                 batch_size=32, epochs=10,
                 class_weight=weights,
                 shuffle=True, verbose=True, callbacks=callbacks)

# Load best weights
model.load_weights(filepath)

# Load test data
test_data = load_images(test_images_folder)

# Get patches for the test data
test_patches = get_patches(test_data, patch_size=patch_size)

# Normalize the test patches
test_patches = (test_patches - patch_mean) / patch_std

# Predict labels for the patches
test_labels = model.predict(test_patches)

# Convert the labels to mask images
_, test_height, test_width, _ = test_data.shape
test_pred = labels_to_images(test_labels, test_height, test_width, patch_size)

# Store predicted masks to files
for i, sample in enumerate(sorted(test_images_folder.iterdir())):
    io.imsave(predictions_folder / sample.name, test_pred[i])

# Create submission file
pred_files = []
for file in sorted(predictions_folder.iterdir(), key=lambda x: int(str(x).split('_')[-1].split('.')[0])):
    pred_files.append(str(file))
masks_to_submission(submission_file, *pred_files)
