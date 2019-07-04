import numpy as np
import argparse

from data import *
from submission import *

from keras import optimizers
from keras.callbacks import Callback, ModelCheckpoint
from keras.layers import *
from keras.models import Model

from pathlib import Path

from sklearn.svm import SVC
from sklearn.utils import class_weight, shuffle

# Parse command line arguments
parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', required=True, default='./data',
                    help='Where the data is stored.')
parser.add_argument('--out_dir', required=True, default='./better_baseline_out',
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

# Split the validation set in half to use for the CNN and SVM
half_val_size = val_images.shape[0] // 2

cnn_val_images = val_images[:half_val_size]
cnn_val_masks = val_masks[:half_val_size]

svm_val_images = val_images[half_val_size:]
svm_val_masks = val_masks[half_val_size:]

# Get patches and labels per patch
patch_size = 16
pad_size = 24

cnn_val_patches, cnn_val_labels = get_patches_labels(
    cnn_val_images, cnn_val_masks, patch_size=patch_size, pad_size=24)
svm_val_patches, svm_val_labels = get_patches_labels(
    svm_val_images, svm_val_masks, patch_size=patch_size, pad_size=24)
train_patches, train_labels = get_patches_labels(
    train_images, train_masks, patch_size=patch_size, pad_size=24)

# Normalize the data
patch_mean = train_patches.mean(axis=0)
patch_std = train_patches.std(axis=0)

train_patches = (train_patches - patch_mean) / patch_std
cnn_val_patches = (cnn_val_patches - patch_mean) / patch_std
svm_val_patches = (svm_val_patches - patch_mean) / patch_std

# Define the model
input_size = patch_size + 2 * pad_size
img_input = Input(shape=(input_size, input_size, 3))

x = Conv2D(filters=32, kernel_size=(3, 3),
           activation='relu', padding='same')(img_input)
x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
x = BatchNormalization()(x)

x = Conv2D(filters=64, kernel_size=(3, 3),
           activation='relu', padding='same')(x)
x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
x = BatchNormalization()(x)

x = Conv2D(filters=64, kernel_size=(3, 3),
           activation='relu', padding='same')(x)
x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
x = BatchNormalization()(x)

x = Conv2D(filters=128, kernel_size=(3, 3),
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
model.compile(optimizer=opt, loss="binary_crossentropy")

hist = model.fit(train_patches, train_labels,
                 validation_data=(cnn_val_patches, cnn_val_labels),
                 batch_size=32, epochs=1,
                 class_weight=weights,
                 verbose=True, callbacks=callbacks)

# Load best weights
model.load_weights(filepath)

# Predict the labels for the training/validation data
train_pred = model.predict(train_patches)
svm_val_pred = model.predict(svm_val_patches)

# Convert training predictions to 7x7 patches
_, train_height, train_width, _ = train_images.shape

train_pred_img = labels_to_images(
    train_pred, train_height // patch_size, train_width // patch_size, 1, 0.5)
train_pred_img = np.expand_dims(train_pred_img, axis=3)
train_pred_patches = get_padded_patches(
    train_pred_img, patch_size=1, pad_size=3)

val_pred_img = labels_to_images(
    svm_val_pred, train_height // patch_size, train_width // patch_size, 1, 0.5)
val_pred_img = np.expand_dims(val_pred_img, axis=3)
val_pred_patches = get_padded_patches(val_pred_img, patch_size=1, pad_size=3)

train_svm_feat = np.reshape(
    train_pred_patches, (train_pred_patches.shape[0], -1))
val_svm_feat = np.reshape(val_pred_patches, (val_pred_patches.shape[0], -1))

# Compute SVM weights
svm_weights = {}
for label in np.unique(train_labels):
    svm_weights[int(label)] = weights[int(label)]

# Train SVM
svm = SVC(C=1, gamma='scale', class_weight=svm_weights, kernel='linear')
svm.fit(train_svm_feat, train_labels)

# Load test data
test_data = load_images(test_images_folder)

# Predict images one by one to avoid memory exhaustion
_, test_height, test_width, _ = test_data.shape
labels_per_image = (test_height // patch_size) * (test_width // patch_size)
cnn_labels = np.zeros((labels_per_image * test_data.shape[0], 1))
svm_labels = np.zeros((labels_per_image * test_data.shape[0], 1))

for i in range(len(test_data)):
    test_image = np.expand_dims(test_data[i], axis=0)
    test_patches = get_padded_patches(
        test_image, patch_size=patch_size, pad_size=pad_size)
    test_patches = (test_patches - patch_mean) / patch_std

    cnn_pred = model.predict(test_patches)
    cnn_labels[i * labels_per_image:(i + 1) * labels_per_image] = cnn_pred

    cnn_pred_img = labels_to_images(
        cnn_pred, test_height // patch_size, test_width // patch_size, 1)
    cnn_pred_img = np.expand_dims(cnn_pred_img, axis=3)
    cnn_pred_patches = get_padded_patches(
        cnn_pred_img, patch_size=1, pad_size=3)

    test_svm_feat = np.reshape(
        cnn_pred_patches, (cnn_pred_patches.shape[0], -1))

    svm_pred = svm.predict(test_svm_feat)
    svm_labels[i * labels_per_image:(i + 1) *
               labels_per_image] = np.expand_dims(svm_pred, axis=1)

# Transform labels to images
test_pred = labels_to_images(svm_labels, test_height, test_width, patch_size)

# Store predicted masks to files
for i, sample in enumerate(sorted(test_images_folder.iterdir())):
    io.imsave(predictions_folder / sample.name, test_pred[i])

# Create submission file
pred_files = []
for file in sorted(predictions_folder.iterdir(), key=lambda x: int(str(x).split('_')[-1].split('.')[0])):
    pred_files.append(str(file))
masks_to_submission(submission_file, *pred_files)
