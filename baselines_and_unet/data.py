import numpy as np

from skimage import io
from skimage.transform import rotate


def load_images(dir, verbose=False):
    """
    Loads the images. The output dimensions are: (nr_images, height, width, 3).
    """
    data = list()
    for file in sorted(dir.iterdir()):
        if verbose:
            print(file)
        img = io.imread(file)
        img = img.astype('float32')
        data.append(img)

    return np.asarray(data)


def load_masks(dir, verbose=False):
    """
    Loads grayscale masks. The output dimensions are: (nr_masks, height, width, 1).
    """
    data = list()
    for file in sorted(dir.iterdir()):
        if verbose:
            print(file)
        img = io.imread(file, as_gray=True)
        img = img.astype('float32')
        data.append(img)

    masks = np.asarray(data)
    masks = np.expand_dims(masks, axis=3)

    return masks


def patch_to_label(patch, threshold=0.25):
    """
    Label the patch according to the specified threshold.
    """
    df = np.mean(patch)
    if df > threshold:
        return 1
    else:
        return 0


def crop_images(img, mask, new_height=224, new_width=224):
    """
    Crops the image into 4 pieces (top-left, top-right, bottom-left, bottom-right)
    of size (new_height, new_width) each.
    """
    n, height, width, img_dim = img.shape
    mask_dim = mask.shape[3]

    new_img = np.zeros((n * 4, new_height, new_width, img_dim,), dtype='float32')
    new_mask = np.zeros((n * 4, new_height, new_width, mask_dim,), dtype='float32')

    for i in range(n):
        new_img[4 * i] = img[i, :new_height, :new_width]
        new_img[4 * i + 1] = img[i, :new_height, width-new_width:]
        new_img[4 * i + 2] = img[i, height-new_height:, :new_width]
        new_img[4 * i + 3] = img[i, height-new_height:, width-new_width:]

        new_mask[4 * i] = mask[i, :new_height, :new_width]
        new_mask[4 * i + 1] = mask[i, :new_height, width-new_width:]
        new_mask[4 * i + 2] = mask[i, height-new_height:, :new_width]
        new_mask[4 * i + 3] = mask[i, height-new_height:, width-new_width:]

    return new_img, new_mask


def rotate_images(img, angles=None):
    """
    Rotates each image by 90, 180, 270 and returns the original images + the
    rotated ones.
    """
    if angles is None:
        angles = [0, 90, 180, 270]
    nr_angles = len(angles)

    n = img.shape[0]

    new_img = np.zeros((n * nr_angles,) + img.shape[1:], dtype='float32')

    for i in range(n):
        for j in range(nr_angles):
            new_img[nr_angles * i + j] = rotate(img[i], angles[j])

    return new_img


def get_patches(images, patch_size=16, step_size=None):
    step_size = step_size if step_size is not None else patch_size

    n, height, width, _ = images.shape

    patches = []
    for i in range(n):
        image = images[i]

        h = 0
        while h + patch_size <= height:
            w = 0
            while w + patch_size <= width:
                patches.append(image[h:h + patch_size, w:w + patch_size])
                w += step_size
            h += step_size

    return np.asarray(patches)


def get_padded_patches(images, patch_size=16, pad_size=None):
    pad_size = pad_size if pad_size is not None else patch_size

    padded_images = np.pad(
        images, ((0, 0), (pad_size, pad_size), (pad_size, pad_size), (0, 0)), 'constant')

    patches = get_patches(
        padded_images, patch_size=patch_size + 2 * pad_size, step_size=patch_size)

    return patches


def get_patches_labels(images, masks, patch_size=16, pad_size=None, true_threshold=0.25):
    n, height, width, _ = images.shape

    if pad_size is None:
        patches = get_patches(images, patch_size=patch_size)
    else:
        patches = get_padded_patches(
            images, patch_size=patch_size, pad_size=pad_size)

    mask_patches = get_patches(masks, patch_size=patch_size)

    labels = np.zeros(len(mask_patches), dtype='float32')
    for i in range(len(mask_patches)):
        labels[i] = patch_to_label(mask_patches[i], true_threshold)

    return patches, labels


def labels_to_images(labels, height, width, patch_size=16, threshold=0.5):
    images = []

    if threshold is not None:
        labels = labels > threshold

    idx = 0
    while idx < len(labels):
        image = np.zeros((height, width), dtype='float32')

        for h in range(0, height, patch_size):
            for w in range(0, width, patch_size):
                image[h:h + patch_size, w:w + patch_size] = labels[idx]
                idx += 1

        images.append(image)

    return np.asarray(images)
