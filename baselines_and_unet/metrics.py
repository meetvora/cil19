import sklearn.metrics as sklm
import numpy as np

from keras import backend as K


def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)


def weighted_binary_crossentropy(weights):
    def weighted_binary_crossentropy_func(y_true, y_pred):
        # Calculate the binary crossentropy
        b_ce = K.binary_crossentropy(y_true, y_pred)

        # Apply the weights
        weight_vector = (1. - y_true) * weights[0] + y_true * weights[1]
        weighted_b_ce = weight_vector * b_ce

        # Return the mean error
        return K.mean(weighted_b_ce)

    return weighted_binary_crossentropy_func


def prec_rec_f1(prediction, labels):
    predict = np.round(prediction).astype(bool)
    target = labels.astype(bool)

    precision = sklm.precision_score(target, predict)
    recall = sklm.recall_score(target, predict)
    f1 = sklm.f1_score(target, predict)

    return precision, recall, f1
