import numpy as np
from utils import device


class ConfusionMatrix(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.zero()

    def update(self, labels_true, labels_pred):
        def _hist(l_true, l_pred):
            mask = (l_true >= 0) & (l_true < self.n_classes)
            hist = np.bincount(
                self.n_classes * l_true[mask].astype(int) + l_pred[mask],
                minlength=self.n_classes**2).reshape(self.n_classes,
                                                     self.n_classes)
            return hist

        _labels_true, _labels_pred = device([labels_true, labels_pred],
                                            gpu=False,
                                            numpy=True)
        for lt, lp in zip(_labels_true, _labels_pred):
            self.confusion_matrix += _hist(lt.flatten(), lp.flatten())

    def __call__(self):
        hist = self.confusion_matrix
        accuracy = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) -
                              np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))
        return (
            {
                "Overall Accuracy:": accuracy,
                "Mean Accuracy:": acc_cls,
                "FreqW Accuracy:": fwavacc,
                "Mean IoU:": mean_iu,
            },
            cls_iu,
        )

    def zero(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
