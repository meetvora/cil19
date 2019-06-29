from typing import List

import ipdb
from torch import nn

from models import deeplab
from models import fcn


class Ensemble(nn.Module):
    def __init__(self, weights: List[int] = [5, 1]):
        super().__init__()
        self.dlab = deeplab.get_model(pretrained=True)
        self.fc = fcn.get_model(pretrained=True)
        self.weights = weights

    def forward(self, x):
        return {
            'out':
            self.weights[0] * self.fc(x)['out'] +
            self.weights[1] * self.dlab(x)['out']
        }
