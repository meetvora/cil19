import torch
import torch.nn.functional as F
import ipdb


class Loss(object):
    implemented = ["cross_entropy"]

    @staticmethod
    def cross_entropy2D(input, target):
        target = torch.squeeze(target)
        (N, C, H, W) = input.size()
        (Nt, Ht, Wt) = target.size()

        if (H, W) != (Ht, Wt):
            input = F.interpolate(input,
                                  size=(Ht, Wt),
                                  mode="bilinear",
                                  align_corners=True)
        _input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, C)
        _target = target.view(-1)
        return F.cross_entropy(_input, _target, size_average=True, ignore_index=250)
