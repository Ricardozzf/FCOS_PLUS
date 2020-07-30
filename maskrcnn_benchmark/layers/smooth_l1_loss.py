# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch


# TODO maybe push this to nn?
def smooth_l1_loss(input, target, beta=1. / 9, size_average=True, weight=None):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    c = 1
    if input.shape[1] != target.shape[1]:
        c = target[:, -1]
        target = target[:,:-1]
    n = torch.abs(input - target) / c.unsqueeze(1)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)

    if weight is not None:
        # many target have no offset ann
        weight_label = (target!=0).float().sum(1).clamp(0,1)
        weight_label = weight_label[:,None].repeat(1,2)
        weight = weight[:,None].repeat(1,2)
        loss = loss * weight * weight_label
    if size_average:
        return loss.mean()
    return loss.sum()
