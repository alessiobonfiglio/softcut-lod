from math import sqrt

import torch
from segmentation_models_pytorch.encoders import get_preprocessing_params as get_segmentation_model_preprocessing_params
from torch import nn


def get_monitor_mode(x):
    if "loss" in x:
        return "min"
    elif "accuracy" in x:
        return "max"
    else:
        raise ValueError("Don't know how to monitor the loss {}".format(x))


def get_min_hw(dataset):
    img, _ = dataset.__getitem__(0)
    min_h, min_w = img.shape[-2:]

    for i in range(1, dataset.__len__()):
        img, _ = dataset.__getitem__(i)
        new_h, new_w = img.shape[-2:]

        if new_h < min_h:
            min_h = new_h
        if new_w < min_w:
            min_w = new_w

    return min_h, min_w


class SubsetWithTransformations:
    def __init__(self, subset, transformations):
        self.subset = subset
        self.transformations = transformations

    def __getitem__(self, idx):
        x, y = self.subset.__getitem__(idx)
        x = self.transformations(x)
        y = self.transformations(y)
        return x, y

    def __len__(self):
        return self.subset.__len__()


def get_preprocessing_params(encoder, encoder_weights):
    params = get_segmentation_model_preprocessing_params(encoder, pretrained=encoder_weights)
    ret = {}
    for k in params.keys():
        if k == 'mean':
            ret['mean'] = [params['mean'][0], params['mean'][1], params['mean'][2]]
        if k == 'std':
            ret['std'] = [params['std'][0], params['std'][1], params['std'][2]]
    return ret


# Binary IoU with logits computation
class BIoUWithLogits(nn.Module):
    def __init__(self, threshold, eps=1e-7):
        super(BIoUWithLogits, self).__init__()
        self.threshold = threshold
        self.eps = eps

    def forward(self, predictions, targets):
        # compute logits
        predictions = torch.sigmoid(predictions)
        # threshold it
        predictions = predictions > self.threshold

        # compute IoU
        intersection = torch.logical_and(predictions, targets).sum()
        union = torch.logical_or(predictions, targets).sum()
        IoU = (intersection + self.eps) / (union + self.eps)

        return IoU

class BIoU(nn.Module):
    def __init__(self, threshold, eps=1e-7):
        super(BIoU, self).__init__()
        self.threshold = threshold
        self.eps = eps

    def forward(self, predictions, targets):
        # threshold the prediction
        predictions = predictions > self.threshold

        # compute IoU
        intersection = torch.logical_and(predictions, targets).sum()
        union = torch.logical_or(predictions, targets).sum()
        IoU = (intersection + self.eps) / (union + self.eps)

        return IoU


def get_class_labels_count(ds):
    # must disable augmentation
    aug = ds.augmentation
    ds.augmentation = None
    pos_count = 0
    neg_count = 0
    tot = 0
    for i in ds:
        _, y = i
        pos_tmp = y.sum().int().item()
        tot_tmp = torch.numel(y)
        pos_count += pos_tmp
        neg_count += tot_tmp - pos_tmp
        tot += tot_tmp
    # re-enable augmentation
    ds.augmentation = aug
    return pos_count, neg_count, tot


def weighted_mean(elements, weights):
    dot = 0
    tot = 0
    for val, w in zip(elements, weights):
        dot += val * w
        tot += w
    if tot != 0.0:
        return dot / tot
    else:
        return 0.0


def weighted_std(elements, weights, mean):
    numerator = 0
    non_zero_weights = 0
    tot = 0
    for val, w in zip(elements, weights):
        tmp = val - mean
        numerator += w * tmp * tmp
        tot += w
        if w != 0.0:
            non_zero_weights += 1
    return sqrt(numerator / (((non_zero_weights - 1) / non_zero_weights) * tot))


def std_for_list_of_tensors(elements):
    mean = 0
    numerator = 0
    for i in elements:
        mean += i
    mean /= len(elements)
    for i in elements:
        tmp = i - mean
        numerator += tmp * tmp
    return sqrt(numerator / (len(elements) - 1))
