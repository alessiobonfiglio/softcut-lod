import torch
from segmentation_models_pytorch.losses import DiceLoss
from torch.nn import BCEWithLogitsLoss


def get_loss_function_by_name(name, from_logits=True):
    if name == "binary_cross_entropy_with_logits":
        return BCEWithLogitsLoss(reduction='none')
    if name == "dice_loss":
        def loss(y_hat, y, ignore_index=None):
            y_hat = torch.unsqueeze(y_hat, 1)
            return DiceLoss('binary', from_logits=from_logits, ignore_index=ignore_index)(y_hat, y)

        return loss
    if name == "focal_dice_loss":
        def dice_loss(input, target):
            smooth = 1e-7
            
            if from_logits:
                input = torch.sigmoid(input)

            iflat = input.view(-1)
            iflat2 = iflat*iflat
            iflat_in = 1 - iflat
            iflat = iflat2/(iflat2 + iflat_in*iflat_in)
            
            tflat = target.view(-1)
            intersection = (iflat * tflat).sum()

            return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))
        
        return dice_loss

    raise ValueError("Loss function {} not found".format(name))


def loss_support_mask(name):
    if name == "binary_cross_entropy_with_logits":
        return True
    if name == "dice_loss":
        return False
    if name == "focal_dice_loss":
        return False
    raise ValueError("Loss function {} not found".format(name))


def loss_support_ignore_index(name):
    if name == "binary_cross_entropy_with_logits":
        return False
    if name == "dice_loss":
        return True
    if name == "focal_dice_loss":
        return False
    raise ValueError("Loss function {} not found".format(name))
