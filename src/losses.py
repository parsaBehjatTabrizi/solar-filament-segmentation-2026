"""Training loss: binary cross-entropy + soft Dice (computed in float32, also under mixed precision)."""
import torch
import torch.nn.functional as F


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    logits = logits.float()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(2, 3))
    dice = (2 * inter + eps) / (prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps)
    return bce + (1 - dice).mean()
