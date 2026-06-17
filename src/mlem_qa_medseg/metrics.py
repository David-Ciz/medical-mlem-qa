from __future__ import annotations

import torch


def dice_from_logits(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    pred = (torch.sigmoid(logits) >= threshold).float()
    return dice_score(pred, target)


def dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    pred = pred.float().flatten(1)
    target = target.float().flatten(1)
    intersection = (pred * target).sum(dim=1)
    denominator = pred.sum(dim=1) + target.sum(dim=1)
    return (2.0 * intersection + eps) / (denominator + eps)


def iou_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    pred = pred.float().flatten(1)
    target = target.float().flatten(1)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - intersection
    return (intersection + eps) / (union + eps)


def precision_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    pred = pred.float().flatten(1)
    target = target.float().flatten(1)
    tp = (pred * target).sum(dim=1)
    fp = (pred * (1 - target)).sum(dim=1)
    return (tp + eps) / (tp + fp + eps)


def recall_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    pred = pred.float().flatten(1)
    target = target.float().flatten(1)
    tp = (pred * target).sum(dim=1)
    fn = ((1 - pred) * target).sum(dim=1)
    return (tp + eps) / (tp + fn + eps)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    probs = torch.sigmoid(logits).flatten(1)
    target_flat = target.float().flatten(1)
    intersection = (probs * target_flat).sum(dim=1)
    denominator = probs.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()

