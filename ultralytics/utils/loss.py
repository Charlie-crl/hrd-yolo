# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import OKS_SIGMA, RLE_WEIGHT
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.freqrrr import (
    box_area_sqrt,
    build_tiny_gaussian_masks,
    build_tiny_mask_and_count,
    clear_freqrrr_aux,
    compute_scale_alpha,
    focal_bce_with_logits,
    get_stride_key,
    normalized_wasserstein_similarity,
    parse_freqrrr_cfg,
)
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import autocast

from .metrics import bbox_iou, probiou
from .tal import bbox2dist, rbox2dist

TINY_LEARNING_DEFAULTS = {
    "enabled": True,
    "use_score_reweight": True,
    "use_loc_calibration": True,
    "tiny_size": 32.0,
    "focus_upper": 64.0,
    "gaussian_eta": 0.50,
    "gaussian_floor": 1.00,
    "gaussian_gamma": 1.00,
    "score_power": 0.50,
    "score_max": 2.50,
    "loc_alpha": 0.50,
    "loc_max": 2.00,
    "center_lambda": 0.25,
    "shape_lambda": 0.15,
    "warmup_iters": 1000,
    "eps": 1.0e-9,
}


def _parse_tiny_learning_cfg(tiny_learning: dict[str, Any] | None) -> dict[str, Any]:
    """Merge a flat tiny_learning config dict with defaults."""
    cfg = dict(TINY_LEARNING_DEFAULTS)
    if tiny_learning:
        cfg.update(tiny_learning)
    for key in ("enabled", "use_score_reweight", "use_loc_calibration"):
        cfg[key] = bool(cfg[key])
    cfg["warmup_iters"] = int(cfg["warmup_iters"])
    for key in (
        "tiny_size",
        "focus_upper",
        "gaussian_eta",
        "gaussian_floor",
        "gaussian_gamma",
        "score_power",
        "score_max",
        "loc_alpha",
        "loc_max",
        "center_lambda",
        "shape_lambda",
        "eps",
    ):
        cfg[key] = float(cfg[key])
    return cfg


def _xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert xyxy boxes to cxcywh boxes."""
    ctr = (boxes[..., :2] + boxes[..., 2:4]) * 0.5
    wh = _box_wh(boxes)
    return torch.cat((ctr, wh), dim=-1)


def _box_wh(boxes_xyxy: torch.Tensor) -> torch.Tensor:
    """Return width and height from xyxy boxes."""
    return (boxes_xyxy[..., 2:4] - boxes_xyxy[..., 0:2]).clamp_min(0)


def _box_area(boxes_xyxy: torch.Tensor) -> torch.Tensor:
    """Return area from xyxy boxes."""
    return _box_wh(boxes_xyxy).prod(-1)


def _warmup_progress(step: int, warmup_iters: int) -> float:
    """Return warmup progress in [0, 1]."""
    if warmup_iters <= 0:
        return 1.0
    return max(min(step / warmup_iters, 1.0), 0.0)


def _groupwise_normalize_by_gt(values: torch.Tensor, fg_mask: torch.Tensor, target_gt_idx: torch.Tensor) -> torch.Tensor:
    """Normalize positive values by the maximum value within each matched-GT group."""
    out = torch.zeros_like(values)
    if values.numel() == 0:
        return out
    eps = torch.finfo(values.dtype).eps if values.is_floating_point() else 1e-9
    for batch_idx in range(values.shape[0]):
        fg_mask_i = fg_mask[batch_idx]
        if not fg_mask_i.any().item():
            continue
        gt_idx_i = target_gt_idx[batch_idx].long()
        for gt_idx in gt_idx_i[fg_mask_i].unique():
            gt_mask = fg_mask_i & gt_idx_i.eq(gt_idx)
            max_value = values[batch_idx, gt_mask].max()
            out[batch_idx, gt_mask] = values[batch_idx, gt_mask] / (max_value + eps)
    return out


class VarifocalLoss(nn.Module):
    """Varifocal loss by Zhang et al.

    Implements the Varifocal Loss function for addressing class imbalance in object detection by focusing on
    hard-to-classify examples and balancing positive/negative samples.

    Attributes:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (float): The balancing factor used to address class imbalance.

    References:
        https://arxiv.org/abs/2008.13367
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        """Initialize the VarifocalLoss class with focusing and balancing parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred_score: torch.Tensor, gt_score: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Compute varifocal loss between predictions and ground truth."""
        weight = self.alpha * pred_score.sigmoid().pow(self.gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5).

    Implements the Focal Loss function for addressing class imbalance by down-weighting easy examples and focusing on
    hard negatives during training.

    Attributes:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (torch.Tensor): The balancing factor used to address class imbalance.
    """

    def __init__(self, gamma: float = 1.5, alpha: float = 0.25):
        """Initialize FocalLoss class with focusing and balancing parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha)

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Calculate focal loss with modulating factors for class imbalance."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= modulating_factor
        if (self.alpha > 0).any():
            self.alpha = self.alpha.to(device=pred.device, dtype=pred.dtype)
            alpha_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class DFLoss(nn.Module):
    """Criterion class for computing Distribution Focal Loss (DFL)."""

    def __init__(self, reg_max: int = 16) -> None:
        """Initialize the DFL module with regularization maximum."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return sum of left and right DFL losses from https://ieeexplore.ieee.org/document/9792391."""
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """Criterion class for computing training losses for bounding boxes."""

    def __init__(self, reg_max: int = 16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute IoU and DFL losses for bounding boxes."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            # normalize ltrb by image size
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl


class TinyObjectBboxLoss(BboxLoss):
    """BBox loss with training-only SCL weighting for tiny-object learning."""

    def __init__(self, reg_max: int, tiny_cfg: dict[str, Any]):
        """Initialize TinyObjectBboxLoss with the merged tiny_learning config."""
        super().__init__(reg_max)
        self.tiny_cfg = tiny_cfg
        self.warmup_progress = 1.0

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
        target_bboxes_px: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the official bbox loss with optional SCL calibration for positives only."""
        if (
            not self.tiny_cfg["enabled"]
            or not self.tiny_cfg["use_loc_calibration"]
            or not fg_mask.any().item()
        ):
            return super().forward(
                pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, imgsz, stride
            )

        eps = self.tiny_cfg["eps"]
        base_weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        stride_xyxy = stride.view(1, -1, 1)
        target_bboxes_px = target_bboxes * stride_xyxy if target_bboxes_px is None else target_bboxes_px
        pred_bboxes_px = pred_bboxes * stride_xyxy

        target_bboxes_px_pos = target_bboxes_px[fg_mask].to(pred_bboxes.dtype)
        pred_bboxes_px_pos = pred_bboxes_px[fg_mask]
        size_px = _box_area(target_bboxes_px_pos).clamp_min(eps).sqrt()

        loc_raw = torch.ones_like(size_px)
        tiny_mask = size_px <= self.tiny_cfg["focus_upper"]
        if tiny_mask.any().item():
            loc_raw[tiny_mask] = (
                self.tiny_cfg["tiny_size"] / (size_px[tiny_mask] + eps)
            ).pow(self.tiny_cfg["loc_alpha"]).clamp_(1.0, self.tiny_cfg["loc_max"])
        loc_scale = 1.0 + self.warmup_progress * (loc_raw - 1.0)
        final_weight = base_weight * loc_scale.unsqueeze(-1)

        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        pred_ctr = 0.5 * (pred_bboxes_px_pos[..., :2] + pred_bboxes_px_pos[..., 2:4])
        tgt_ctr = 0.5 * (target_bboxes_px_pos[..., :2] + target_bboxes_px_pos[..., 2:4])
        tgt_wh = _box_wh(target_bboxes_px_pos).clamp_min(eps)
        pred_wh = _box_wh(pred_bboxes_px_pos).clamp_min(eps)
        center_err = (pred_ctr - tgt_ctr).abs().sum(-1, keepdim=True) / (
            (tgt_wh[..., 0:1] * tgt_wh[..., 1:2]).clamp_min(eps).sqrt() + eps
        )
        shape_err = ((pred_wh + eps) / (tgt_wh + eps)).log().abs().sum(-1, keepdim=True)

        loss_iou = (
            (
                (1.0 - iou)
                + self.tiny_cfg["center_lambda"] * center_err
                + self.tiny_cfg["shape_lambda"] * shape_err
            )
            * final_weight
        ).sum() / target_scores_sum

        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = (
                self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * final_weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True)
                * final_weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl


class RLELoss(nn.Module):
    """Residual Log-Likelihood Estimation Loss.

    Attributes:
        size_average (bool): Option to average the loss by the batch_size.
        use_target_weight (bool): Option to use weighted loss.
        residual (bool): Option to add L1 loss and let the flow learn the residual error distribution.

    References:
        https://arxiv.org/abs/2107.11291
        https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/losses/regression_loss.py
    """

    def __init__(self, use_target_weight: bool = True, size_average: bool = True, residual: bool = True):
        """Initialize RLELoss with target weight and residual options.

        Args:
            use_target_weight (bool): Whether to use target weights for loss calculation.
            size_average (bool): Whether to average the loss over elements.
            residual (bool): Whether to include residual log-likelihood term.
        """
        super().__init__()
        self.size_average = size_average
        self.use_target_weight = use_target_weight
        self.residual = residual

    def forward(
        self, sigma: torch.Tensor, log_phi: torch.Tensor, error: torch.Tensor, target_weight: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            sigma (torch.Tensor): Output sigma, shape (N, D).
            log_phi (torch.Tensor): Output log_phi, shape (N).
            error (torch.Tensor): Error, shape (N, D).
            target_weight (torch.Tensor): Weights across different joint types, shape (N).
        """
        log_sigma = torch.log(sigma)
        loss = log_sigma - log_phi.unsqueeze(1)

        if self.residual:
            loss += torch.log(sigma * 2) + torch.abs(error)

        if self.use_target_weight:
            assert target_weight is not None, "'target_weight' should not be None when 'use_target_weight' is True."
            if target_weight.dim() == 1:
                target_weight = target_weight.unsqueeze(1)
            loss *= target_weight

        if self.size_average:
            loss /= len(loss)

        return loss.sum()


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses for rotated bounding boxes."""

    def __init__(self, reg_max: int):
        """Initialize the RotatedBboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max)

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute IoU and DFL losses for rotated bounding boxes."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = rbox2dist(
                target_bboxes[..., :4], anchor_points, target_bboxes[..., 4:5], reg_max=self.dfl_loss.reg_max - 1
            )
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = rbox2dist(target_bboxes[..., :4], anchor_points, target_bboxes[..., 4:5])
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl


class MultiChannelDiceLoss(nn.Module):
    """Criterion class for computing multi-channel Dice losses."""

    def __init__(self, smooth: float = 1e-6, reduction: str = "mean"):
        """Initialize MultiChannelDiceLoss with smoothing and reduction options.

        Args:
            smooth (float): Smoothing factor to avoid division by zero.
            reduction (str): Reduction method ('mean', 'sum', or 'none').
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate multi-channel Dice loss between predictions and targets."""
        assert pred.size() == target.size(), "the size of predict and target must be equal."

        pred = pred.sigmoid()
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice
        dice_loss = dice_loss.mean(dim=1)

        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        else:
            return dice_loss


class BCEDiceLoss(nn.Module):
    """Criterion class for computing combined BCE and Dice losses."""

    def __init__(self, weight_bce: float = 0.5, weight_dice: float = 0.5):
        """Initialize BCEDiceLoss with BCE and Dice weight factors.

        Args:
            weight_bce (float): Weight factor for BCE loss component.
            weight_dice (float): Weight factor for Dice loss component.
        """
        super().__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = MultiChannelDiceLoss(smooth=1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate combined BCE and Dice loss between predictions and targets."""
        _, _, mask_h, mask_w = pred.shape
        if tuple(target.shape[-2:]) != (mask_h, mask_w):  # downsample to the same size as pred
            target = F.interpolate(target, (mask_h, mask_w), mode="nearest")
        return self.weight_bce * self.bce(pred, target) + self.weight_dice * self.dice(pred, target)


class KeypointLoss(nn.Module):
    """Criterion class for computing keypoint losses."""

    def __init__(self, sigmas: torch.Tensor) -> None:
        """Initialize the KeypointLoss class with keypoint sigmas."""
        super().__init__()
        self.sigmas = sigmas

    def forward(
        self, pred_kpts: torch.Tensor, gt_kpts: torch.Tensor, kpt_mask: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """Calculate keypoint loss factor and Euclidean distance loss for keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class v8DetectionLoss:
    """Criterion class for computing training losses for YOLOv8 object detection."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):  # model must be de-paralleled
        """Initialize v8DetectionLoss with model parameters and task-aligned assignment settings."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(
            topk=tal_topk,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,
            stride=self.stride.tolist(),
            topk2=tal_topk2,
        )
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets by converting to tensor format and scaling coordinates."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            batch_idx = targets[:, 0].long()  # image index
            _, counts = batch_idx.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=self.device)
            offsets.scatter_add_(0, batch_idx + 1, torch.ones_like(batch_idx))
            offsets = offsets.cumsum(0)
            within_idx = torch.arange(nl, device=self.device) - offsets[batch_idx]
            out[batch_idx, within_idx] = targets[:, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size and return foreground mask and
        target indices.
        """
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )  # loss(box, cls, dfl)

    def parse_output(
        self, preds: dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Parse model predictions to extract features."""
        return preds[1] if isinstance(preds, tuple) else preds

    def __call__(
        self,
        preds: dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        return self.loss(self.parse_output(preds), batch)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate detection loss using assigned targets."""
        batch_size = preds["boxes"].shape[0]
        loss, loss_detach = self.get_assigned_targets_and_loss(preds, batch)[1:]
        return loss * batch_size, loss_detach


class TinyObjectDetectionLoss(v8DetectionLoss):
    """Training-only TLR/SCL detection loss built on the active YOLO26 detection path."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None, shared_state: dict[str, int] | None = None):
        """Initialize TinyObjectDetectionLoss from the active v8DetectionLoss path."""
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        self.tiny_cfg = _parse_tiny_learning_cfg(getattr(model, "tiny_learning_cfg", {}))
        self.shared_state = shared_state
        self.local_steps = 0
        self.warmup_progress = 1.0
        self.bbox_loss = TinyObjectBboxLoss(self.reg_max, self.tiny_cfg).to(self.device)

    def _current_step(self) -> int:
        """Return the shared or local batch step used by warmup."""
        if self.shared_state is not None:
            return int(self.shared_state.get("step", 0))
        return self.local_steps

    def build_positive_reweight(
        self,
        target_bboxes_px: torch.Tensor,
        target_gt_idx: torch.Tensor,
        fg_mask: torch.Tensor,
        anchor_points: torch.Tensor,
        stride_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Build TLR reweight factors for positives after official assignment."""
        out = torch.ones(target_gt_idx.shape, device=target_bboxes_px.device, dtype=target_bboxes_px.dtype)
        if (
            not self.tiny_cfg["enabled"]
            or not self.tiny_cfg["use_score_reweight"]
            or not fg_mask.any().item()
        ):
            return out

        eps = self.tiny_cfg["eps"]
        anchors_px = (anchor_points * stride_tensor).unsqueeze(0).expand(target_bboxes_px.shape[0], -1, -1)
        target_cxcywh_px = _xyxy_to_cxcywh(target_bboxes_px)
        gt_centers = target_cxcywh_px[..., :2]
        gt_wh = target_cxcywh_px[..., 2:4].clamp_min(eps)
        size_px = (gt_wh[..., 0] * gt_wh[..., 1]).clamp_min(eps).sqrt()
        stride_px = stride_tensor.squeeze(-1).unsqueeze(0).expand_as(size_px)

        sigma_x = torch.maximum(self.tiny_cfg["gaussian_eta"] * gt_wh[..., 0], self.tiny_cfg["gaussian_floor"] * stride_px)
        sigma_y = torch.maximum(self.tiny_cfg["gaussian_eta"] * gt_wh[..., 1], self.tiny_cfg["gaussian_floor"] * stride_px)
        delta = anchors_px - gt_centers
        prior = torch.exp(
            -0.5 * ((delta[..., 0] / (sigma_x + eps)).pow(2) + (delta[..., 1] / (sigma_y + eps)).pow(2))
        )
        prior = prior * fg_mask.to(prior.dtype)
        prior_norm = _groupwise_normalize_by_gt(prior, fg_mask, target_gt_idx)

        boost = torch.ones_like(size_px)
        tiny_mask = fg_mask & size_px.le(self.tiny_cfg["focus_upper"])
        if tiny_mask.any().item():
            boost[tiny_mask] = (
                self.tiny_cfg["tiny_size"] / (size_px[tiny_mask] + eps)
            ).pow(self.tiny_cfg["score_power"]).clamp_(1.0, self.tiny_cfg["score_max"])

        q_raw = (
            prior_norm.clamp_min(eps).pow(self.tiny_cfg["gaussian_gamma"]) * boost
        ).clamp_(1.0, self.tiny_cfg["score_max"])
        q = 1.0 + self.warmup_progress * (q_raw - 1.0)
        out[fg_mask] = q[fg_mask]
        return out

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple:
        """Run the official assigner, then inject training-only TLR and SCL weighting."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_bboxes_px = target_bboxes.clone()
        self.warmup_progress = _warmup_progress(self._current_step(), self.tiny_cfg["warmup_iters"])
        self.bbox_loss.warmup_progress = self.warmup_progress

        pos_reweight = self.build_positive_reweight(target_bboxes_px, target_gt_idx, fg_mask, anchor_points, stride_tensor)
        if self.tiny_cfg["use_score_reweight"]:
            target_scores = target_scores * pos_reweight.unsqueeze(-1).to(target_scores.dtype)

        target_scores_sum = target_scores.sum().clamp_min(1.0)
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if fg_mask.any().item():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
                target_bboxes_px=target_bboxes_px,
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate detection loss while advancing a local warmup step outside end-to-end mode."""
        if self.shared_state is None:
            self.local_steps += 1
        batch_size = preds["boxes"].shape[0]
        loss, loss_detach = self.get_assigned_targets_and_loss(preds, batch)[1:]
        return loss * batch_size, loss_detach


class ScaleFrequencyBboxLoss(BboxLoss):
    """Scale-frequency calibrated box loss with IoU/NWD mixing."""

    def __init__(self, reg_max: int, sfqa_cfg: dict[str, Any]):
        """Initialize the SFQA box loss with the parsed config."""
        super().__init__(reg_max)
        self.sfqa_cfg = sfqa_cfg
        self.last_stats = {
            "loss_box_iou_part": 0.0,
            "loss_box_nwd_part": 0.0,
            "loss_box_scale_weight_mean": 0.0,
            "alpha_tiny_mean": 0.0,
            "alpha_all_mean": 0.0,
        }

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute scale-calibrated IoU/NWD mixed box loss while keeping DFL unchanged."""
        if not fg_mask.any():
            self.last_stats = {k: 0.0 for k in self.last_stats}
            return pred_bboxes.sum() * 0.0, pred_dist.sum() * 0.0

        eps = 1e-9
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        stride_xyxy = stride.view(1, -1, 1)
        pred_bboxes_px = pred_bboxes * stride_xyxy
        target_bboxes_px = target_bboxes * stride_xyxy

        gt_boxes_pos = target_bboxes_px[fg_mask]
        pred_boxes_pos = pred_bboxes_px[fg_mask]
        gt_sizes = box_area_sqrt(gt_boxes_pos, eps=eps)
        alpha_s = compute_scale_alpha(
            gt_sizes,
            self.sfqa_cfg.get("s0", 24.0),
            self.sfqa_cfg.get("gamma", 1.0),
            self.sfqa_cfg.get("alpha_max", 0.70),
        )
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True).clamp(0.0, 1.0)
        nwd = normalized_wasserstein_similarity(
            pred_boxes_pos, gt_boxes_pos, nwd_c=self.sfqa_cfg.get("nwd_c", 12.8), eps=eps
        ).unsqueeze(-1)
        loss_iou_part = 1.0 - iou
        loss_nwd_part = 1.0 - nwd
        mixed = (1.0 - alpha_s.unsqueeze(-1)) * loss_iou_part + alpha_s.unsqueeze(-1) * loss_nwd_part

        scale_weight = (float(self.sfqa_cfg.get("s_ref", 32.0)) / (gt_sizes + eps)).pow(float(self.sfqa_cfg.get("mu", 0.5)))
        scale_weight = scale_weight.clamp(1.0, float(self.sfqa_cfg.get("w_max", 1.5))).unsqueeze(-1)
        weighted_mixed = mixed * scale_weight
        loss_box = (weighted_mixed * weight).sum() / target_scores_sum

        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        tiny_mask = gt_sizes < float(self.sfqa_cfg.get("tiny_thr", 32.0))
        self.last_stats = {
            "loss_box_iou_part": float(loss_iou_part.detach().mean().item()),
            "loss_box_nwd_part": float(loss_nwd_part.detach().mean().item()),
            "loss_box_scale_weight_mean": float(scale_weight.detach().mean().item()),
            "alpha_tiny_mean": float(alpha_s[tiny_mask].detach().mean().item()) if tiny_mask.any() else 0.0,
            "alpha_all_mean": float(alpha_s.detach().mean().item()),
        }
        return loss_box, loss_dfl


class FreqRRRDetectionLoss(v8DetectionLoss):
    """Phase 2 detection loss for FreqRRR with TGF supervision and FQA assignment ranking."""

    def __init__(
        self,
        model,
        tal_topk: int = 10,
        tal_topk2: int | None = None,
        include_gate_loss: bool = True,
    ):
        """Initialize the FreqRRR detection loss on top of the existing YOLO26 path."""
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        self.model_ref = model
        self.head = model.model[-1]
        self.include_gate_loss = include_gate_loss
        self.freqrrr_cfg = parse_freqrrr_cfg(getattr(model, "freqrrr_cfg", {}))
        self.fprb_cfg = self.freqrrr_cfg["fprb"]
        self.tgf_cfg = self.freqrrr_cfg["tgf"]
        self.fqa_cfg = self.freqrrr_cfg["fqa"]
        self.rr_gate_cfg = self.freqrrr_cfg["rr_gate"]
        self.tgf_detail_cfg = self.freqrrr_cfg["tgf_detail"]
        self.freq_gate_cfg = self.freqrrr_cfg["freq_gate"]
        assigner_fqa_cfg = dict(self.fqa_cfg)
        assigner_fqa_cfg["tiny_thr"] = float(self.freqrrr_cfg["tiny_thr"])
        self.assigner = TaskAlignedAssigner(
            topk=tal_topk,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,
            stride=self.stride.tolist(),
            topk2=tal_topk2,
            fqa_cfg=assigner_fqa_cfg,
        )
        self.loss_names = tuple(self._build_loss_names())

    def _build_loss_names(self) -> list[str]:
        """Create the flat list of train/val stats that should be logged."""
        names = ["box_loss", "cls_loss", "dfl_loss", "loss_gate"]
        for stride in sorted(set(self.fprb_cfg.get("strides", [8]) or [8])):
            names.extend(
                [
                    f"fprb_gain_stride{int(stride)}",
                    f"fprb_high_abs_mean_stride{int(stride)}",
                    f"fprb_res_abs_mean_stride{int(stride)}",
                ]
            )
        names.extend(
            [
                "gate_pos_mean",
                "gate_neg_mean",
                "num_tiny_gt",
                "fqa_lambda",
                "fqa_gate_mean",
                "fqa_boost_mean",
                "fqa_num_tiny_gt",
            ]
        )
        names.extend(
            [
                "rg_l",
                "rg_p",
                "rg_n",
                "rg_gap",
                "rg_gain",
                "rg_nt",
                "ta_l",
                "ta_p",
                "ta_n",
                "ta_num",
                "ta_sum",
                "ta_mean",
                "matched_target_bboxes_min",
                "matched_target_bboxes_max",
                "matched_area_mean",
                "fa_l",
                "fa_p",
                "fa_n",
                "fa_gap",
                "fa_e",
                "fa_c",
                "taf_l",
                "taf_p",
                "taf_n",
                "taf_gap",
                "dt_l",
                "df_l",
                "df_w",
                "df_p",
                "df_n",
                "df_gap",
                "df_num",
                "df_sum",
                "td_l",
                "td_p",
                "td_n",
                "td_gap",
                "td_g",
                "td_nt",
                "fg_l",
                "fg_p",
                "fg_n",
                "fg_gap",
                "fg_gain",
                "fg_nt",
                "fg_e",
                "fg_c",
            ]
        )
        return names

    def _init_log_stats(self) -> dict[str, float]:
        """Initialize all logged metrics to zero for clean fallback behavior."""
        return {name: 0.0 for name in self.loss_names}

    def _make_stats_tensor(self, stats: dict[str, float], dtype: torch.dtype) -> torch.Tensor:
        """Convert the flat stats dictionary to a tensor matching loss_names order."""
        return torch.tensor([float(stats[name]) for name in self.loss_names], device=self.device, dtype=dtype)

    def _get_head_aux(self) -> dict[str, dict]:
        """Return the latest FreqRRR auxiliary outputs collected by the detect head."""
        aux = getattr(self.head, "freqrrr_aux", None)
        return aux if isinstance(aux, dict) else {"fprb": {}, "tgf": {}}

    def _get_current_epoch(self) -> float:
        """Read the trainer-supplied epoch from the live model with a safe zero fallback."""
        current_epoch = getattr(self.model_ref, "current_epoch", 0)
        return float(current_epoch if current_epoch is not None else 0.0)

    def _build_assignment_gate(
        self, preds: dict[str, torch.Tensor], dtype: torch.dtype
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Flatten per-level TGF gates to the concatenated anchor order used by the assigner."""
        feats = preds.get("feats", [])
        if not feats or not self.fqa_cfg.get("enabled", False):
            return None, None

        batch_size = preds["boxes"].shape[0]
        aux = self._get_head_aux().get("tgf", {})
        gate_levels = []
        valid_levels = []
        enabled_tgf_strides = set(int(s) for s in self.tgf_cfg.get("strides", []))

        for feat, stride in zip(feats, self.stride.tolist()):
            h, w = feat.shape[-2:]
            hw = h * w
            stride = int(stride)
            gate_level = torch.zeros(batch_size, hw, device=self.device, dtype=dtype)
            gate_valid = torch.zeros(batch_size, hw, device=self.device, dtype=torch.bool)
            item = aux.get(get_stride_key(stride), {})
            gate_source = None
            if stride in enabled_tgf_strides and item:
                gate_source = item.get("gate_detached") if self.fqa_cfg.get("detach_gate", True) else item.get("gate")
                if gate_source is None:
                    gate_source = item.get("gate")
            if gate_source is not None:
                if self.fqa_cfg.get("detach_gate", True):
                    gate_source = gate_source.detach()
                gate_tensor = gate_source.to(device=self.device, dtype=dtype)
                if gate_tensor.ndim == 4 and gate_tensor.shape[1] == 1:
                    gate_tensor = gate_tensor.flatten(2).squeeze(1)
                elif gate_tensor.ndim == 3 and gate_tensor.shape[1] == 1:
                    gate_tensor = gate_tensor.squeeze(1)
                elif gate_tensor.ndim != 2:
                    gate_tensor = None
                if gate_tensor is not None and gate_tensor.shape == gate_level.shape:
                    gate_level = torch.nan_to_num(gate_tensor, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
                    gate_valid = torch.ones_like(gate_level, dtype=torch.bool)
            gate_levels.append(gate_level)
            valid_levels.append(gate_valid)

        if not gate_levels:
            return None, None
        return torch.cat(gate_levels, dim=1), torch.cat(valid_levels, dim=1)

    def _collect_fprb_stats(self, stats: dict[str, float]) -> None:
        """Collect optional FPRB monitoring stats from the detect head."""
        aux = self._get_head_aux().get("fprb", {})
        for stride in sorted(set(self.fprb_cfg.get("strides", [8]) or [8])):
            key = get_stride_key(stride)
            item = aux.get(key, {})
            stats[f"fprb_gain_stride{int(stride)}"] = float(item.get("gain", 0.0) or 0.0)
            stats[f"fprb_high_abs_mean_stride{int(stride)}"] = float(item.get("high_abs_mean", 0.0) or 0.0)
            stats[f"fprb_res_abs_mean_stride{int(stride)}"] = float(item.get("res_abs_mean", 0.0) or 0.0)

    def _collect_assigner_stats(self, stats: dict[str, float]) -> None:
        """Collect FQA-specific ranking stats exported by the assigner."""
        assigner_stats = getattr(self.assigner, "last_stats", {})
        stats["fqa_lambda"] = float(assigner_stats.get("fq_lam", 0.0) or 0.0)
        stats["fqa_gate_mean"] = float(assigner_stats.get("fq_g", 0.0) or 0.0)
        stats["fqa_boost_mean"] = float(assigner_stats.get("fq_bst", 0.0) or 0.0)
        stats["fqa_num_tiny_gt"] = float(assigner_stats.get("fq_tiny", 0.0) or 0.0)

    def _compute_gate_loss(
        self,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
        preds: dict[str, torch.Tensor],
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute tiny-guided gate supervision across all enabled TGF levels."""
        zero = preds["boxes"].sum() * 0.0
        stats = {"loss_gate": 0.0, "gate_pos_mean": 0.0, "gate_neg_mean": 0.0, "num_tiny_gt": 0.0}
        if not (self.include_gate_loss and self.tgf_cfg["enabled"]):
            return zero, stats

        aux = self._get_head_aux().get("tgf", {})
        if not aux:
            return zero, stats

        feature_shapes = {}
        for feat, stride in zip(preds["feats"], self.stride.tolist()):
            stride = int(stride)
            if stride in self.tgf_cfg["strides"]:
                feature_shapes[stride] = feat.shape[-2:]
        masks, num_tiny_gt = build_tiny_gaussian_masks(
            gt_bboxes,
            mask_gt,
            feature_shapes=feature_shapes,
            tiny_thr=self.freqrrr_cfg["tiny_thr"],
            sigma_scale=0.25,
            min_sigma=1.0,
            max_sigma=4.0,
            device=self.device,
            dtype=dtype,
        )
        if not masks:
            return zero, stats

        loss_terms = []
        pos_sum = 0.0
        pos_count = 0
        neg_sum = 0.0
        neg_count = 0
        for stride in sorted(masks):
            key = get_stride_key(stride)
            item = aux.get(key, None)
            if item is None or item.get("gate_logits") is None or item.get("gate") is None:
                continue
            target = masks[stride].to(device=self.device, dtype=item["gate_logits"].dtype)
            loss_terms.append(
                focal_bce_with_logits(
                    item["gate_logits"],
                    target,
                    alpha=0.75,
                    gamma=2.0,
                )
            )
            gate = item["gate"].detach()
            pos_mask = target > 0.5
            neg_mask = ~pos_mask
            if pos_mask.any():
                pos_sum += float(gate[pos_mask].mean().item())
                pos_count += 1
            if neg_mask.any():
                neg_sum += float(gate[neg_mask].mean().item())
                neg_count += 1

        if not loss_terms:
            return zero, stats

        gate_loss = torch.stack(loss_terms).mean()
        gate_loss = gate_loss * float(self.tgf_cfg["gate_loss_weight"])
        stats = {
            "loss_gate": float(gate_loss.detach().item()),
            "gate_pos_mean": pos_sum / max(pos_count, 1),
            "gate_neg_mean": neg_sum / max(neg_count, 1),
            "num_tiny_gt": float(num_tiny_gt),
        }
        return gate_loss, stats

    def _iter_aux_modules(self, logits_attr: str):
        """Yield modules that expose a specific RRFusion auxiliary logits attribute."""
        for module in self.model_ref.modules():
            if hasattr(module, logits_attr):
                yield module

    @staticmethod
    def _mean_module_float(modules: list[nn.Module], attr: str) -> float:
        """Return the mean of a float monitoring attribute across modules."""
        values = [float(getattr(module, attr, 0.0) or 0.0) for module in modules]
        return sum(values) / max(len(values), 1)

    def _sanitize_aux_loss(self, aux_loss: torch.Tensor, zero: torch.Tensor, name: str) -> torch.Tensor:
        """Return a finite auxiliary loss, warning and zeroing if NaN/Inf appears."""
        if not torch.isfinite(aux_loss.detach()).all().item():
            LOGGER.warning(f"{name} auxiliary loss produced NaN/Inf; zeroing this term.")
            return zero
        return aux_loss

    @staticmethod
    def _freq_aux_target_mode(cfg: dict[str, Any]) -> str:
        """Resolve the FreqAux target mode, preserving legacy inheritance by default."""
        mode = str(cfg.get("freq_aux_target_mode", "inherit")).lower()
        return str(cfg.get("target_mode", "tiny_mask")).lower() if mode == "inherit" else mode

    @staticmethod
    def _is_dual_target_freq_aux_cfg(cfg: dict[str, Any]) -> bool:
        """Return True for tiny-mask RRGate + task-aligned FreqAux supervision."""
        return (
            bool(cfg.get("enabled", False))
            and bool(cfg.get("use_freq_aux", False))
            and str(cfg.get("target_mode", "tiny_mask")).lower() == "tiny_mask"
            and FreqRRRDetectionLoss._freq_aux_target_mode(cfg) == "task_aligned"
        )

    def _scheduled_freq_aux_weight(self, cfg: dict[str, Any], current_epoch: float | None = None) -> float:
        """Return the FreqAux loss weight after start/warmup scheduling."""
        base_weight = float(cfg.get("freq_aux_loss_weight", cfg.get("loss_weight", 0.0)))
        start_epoch = float(cfg.get("freq_aux_start_epoch", 0))
        warmup_epochs = max(float(cfg.get("freq_aux_warmup_epochs", 0)), 0.0)
        epoch = self._get_current_epoch() if current_epoch is None else float(current_epoch)
        if epoch < start_epoch:
            scale = 0.0
        elif warmup_epochs <= 0.0:
            scale = 1.0
        else:
            scale = min(max((epoch - start_epoch) / warmup_epochs, 0.0), 1.0)
        return base_weight * scale

    def _select_p3_indices(self, stride_tensor: torch.Tensor, feature_hw: tuple[int, int]) -> torch.Tensor | None:
        """Return flattened assigner indices for the stride-8/P3 feature map."""
        h, w = int(feature_hw[0]), int(feature_hw[1])
        expected = h * w
        stride_flat = stride_tensor.reshape(-1)
        stride_match = torch.isclose(
            stride_flat.float(),
            torch.full_like(stride_flat.float(), 8.0),
            rtol=0.0,
            atol=1.0e-6,
        )
        if int(stride_match.sum().item()) == expected:
            return torch.where(stride_match)[0]
        if expected <= stride_flat.numel():
            return torch.arange(expected, device=stride_tensor.device)
        return None

    def _build_task_gate_target(
        self,
        logits: torch.Tensor,
        fg_mask: torch.Tensor | None,
        target_scores: torch.Tensor | None,
        target_bboxes: torch.Tensor | None,
        stride_tensor: torch.Tensor | None,
        tiny_only: bool,
        target_value_mode: str = "score",
    ) -> tuple[torch.Tensor, int, float, dict[str, float]]:
        """Project assigner positives to a detached [B, 1, H, W] stride-8 gate target."""
        batch_size, _, h, w = logits.shape
        target = logits.new_zeros(batch_size, 1, h, w)
        debug = {
            "matched_target_bboxes_min": 0.0,
            "matched_target_bboxes_max": 0.0,
            "matched_area_mean": 0.0,
            "matched_target_bboxes_count": 0.0,
        }
        if fg_mask is None or stride_tensor is None:
            return target.detach(), 0, 0.0, debug

        p3_indices = self._select_p3_indices(stride_tensor.to(device=logits.device), (h, w))
        if p3_indices is None or p3_indices.numel() != h * w:
            return target.detach(), 0, 0.0, debug

        fg = fg_mask.detach().to(device=logits.device).bool()
        if fg.ndim != 2 or fg.shape[0] != batch_size or fg.shape[1] <= int(p3_indices.max().item()):
            return target.detach(), 0, 0.0, debug
        p3_fg = fg[:, p3_indices]

        if target_value_mode == "binary":
            scores = torch.ones_like(p3_fg, dtype=logits.dtype)
        elif target_scores is not None and target_scores.ndim == 3 and target_scores.shape[:2] == fg.shape:
            scores = target_scores.detach().amax(dim=-1).to(device=logits.device, dtype=logits.dtype)[:, p3_indices]
            scores = torch.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        else:
            scores = torch.ones_like(p3_fg, dtype=logits.dtype)

        pos_mask = p3_fg
        if tiny_only and target_bboxes is not None and target_bboxes.ndim == 3 and target_bboxes.shape[:2] == fg.shape:
            boxes = target_bboxes.detach().to(device=logits.device, dtype=torch.float32)[:, p3_indices]
            wh = (boxes[..., 2:4] - boxes[..., 0:2]).clamp_min(0.0)
            areas = (wh[..., 0] * wh[..., 1]).clamp_min(0.0)
            sizes = areas.sqrt()
            matched_boxes = boxes[p3_fg]
            matched_areas = areas[p3_fg]
            if matched_boxes.numel():
                debug.update(
                    {
                        "matched_target_bboxes_min": float(matched_boxes.detach().min().item()),
                        "matched_target_bboxes_max": float(matched_boxes.detach().max().item()),
                        "matched_area_mean": float(matched_areas.detach().mean().item()),
                        "matched_target_bboxes_count": float(matched_boxes.shape[0]),
                    }
                )
            pos_mask = pos_mask & sizes.lt(float(self.freqrrr_cfg["tiny_thr"]))
        elif target_bboxes is not None and target_bboxes.ndim == 3 and target_bboxes.shape[:2] == fg.shape:
            boxes = target_bboxes.detach().to(device=logits.device, dtype=torch.float32)[:, p3_indices]
            matched_boxes = boxes[p3_fg]
            if matched_boxes.numel():
                wh = (matched_boxes[..., 2:4] - matched_boxes[..., 0:2]).clamp_min(0.0)
                matched_areas = (wh[..., 0] * wh[..., 1]).clamp_min(0.0)
                debug.update(
                    {
                        "matched_target_bboxes_min": float(matched_boxes.detach().min().item()),
                        "matched_target_bboxes_max": float(matched_boxes.detach().max().item()),
                        "matched_area_mean": float(matched_areas.detach().mean().item()),
                        "matched_target_bboxes_count": float(matched_boxes.shape[0]),
                    }
                )

        values = torch.where(pos_mask, scores, torch.zeros_like(scores))
        target = values.reshape(batch_size, 1, h, w).clamp(0.0, 1.0).detach()
        pos_num = int(pos_mask.sum().detach().item())
        target_sum = float(target.sum().detach().item())
        return target, pos_num, target_sum, debug

    def _compute_spatial_aux_gate_loss(
        self,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
        imgsz: torch.Tensor,
        preds: dict[str, torch.Tensor],
        dtype: torch.dtype,
        cfg: dict[str, Any],
        prefix: str,
        logits_attr: str,
        gate_attr: str,
        gain_attr: str | None,
        loss_key: str,
        pos_key: str,
        neg_key: str,
        gap_key: str | None,
        gain_key: str | None,
        nt_key: str | None,
    ) -> tuple[torch.Tensor, dict[str, float], list[nn.Module]]:
        """Compute a tiny-mask focal BCE loss for an internal learned spatial gate."""
        zero = preds["boxes"].sum() * 0.0
        stats = {
            loss_key: 0.0,
            pos_key: 0.0,
            neg_key: 0.0,
        }
        if gap_key is not None:
            stats[gap_key] = 0.0
        if gain_key is not None:
            stats[gain_key] = 0.0
        if nt_key is not None:
            stats[nt_key] = 0.0
        modules = list(self._iter_aux_modules(logits_attr))
        if modules and gain_key is not None and gain_attr is not None:
            stats[gain_key] = self._mean_module_float(modules, gain_attr)
        if not (self.include_gate_loss and cfg.get("enabled", False)):
            return zero, stats, modules

        batch_size = preds["boxes"].shape[0]
        stride = int(self.stride[0].item() if isinstance(self.stride, torch.Tensor) else self.stride[0])
        loss_terms = []
        zero_terms = []
        pos_sum = 0.0
        pos_count = 0
        neg_sum = 0.0
        neg_count = 0
        num_tiny_gt = 0
        for module in modules:
            logits = getattr(module, logits_attr, None)
            gate = getattr(module, gate_attr, None)
            if logits is None:
                continue
            target, tiny_count = build_tiny_mask_and_count(
                (gt_bboxes, mask_gt),
                stride=stride,
                feature_hw=logits.shape[-2:],
                batch_size=batch_size,
                image_size=imgsz,
                tiny_thr=self.freqrrr_cfg["tiny_thr"],
                device=logits.device,
                dtype=logits.dtype,
            )
            num_tiny_gt = max(num_tiny_gt, tiny_count)
            if not target.gt(0.0).any().item():
                zero_terms.append(logits.sum() * 0.0)
                continue
            loss_terms.append(focal_bce_with_logits(logits, target.detach(), alpha=0.75, gamma=2.0))
            gate_detached = (gate if gate is not None else logits.sigmoid()).detach()
            target_stats = target.to(device=gate_detached.device, dtype=gate_detached.dtype)
            pos_mask = target_stats > 0.5
            neg_mask = ~pos_mask
            if pos_mask.any():
                pos_sum += float(gate_detached[pos_mask].mean().item())
                pos_count += 1
            if neg_mask.any():
                neg_sum += float(gate_detached[neg_mask].mean().item())
                neg_count += 1

        if not loss_terms:
            return (torch.stack(zero_terms).sum() if zero_terms else zero), stats, modules

        aux_loss = torch.stack(loss_terms).mean() * float(cfg.get("loss_weight", 0.0))
        aux_loss = self._sanitize_aux_loss(aux_loss, zero, loss_key)
        pos_mean = pos_sum / max(pos_count, 1)
        neg_mean = neg_sum / max(neg_count, 1)
        stats[loss_key] = float(aux_loss.detach().item())
        stats[pos_key] = pos_mean
        stats[neg_key] = neg_mean
        if gap_key is not None:
            stats[gap_key] = pos_mean - neg_mean
        if gain_key is not None and gain_attr is not None:
            stats[gain_key] = self._mean_module_float(modules, gain_attr)
        if nt_key is not None:
            stats[nt_key] = float(num_tiny_gt)
        return aux_loss, stats, modules

    def _compute_task_aligned_aux_gate_loss(
        self,
        preds: dict[str, torch.Tensor],
        dtype: torch.dtype,
        cfg: dict[str, Any],
        logits_attr: str,
        gate_attr: str,
        loss_key: str,
        pos_key: str,
        neg_key: str,
        gap_key: str | None,
        num_key: str | None,
        sum_key: str | None,
        fg_mask: torch.Tensor | None,
        target_scores: torch.Tensor | None,
        target_bboxes: torch.Tensor | None,
        stride_tensor: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, float], list[nn.Module]]:
        """Compute focal BCE using assigner-aligned P3 positive points as the gate target."""
        zero = preds["boxes"].sum() * 0.0
        stats = {loss_key: 0.0, pos_key: 0.0, neg_key: 0.0}
        if gap_key is not None:
            stats[gap_key] = 0.0
        if num_key is not None:
            stats[num_key] = 0.0
        if sum_key is not None:
            stats[sum_key] = 0.0
        if num_key == "ta_num":
            stats.update(
                {
                    "ta_mean": 0.0,
                    "matched_target_bboxes_min": 0.0,
                    "matched_target_bboxes_max": 0.0,
                    "matched_area_mean": 0.0,
                }
            )

        modules = list(self._iter_aux_modules(logits_attr))
        if not (self.include_gate_loss and cfg.get("enabled", False)):
            return zero, stats, modules

        loss_terms = []
        zero_terms = []
        pos_sum = 0.0
        pos_count = 0
        neg_sum = 0.0
        neg_count = 0
        total_pos = 0
        total_target_sum = 0.0
        matched_min = None
        matched_max = None
        matched_area_sum = 0.0
        matched_count = 0.0
        for module in modules:
            logits = getattr(module, logits_attr, None)
            gate = getattr(module, gate_attr, None)
            if logits is None:
                continue

            target, pos_num, target_sum, target_debug = self._build_task_gate_target(
                logits=logits,
                fg_mask=fg_mask,
                target_scores=target_scores,
                target_bboxes=target_bboxes,
                stride_tensor=stride_tensor,
                tiny_only=bool(cfg.get("tiny_only", True)),
                target_value_mode=str(cfg.get("target_value_mode", "score")),
            )
            total_pos += pos_num
            total_target_sum += target_sum
            debug_count = float(target_debug.get("matched_target_bboxes_count", 0.0) or 0.0)
            if debug_count > 0.0:
                debug_min = float(target_debug["matched_target_bboxes_min"])
                debug_max = float(target_debug["matched_target_bboxes_max"])
                matched_min = debug_min if matched_min is None else min(matched_min, debug_min)
                matched_max = debug_max if matched_max is None else max(matched_max, debug_max)
                matched_area_sum += float(target_debug["matched_area_mean"]) * debug_count
                matched_count += debug_count
            if pos_num <= 0:
                zero_terms.append(logits.sum() * 0.0)
                continue

            loss_terms.append(focal_bce_with_logits(logits, target.detach(), alpha=0.75, gamma=2.0))
            gate_detached = (gate if gate is not None else logits.sigmoid()).detach()
            target_stats = target.to(device=gate_detached.device, dtype=gate_detached.dtype)
            pos_mask = target_stats > 0.0
            neg_mask = ~pos_mask
            if pos_mask.any():
                pos_sum += float(gate_detached[pos_mask].mean().item())
                pos_count += 1
            if neg_mask.any():
                neg_sum += float(gate_detached[neg_mask].mean().item())
                neg_count += 1

        if not loss_terms:
            aux_loss = torch.stack(zero_terms).sum() if zero_terms else zero
        else:
            aux_loss = torch.stack(loss_terms).mean() * float(cfg.get("loss_weight", 0.0))
            aux_loss = self._sanitize_aux_loss(aux_loss, zero, loss_key)

        pos_mean = pos_sum / max(pos_count, 1)
        neg_mean = neg_sum / max(neg_count, 1)
        stats[loss_key] = float(aux_loss.detach().item())
        stats[pos_key] = pos_mean
        stats[neg_key] = neg_mean
        if gap_key is not None:
            stats[gap_key] = pos_mean - neg_mean
        if num_key is not None:
            stats[num_key] = float(total_pos)
        if sum_key is not None:
            stats[sum_key] = float(total_target_sum)
        if num_key == "ta_num":
            stats["ta_mean"] = float(total_target_sum) / float(max(total_pos, 1))
            if matched_count > 0.0:
                stats["matched_target_bboxes_min"] = float(matched_min)
                stats["matched_target_bboxes_max"] = float(matched_max)
                stats["matched_area_mean"] = float(matched_area_sum / matched_count)
                if (
                    bool(self.freqrrr_cfg.get("log", {}).get("debug_metrics", False))
                    and float(matched_max) <= 2.0
                    and not getattr(self, "_tagate_norm_warned", False)
                ):
                    LOGGER.warning("TAGate target_bboxes may be normalized; tiny_thr pixel filtering may be wrong.")
                    self._tagate_norm_warned = True
        return aux_loss, stats, modules

    def _compute_rr_gate_loss(
        self,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
        imgsz: torch.Tensor,
        preds: dict[str, torch.Tensor],
        dtype: torch.dtype,
        fg_mask: torch.Tensor | None = None,
        target_scores: torch.Tensor | None = None,
        target_bboxes: torch.Tensor | None = None,
        stride_tensor: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute auxiliary loss for RRFusionTinyGate."""
        if self.rr_gate_cfg.get("target_mode") == "task_aligned":
            loss, stats, _ = self._compute_task_aligned_aux_gate_loss(
                preds=preds,
                dtype=dtype,
                cfg=self.rr_gate_cfg,
                logits_attr="last_rr_spatial_gate_logits",
                gate_attr="last_rr_spatial_gate",
                loss_key="ta_l",
                pos_key="ta_p",
                neg_key="ta_n",
                gap_key=None,
                num_key="ta_num",
                sum_key="ta_sum",
                fg_mask=fg_mask,
                target_scores=target_scores,
                target_bboxes=target_bboxes,
                stride_tensor=stride_tensor,
            )
            return loss, stats
        loss, stats, _ = self._compute_spatial_aux_gate_loss(
            gt_bboxes,
            mask_gt,
            imgsz,
            preds,
            dtype,
            self.rr_gate_cfg,
            "rg",
            "last_rr_spatial_gate_logits",
            "last_rr_spatial_gate",
            "last_detail_gain",
            "rg_l",
            "rg_p",
            "rg_n",
            "rg_gap",
            "rg_gain",
            "rg_nt",
        )
        if self._is_dual_target_freq_aux_cfg(self.rr_gate_cfg):
            stats["dt_l"] = float(stats.get("rg_l", 0.0))
        return loss, stats

    def _compute_tgf_detail_loss(
        self,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
        imgsz: torch.Tensor,
        preds: dict[str, torch.Tensor],
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute auxiliary loss for RRFusionTGFDetail."""
        loss, stats, _ = self._compute_spatial_aux_gate_loss(
            gt_bboxes,
            mask_gt,
            imgsz,
            preds,
            dtype,
            self.tgf_detail_cfg,
            "td",
            "last_tgf_detail_logits",
            "last_tgf_detail_gate",
            "last_td_gain",
            "td_l",
            "td_p",
            "td_n",
            "td_gap",
            "td_g",
            "td_nt",
        )
        return loss, stats

    def _compute_freq_gate_loss(
        self,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
        imgsz: torch.Tensor,
        preds: dict[str, torch.Tensor],
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute auxiliary loss for RRFusionFreqGate."""
        loss, stats, modules = self._compute_spatial_aux_gate_loss(
            gt_bboxes,
            mask_gt,
            imgsz,
            preds,
            dtype,
            self.freq_gate_cfg,
            "fg",
            "last_freq_spatial_gate_logits",
            "last_freq_spatial_gate",
            "last_detail_gain",
            "fg_l",
            "fg_p",
            "fg_n",
            "fg_gap",
            "fg_gain",
            "fg_nt",
        )
        stats["fg_e"] = self._mean_module_float(modules, "last_freq_energy_mean")
        stats["fg_c"] = self._mean_module_float(modules, "last_freq_contrast_mean")
        return loss, stats

    def _compute_freq_aux_gate_loss(
        self,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
        imgsz: torch.Tensor,
        preds: dict[str, torch.Tensor],
        dtype: torch.dtype,
        fg_mask: torch.Tensor | None = None,
        target_scores: torch.Tensor | None = None,
        target_bboxes: torch.Tensor | None = None,
        stride_tensor: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the train-only frequency-aware auxiliary gate loss on RRFusionTinyGate."""
        cfg = dict(self.rr_gate_cfg)
        cfg["enabled"] = bool(cfg.get("enabled", False) and cfg.get("use_freq_aux", False))
        freq_aux_target_mode = self._freq_aux_target_mode(cfg)
        current_weight = self._scheduled_freq_aux_weight(cfg)
        cfg["loss_weight"] = current_weight
        if cfg.get("target_mode") == "tiny_mask" and freq_aux_target_mode == "task_aligned":
            loss, stats, modules = self._compute_task_aligned_aux_gate_loss(
                preds=preds,
                dtype=dtype,
                cfg=cfg,
                logits_attr="last_freq_aux_spatial_gate_logits",
                gate_attr="last_freq_aux_spatial_gate",
                loss_key="df_l",
                pos_key="df_p",
                neg_key="df_n",
                gap_key="df_gap",
                num_key="df_num",
                sum_key="df_sum",
                fg_mask=fg_mask,
                target_scores=target_scores,
                target_bboxes=target_bboxes,
                stride_tensor=stride_tensor,
            )
            stats["df_w"] = float(current_weight)
            return loss, stats

        if freq_aux_target_mode == "task_aligned":
            loss, stats, modules = self._compute_task_aligned_aux_gate_loss(
                preds=preds,
                dtype=dtype,
                cfg=cfg,
                logits_attr="last_freq_aux_spatial_gate_logits",
                gate_attr="last_freq_aux_spatial_gate",
                loss_key="taf_l",
                pos_key="taf_p",
                neg_key="taf_n",
                gap_key="taf_gap",
                num_key=None,
                sum_key=None,
                fg_mask=fg_mask,
                target_scores=target_scores,
                target_bboxes=target_bboxes,
                stride_tensor=stride_tensor,
            )
            return loss, stats

        loss, stats, modules = self._compute_spatial_aux_gate_loss(
            gt_bboxes,
            mask_gt,
            imgsz,
            preds,
            dtype,
            cfg,
            "fa",
            "last_freq_aux_spatial_gate_logits",
            "last_freq_aux_spatial_gate",
            None,
            "fa_l",
            "fa_p",
            "fa_n",
            "fa_gap",
            None,
            None,
        )
        stats["fa_e"] = self._mean_module_float(modules, "last_freq_aux_energy_mean")
        stats["fa_c"] = self._mean_module_float(modules, "last_freq_aux_contrast_mean")
        return loss, stats

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple:
        """Run the standard detection loss path, then add optional TGF gate supervision."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl, gate
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        gate_response, gate_valid = self._build_assignment_gate(preds, dtype)
        current_epoch = self._get_current_epoch()

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
            gate_response=gate_response,
            stride_tensor=stride_tensor,
            gate_valid=gate_valid,
            current_epoch=current_epoch,
        )

        target_scores_sum = target_scores.sum().clamp_min(1.0)
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if fg_mask.any():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

        gate_loss, gate_stats = self._compute_gate_loss(gt_bboxes, mask_gt, preds, dtype)
        rr_gate_loss, rr_gate_stats = self._compute_rr_gate_loss(
            gt_bboxes,
            mask_gt,
            imgsz,
            preds,
            dtype,
            fg_mask=fg_mask,
            target_scores=target_scores,
            target_bboxes=target_bboxes,
            stride_tensor=stride_tensor,
        )
        freq_aux_loss, freq_aux_stats = self._compute_freq_aux_gate_loss(
            gt_bboxes,
            mask_gt,
            imgsz,
            preds,
            dtype,
            fg_mask=fg_mask,
            target_scores=target_scores,
            target_bboxes=target_bboxes,
            stride_tensor=stride_tensor,
        )
        tgf_detail_loss, tgf_detail_stats = self._compute_tgf_detail_loss(gt_bboxes, mask_gt, imgsz, preds, dtype)
        freq_gate_loss, freq_gate_stats = self._compute_freq_gate_loss(gt_bboxes, mask_gt, imgsz, preds, dtype)
        loss[3] = gate_loss + rr_gate_loss + freq_aux_loss + tgf_detail_loss + freq_gate_loss
        gate_stats.update(rr_gate_stats)
        gate_stats.update(freq_aux_stats)
        gate_stats.update(tgf_detail_stats)
        gate_stats.update(freq_gate_stats)
        self._collect_assigner_stats(gate_stats)

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            gate_stats,
        )

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the FreqRRR detection loss and the extended logging vector."""
        batch_size = preds["boxes"].shape[0]
        (_, _, _, _, _), loss, gate_stats = self.get_assigned_targets_and_loss(preds, batch)
        stats = self._init_log_stats()
        stats["box_loss"] = float(loss[0].detach().item())
        stats["cls_loss"] = float(loss[1].detach().item())
        stats["dfl_loss"] = float(loss[2].detach().item())
        stats["loss_gate"] = float(loss[3].detach().item())
        stats.update(gate_stats)
        self._collect_fprb_stats(stats)
        return loss * batch_size, self._make_stats_tensor(stats, preds["boxes"].dtype).detach()


class FreqRRRE2EDetectLoss:
    """End-to-end wrapper that applies gate supervision only once while keeping dual assignment."""

    def __init__(self, model):
        """Initialize one-to-many and one-to-one FreqRRR losses."""
        self.one2many = FreqRRRDetectionLoss(model, tal_topk=10, include_gate_loss=True)
        self.one2one = FreqRRRDetectionLoss(model, tal_topk=7, tal_topk2=1, include_gate_loss=False)
        self.loss_names = self.one2many.loss_names
        self.updates = 0
        self.total = 1.0
        self.o2m = 0.8
        self.o2o = self.total - self.o2m
        self.o2m_copy = self.o2m
        self.final_o2m = 0.1

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate weighted one-to-many/one-to-one losses with a single gate loss term."""
        preds = self.one2many.parse_output(preds)
        one2many, one2one = preds["one2many"], preds["one2one"]
        try:
            loss_one2many = self.one2many.loss(one2many, batch)
            loss_one2one = self.one2one.loss(one2one, batch)
            merged = loss_one2many[0].clone()
            merged[:3] = loss_one2many[0][:3] * self.o2m + loss_one2one[0][:3] * self.o2o
            merged[3] = loss_one2many[0][3]
            return merged, loss_one2many[1]
        finally:
            clear_freqrrr_aux(self.one2many.model_ref)

    def update(self) -> None:
        """Update the one-to-many / one-to-one branch weighting schedule."""
        self.updates += 1
        self.o2m = self.decay(self.updates)
        self.o2o = max(self.total - self.o2m, 0)

    def decay(self, x) -> float:
        """Calculate the decayed one-to-many branch weight."""
        return max(1 - x / max(self.one2one.hyp.epochs - 1, 1), 0) * (self.o2m_copy - self.final_o2m) + self.final_o2m

class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 segmentation."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):  # model must be de-paralleled
        """Initialize the v8SegmentationLoss class with model parameters and mask overlap setting."""
        super().__init__(model, tal_topk, tal_topk2)
        self.overlap = model.args.overlap_mask
        self.bcedice_loss = BCEDiceLoss(weight_bce=0.5, weight_dice=0.5)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate and return the combined loss for detection and segmentation."""
        pred_masks, proto = preds["mask_coefficient"].permute(0, 2, 1).contiguous(), preds["proto"]
        loss = torch.zeros(5, device=self.device)  # box, seg, cls, dfl, semseg
        if isinstance(proto, tuple) and len(proto) == 2:
            proto, pred_semseg = proto
        else:
            pred_semseg = None
        (fg_mask, target_gt_idx, target_bboxes, _, _), det_loss, _ = self.get_assigned_targets_and_loss(preds, batch)
        # NOTE: re-assign index for consistency for now. Need to be removed in the future.
        loss[0], loss[2], loss[3] = det_loss[0], det_loss[1], det_loss[2]

        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        if fg_mask.sum():
            # Masks loss
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                # masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]
                proto = F.interpolate(proto, masks.shape[-2:], mode="bilinear", align_corners=False)

            imgsz = (
                torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_masks.dtype) * self.stride[0]
            )
            loss[1] = self.calculate_segmentation_loss(
                fg_mask,
                masks,
                target_gt_idx,
                target_bboxes,
                batch["batch_idx"].view(-1, 1),
                proto,
                pred_masks,
                imgsz,
            )
            if pred_semseg is not None:
                sem_masks = batch["sem_masks"].to(self.device)  # NxHxW
                sem_masks = F.one_hot(sem_masks.long(), num_classes=self.nc).permute(0, 3, 1, 2).float()  # NxCxHxW

                if self.overlap:
                    mask_zero = masks == 0  # NxHxW
                    sem_masks[mask_zero.unsqueeze(1).expand_as(sem_masks)] = 0
                else:
                    batch_idx = batch["batch_idx"].view(-1)  # [total_instances]
                    for i in range(batch_size):
                        instance_mask_i = masks[batch_idx == i]  # [num_instances_i, H, W]
                        if len(instance_mask_i) == 0:
                            continue
                        sem_masks[i, :, instance_mask_i.sum(dim=0) == 0] = 0

                loss[4] = self.bcedice_loss(pred_semseg, sem_masks)
                loss[4] *= self.hyp.box  # seg gain

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss
            if pred_semseg is not None:
                loss[4] += (pred_semseg * 0).sum()

        loss[1] *= self.hyp.box  # seg gain
        return loss * batch_size, loss.detach()  # loss(box, seg, cls, dfl, semseg)

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (N, H, W), where N is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (N, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (N, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (N,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if self.overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 pose estimation."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int = 10):  # model must be de-paralleled
        """Initialize v8PoseLoss with model parameters and keypoint-specific loss functions."""
        super().__init__(model, tal_topk, tal_topk2)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the total loss and detach it for pose estimation."""
        pred_kpts = preds["kpts"].permute(0, 2, 1).contiguous()
        loss = torch.zeros(5, device=self.device)  # box, kpt_location, kpt_visibility, cls, dfl
        (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), det_loss, _ = (
            self.get_assigned_targets_and_loss(preds, batch)
        )
        # NOTE: re-assign index for consistency for now. Need to be removed in the future.
        loss[0], loss[3], loss[4] = det_loss[0], det_loss[1], det_loss[2]

        batch_size = pred_kpts.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_kpts.dtype) * self.stride[0]

        # Pboxes
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        # Keypoint loss
        if fg_mask.sum():
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask,
                target_gt_idx,
                keypoints,
                batch["batch_idx"].view(-1, 1),
                stride_tensor,
                target_bboxes,
                pred_kpts,
            )

        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain

        return loss * batch_size, loss.detach()  # loss(box, pose, kobj, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points: torch.Tensor, pred_kpts: torch.Tensor) -> torch.Tensor:
        """Decode predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def _select_target_keypoints(
        self,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        target_gt_idx: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Select target keypoints for each anchor based on batch index and target ground truth index.

        Args:
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).

        Returns:
            (torch.Tensor): Selected keypoints tensor, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # Vectorized fill: compute within-batch position for each keypoint using cumulative offsets
        batch_idx_long = batch_idx.long()
        offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=keypoints.device)
        offsets.scatter_add_(0, batch_idx_long + 1, torch.ones_like(batch_idx_long))
        offsets = offsets.cumsum(0)
        within_idx = torch.arange(len(batch_idx), device=keypoints.device) - offsets[batch_idx_long]
        batched_keypoints[batch_idx_long, within_idx] = keypoints

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        return selected_keypoints

    def calculate_keypoints_loss(
        self,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        stride_tensor: torch.Tensor,
        target_bboxes: torch.Tensor,
        pred_kpts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            kpts_loss (torch.Tensor): The keypoints loss.
            kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        # Select target keypoints using helper method
        selected_keypoints = self._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks)

        # Divide coordinates by stride
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            target_bboxes /= stride_tensor
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class PoseLoss26(v8PoseLoss):
    """Criterion class for computing training losses for YOLOv8 pose estimation with RLE loss support."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):  # model must be de-paralleled
        """Initialize PoseLoss26 with model parameters and keypoint-specific loss functions including RLE loss."""
        super().__init__(model, tal_topk, tal_topk2)
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        self.rle_loss = None
        self.flow_model = model.model[-1].flow_model if hasattr(model.model[-1], "flow_model") else None
        if self.flow_model is not None:
            self.rle_loss = RLELoss(use_target_weight=True).to(self.device)
            self.target_weights = (
                torch.from_numpy(RLE_WEIGHT).to(self.device) if is_pose else torch.ones(nkpt, device=self.device)
            )

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the total loss and detach it for pose estimation."""
        pred_kpts = preds["kpts"].permute(0, 2, 1).contiguous()
        loss = torch.zeros(
            6 if self.rle_loss else 5, device=self.device
        )  # box, kpt_location, kpt_visibility, cls, dfl[, rle]
        (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), det_loss, _ = (
            self.get_assigned_targets_and_loss(preds, batch)
        )
        # NOTE: re-assign index for consistency for now. Need to be removed in the future.
        loss[0], loss[3], loss[4] = det_loss[0], det_loss[1], det_loss[2]

        batch_size = pred_kpts.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_kpts.dtype) * self.stride[0]

        pred_kpts = pred_kpts.view(batch_size, -1, *self.kpt_shape)  # (b, h*w, 17, 3)

        if self.rle_loss and preds.get("kpts_sigma", None) is not None:
            pred_sigma = preds["kpts_sigma"].permute(0, 2, 1).contiguous()
            pred_sigma = pred_sigma.view(batch_size, -1, self.kpt_shape[0], 2)  # (b, h*w, 17, 2)
            pred_kpts = torch.cat([pred_kpts, pred_sigma], dim=-1)  # (b, h*w, 17, 5)

        pred_kpts = self.kpts_decode(anchor_points, pred_kpts)

        # Keypoint loss
        if fg_mask.sum():
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            keypoints_loss = self.calculate_keypoints_loss(
                fg_mask,
                target_gt_idx,
                keypoints,
                batch["batch_idx"].view(-1, 1),
                stride_tensor,
                target_bboxes,
                pred_kpts,
            )
            loss[1] = keypoints_loss[0]
            loss[2] = keypoints_loss[1]
            if self.rle_loss is not None:
                loss[5] = keypoints_loss[2]

        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        if self.rle_loss is not None:
            loss[5] *= self.hyp.rle  # rle gain

        return loss * batch_size, loss.detach()  # loss(box, kpt_location, kpt_visibility, cls, dfl[, rle])

    @staticmethod
    def kpts_decode(anchor_points: torch.Tensor, pred_kpts: torch.Tensor) -> torch.Tensor:
        """Decode predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., 0] += anchor_points[:, [0]]
        y[..., 1] += anchor_points[:, [1]]
        return y

    def calculate_rle_loss(self, pred_kpt: torch.Tensor, gt_kpt: torch.Tensor, kpt_mask: torch.Tensor) -> torch.Tensor:
        """Calculate the RLE (Residual Log-likelihood Estimation) loss for keypoints.

        Args:
            pred_kpt (torch.Tensor): Predicted kpts with sigma, shape (N, num_keypoints, kpts_dim) where kpts_dim >= 4.
            gt_kpt (torch.Tensor): Ground truth keypoints, shape (N, num_keypoints, kpts_dim).
            kpt_mask (torch.Tensor): Mask for valid keypoints, shape (N, num_keypoints).

        Returns:
            (torch.Tensor): The RLE loss.
        """
        pred_kpt_visible = pred_kpt[kpt_mask]
        gt_kpt_visible = gt_kpt[kpt_mask]
        pred_coords = pred_kpt_visible[:, 0:2]
        pred_sigma = pred_kpt_visible[:, -2:]
        gt_coords = gt_kpt_visible[:, 0:2]

        target_weights = self.target_weights.unsqueeze(0).repeat(kpt_mask.shape[0], 1)
        target_weights = target_weights[kpt_mask]

        pred_sigma = pred_sigma.sigmoid()
        error = (pred_coords - gt_coords) / (pred_sigma + 1e-9)

        # Filter out NaN and Inf values to prevent MultivariateNormal validation errors
        valid_mask = ~(torch.isnan(error) | torch.isinf(error)).any(dim=-1)
        if not valid_mask.any():
            return torch.tensor(0.0, device=pred_kpt.device)

        error = error[valid_mask]
        error = error.clamp(-100, 100)  # Prevent numerical instability
        pred_sigma = pred_sigma[valid_mask]
        target_weights = target_weights[valid_mask]

        log_phi = self.flow_model.log_prob(error)

        return self.rle_loss(pred_sigma, log_phi, error, target_weights)

    def calculate_keypoints_loss(
        self,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        stride_tensor: torch.Tensor,
        target_bboxes: torch.Tensor,
        pred_kpts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            kpts_loss (torch.Tensor): The keypoints loss.
            kpts_obj_loss (torch.Tensor): The keypoints object loss.
            rle_loss (torch.Tensor): The RLE loss.
        """
        # Select target keypoints using inherited helper method
        selected_keypoints = self._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks)

        # Divide coordinates by stride
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0
        rle_loss = 0

        if masks.any():
            target_bboxes /= stride_tensor
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if self.rle_loss is not None and (pred_kpt.shape[-1] == 4 or pred_kpt.shape[-1] == 5):
                rle_loss = self.calculate_rle_loss(pred_kpt, gt_kpt, kpt_mask)
                rle_loss = rle_loss.clamp(min=0)
            if pred_kpt.shape[-1] == 3 or pred_kpt.shape[-1] == 5:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss, rle_loss


class v8ClassificationLoss:
    """Criterion class for computing training losses for classification."""

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the classification loss between predictions and true labels."""
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        return loss, loss.detach()


class v8OBBLoss(v8DetectionLoss):
    """Calculates losses for object detection, classification, and box distribution in rotated YOLO models."""

    def __init__(self, model, tal_topk=10, tal_topk2: int | None = None):
        """Initialize v8OBBLoss with model, assigner, and rotated bbox loss; model must be de-paralleled."""
        super().__init__(model, tal_topk=tal_topk)
        self.assigner = RotatedTaskAlignedAssigner(
            topk=tal_topk,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,
            stride=self.stride.tolist(),
            topk2=tal_topk2,
        )
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets for oriented bounding box detection."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            batch_idx = targets[:, 0].long()  # image index
            _, counts = batch_idx.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            packed_targets = targets[:, 1:].clone()
            packed_targets[:, 1:5].mul_(scale_tensor)
            offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=self.device)
            offsets.scatter_add_(0, batch_idx + 1, torch.ones_like(batch_idx))
            offsets = offsets.cumsum(0)
            within_idx = torch.arange(len(targets), device=self.device) - offsets[batch_idx]
            out[batch_idx, within_idx] = packed_targets
        return out

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate and return the loss for oriented bounding box detection."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl, angle
        pred_distri, pred_scores, pred_angle = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
            preds["angle"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        batch_size = pred_angle.shape[0]  # batch size

        dtype = pred_scores.dtype
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        # targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * float(imgsz[1]), targets[:, 5] * float(imgsz[0])
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolo26n-obb.pt data=dota8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # xyxy, (b, h*w, 4)

        bboxes_for_assigner = pred_bboxes.clone().detach()
        # Only the first four elements need to be scaled
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )
            weight = target_scores.sum(-1)[fg_mask]
            loss[3] = self.calculate_angle_loss(
                pred_bboxes, target_bboxes, fg_mask, weight, target_scores_sum
            )  # angle loss
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        loss[3] *= self.hyp.angle  # angle gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl, angle)

    def bbox_decode(
        self, anchor_points: torch.Tensor, pred_dist: torch.Tensor, pred_angle: torch.Tensor
    ) -> torch.Tensor:
        """Decode predicted object bounding box coordinates from anchor points and distribution.

        Args:
            anchor_points (torch.Tensor): Anchor points, (h*w, 2).
            pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
            pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).

        Returns:
            (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)

    def calculate_angle_loss(self, pred_bboxes, target_bboxes, fg_mask, weight, target_scores_sum, lambda_val=3):
        """Calculate oriented angle loss.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes with shape [N, 5] (x, y, w, h, theta).
            target_bboxes (torch.Tensor): Target bounding boxes with shape [N, 5] (x, y, w, h, theta).
            fg_mask (torch.Tensor): Foreground mask indicating valid predictions.
            weight (torch.Tensor): Loss weights for each prediction.
            target_scores_sum (torch.Tensor): Sum of target scores for normalization.
            lambda_val (int): Controls the sensitivity to aspect ratio.

        Returns:
            (torch.Tensor): The calculated angle loss.
        """
        w_gt = target_bboxes[..., 2]
        h_gt = target_bboxes[..., 3]
        pred_theta = pred_bboxes[..., 4]
        target_theta = target_bboxes[..., 4]

        log_ar = torch.log((w_gt + 1e-9) / (h_gt + 1e-9))
        scale_weight = torch.exp(-(log_ar**2) / (lambda_val**2))

        delta_theta = pred_theta - target_theta
        delta_theta_wrapped = delta_theta - torch.round(delta_theta / math.pi) * math.pi
        ang_loss = torch.sin(2 * delta_theta_wrapped[fg_mask]) ** 2

        ang_loss = scale_weight[fg_mask] * ang_loss
        ang_loss = ang_loss * weight

        return ang_loss.sum() / target_scores_sum


class E2EDetectLoss:
    """Criterion class for computing training losses for end-to-end detection."""

    def __init__(self, model):
        """Initialize E2EDetectLoss with one-to-many and one-to-one detection losses using the provided model."""
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]


class E2ELoss:
    """Criterion class for computing training losses for end-to-end detection."""

    def __init__(self, model, loss_fn=v8DetectionLoss):
        """Initialize E2ELoss with one-to-many and one-to-one detection losses using the provided model."""
        self.one2many = loss_fn(model, tal_topk=10)
        self.one2one = loss_fn(model, tal_topk=7, tal_topk2=1)
        self.updates = 0
        self.total = 1.0
        # init gain
        self.o2m = 0.8
        self.o2o = self.total - self.o2m
        self.o2m_copy = self.o2m
        # final gain
        self.final_o2m = 0.1

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = self.one2many.parse_output(preds)
        one2many, one2one = preds["one2many"], preds["one2one"]
        loss_one2many = self.one2many.loss(one2many, batch)
        loss_one2one = self.one2one.loss(one2one, batch)
        return loss_one2many[0] * self.o2m + loss_one2one[0] * self.o2o, loss_one2one[1]

    def update(self) -> None:
        """Update the weights for one-to-many and one-to-one losses based on the decay schedule."""
        self.updates += 1
        self.o2m = self.decay(self.updates)
        self.o2o = max(self.total - self.o2m, 0)

    def decay(self, x) -> float:
        """Calculate the decayed weight for one-to-many loss based on the current update step."""
        return max(1 - x / max(self.one2one.hyp.epochs - 1, 1), 0) * (self.o2m_copy - self.final_o2m) + self.final_o2m


class TinyObjectE2EDetectLoss:
    """End-to-end detection wrapper with shared-step training-only tiny-object learning."""

    def __init__(self, model):
        """Initialize TinyObjectE2EDetectLoss with shared warmup state across both branches."""
        self.shared_state = {"step": 0}
        self.one2many = TinyObjectDetectionLoss(model, tal_topk=10, shared_state=self.shared_state)
        self.one2one = TinyObjectDetectionLoss(model, tal_topk=7, tal_topk2=1, shared_state=self.shared_state)
        self.updates = 0
        self.total = 1.0
        self.o2m = 0.8
        self.o2o = self.total - self.o2m
        self.o2m_copy = self.o2m
        self.final_o2m = 0.1

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the official end-to-end combined loss with a shared tiny-learning warmup step."""
        self.shared_state["step"] += 1
        preds = self.one2many.parse_output(preds)
        one2many, one2one = preds["one2many"], preds["one2one"]
        loss_one2many = self.one2many.loss(one2many, batch)
        loss_one2one = self.one2one.loss(one2one, batch)
        return loss_one2many[0] * self.o2m + loss_one2one[0] * self.o2o, loss_one2one[1]

    def update(self) -> None:
        """Update the official one-to-many and one-to-one branch weights."""
        self.updates += 1
        self.o2m = self.decay(self.updates)
        self.o2o = max(self.total - self.o2m, 0)

    def decay(self, x) -> float:
        """Calculate the decayed one-to-many branch weight."""
        return max(1 - x / max(self.one2one.hyp.epochs - 1, 1), 0) * (self.o2m_copy - self.final_o2m) + self.final_o2m


class TVPDetectLoss:
    """Criterion class for computing training losses for text-visual prompt detection."""

    def __init__(self, model, tal_topk=10, tal_topk2: int | None = None):
        """Initialize TVPDetectLoss with task-prompt and visual-prompt criteria using the provided model."""
        self.vp_criterion = v8DetectionLoss(model, tal_topk, tal_topk2)
        # NOTE: store following info as it's changeable in __call__
        self.hyp = self.vp_criterion.hyp
        self.ori_nc = self.vp_criterion.nc
        self.ori_no = self.vp_criterion.no
        self.ori_reg_max = self.vp_criterion.reg_max

    def parse_output(self, preds) -> dict[str, torch.Tensor]:
        """Parse model predictions to extract features."""
        return self.vp_criterion.parse_output(preds)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt detection."""
        return self.loss(self.parse_output(preds), batch)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt detection."""
        if self.ori_nc == preds["scores"].shape[1]:
            loss = torch.zeros(3, device=self.vp_criterion.device, requires_grad=True)
            return loss, loss.detach()

        preds["scores"] = self._get_vp_features(preds)
        vp_loss = self.vp_criterion(preds, batch)
        box_loss = vp_loss[0][1]
        return box_loss, vp_loss[1]

    def _get_vp_features(self, preds: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        """Extract visual-prompt features from the model output."""
        scores = preds["scores"]
        vnc = scores.shape[1]

        self.vp_criterion.nc = vnc
        self.vp_criterion.no = vnc + self.vp_criterion.reg_max * 4
        self.vp_criterion.assigner.num_classes = vnc
        return scores


class TVPSegmentLoss(TVPDetectLoss):
    """Criterion class for computing training losses for text-visual prompt segmentation."""

    def __init__(self, model, tal_topk=10):
        """Initialize TVPSegmentLoss with task-prompt and visual-prompt criteria using the provided model."""
        super().__init__(model)
        self.vp_criterion = v8SegmentationLoss(model, tal_topk)
        self.hyp = self.vp_criterion.hyp

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt segmentation."""
        return self.loss(self.parse_output(preds), batch)

    def loss(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt segmentation."""
        if self.ori_nc == preds["scores"].shape[1]:
            loss = torch.zeros(4, device=self.vp_criterion.device, requires_grad=True)
            return loss, loss.detach()

        preds["scores"] = self._get_vp_features(preds)
        vp_loss = self.vp_criterion(preds, batch)
        cls_loss = vp_loss[0][2]
        return cls_loss, vp_loss[1]
