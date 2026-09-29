# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch
import torch.nn as nn

from . import LOGGER
from .freqrrr import box_area_sqrt, compute_scale_alpha, normalized_wasserstein_similarity
from .metrics import bbox_iou, probiou
from .ops import xywh2xyxy, xywhr2xyxyxyxy, xyxy2xywh
from .torch_utils import TORCH_1_11


def compute_fqa_lambda(current_epoch: int | float | None, lambda_f: float, warmup_epochs: int) -> float:
    """Compute the current FQA lambda after warmup scaling."""
    epoch = float(current_epoch or 0.0)
    if warmup_epochs <= 0:
        warmup = 1.0
    else:
        warmup = min(max(epoch / float(warmup_epochs), 0.0), 1.0)
    return float(lambda_f) * warmup


def apply_fqa_ranking_boost(
    metric_base: torch.Tensor,
    gate_flat: torch.Tensor | None,
    tiny_gt_mask: torch.Tensor,
    freq_lambda: float,
    gate_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply a detached frequency boost for top-k ranking only."""
    if gate_flat is None or float(freq_lambda) <= 0.0:
        return metric_base

    gate = gate_flat.to(device=metric_base.device, dtype=metric_base.dtype)
    gate = torch.nan_to_num(gate, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    if gate_valid is not None:
        valid = gate_valid.to(device=metric_base.device, dtype=torch.bool)
        gate = torch.where(valid, gate, torch.zeros_like(gate))

    boost = 1.0 + float(freq_lambda) * gate
    boost_broadcast = boost.unsqueeze(1)
    metric_for_topk = torch.where(tiny_gt_mask.bool(), metric_base * boost_broadcast, metric_base)
    return torch.nan_to_num(metric_for_topk, nan=0.0, posinf=0.0, neginf=0.0)


class TaskAlignedAssigner(nn.Module):
    """A task-aligned assigner for object detection.

    This class assigns ground-truth (gt) objects to anchors based on the task-aligned metric, which combines both
    classification and localization information.

    Attributes:
        topk (int): The number of top candidates to consider.
        topk2 (int): Secondary topk value for additional filtering.
        num_classes (int): The number of object classes.
        alpha (float): The alpha parameter for the classification component of the task-aligned metric.
        beta (float): The beta parameter for the localization component of the task-aligned metric.
        stride (list): List of stride values for different feature levels.
        stride_val (int): The stride value used for select_candidates_in_gts.
        eps (float): A small value to prevent division by zero.
    """

    def __init__(
        self,
        topk: int = 13,
        num_classes: int = 80,
        alpha: float = 1.0,
        beta: float = 6.0,
        stride: list = [8, 16, 32],
        eps: float = 1e-9,
        topk2=None,
        sfqa_cfg: dict | None = None,
        fqa_cfg: dict | None = None,
    ):
        """Initialize a TaskAlignedAssigner object with customizable hyperparameters.

        Args:
            topk (int, optional): The number of top candidates to consider.
            num_classes (int, optional): The number of object classes.
            alpha (float, optional): The alpha parameter for the classification component of the task-aligned metric.
            beta (float, optional): The beta parameter for the localization component of the task-aligned metric.
            stride (list, optional): List of stride values for different feature levels.
            eps (float, optional): A small value to prevent division by zero.
            topk2 (int, optional): Secondary topk value for additional filtering.
        """
        super().__init__()
        self.topk = topk
        self.topk2 = topk2 or topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.stride = stride
        self.stride_val = self.stride[1] if len(self.stride) > 1 else self.stride[0]
        self.eps = eps
        self.sfqa_cfg = sfqa_cfg or {}
        self.sfqa_enabled = bool(self.sfqa_cfg.get("enabled", False))
        self.fqa_cfg = fqa_cfg or {}
        self.fqa_enabled = bool(self.fqa_cfg.get("enabled", False))
        self.last_stats = {
            "mean_iou": 0.0,
            "mean_nwd": 0.0,
            "mean_q_geo": 0.0,
            "mean_q_assign": 0.0,
            "mean_alpha_s": 0.0,
            "num_pos_tiny": 0.0,
            "num_pos_all": 0.0,
            "fq_lam": 0.0,
            "fq_g": 0.0,
            "fq_bst": 0.0,
            "fq_tiny": 0.0,
        }

    @torch.no_grad()
    def forward(
        self,
        pd_scores,
        pd_bboxes,
        anc_points,
        gt_labels,
        gt_bboxes,
        mask_gt,
        gate_response=None,
        stride_tensor=None,
        gate_valid=None,
        current_epoch: int | float | None = None,
    ):
        """Compute the task-aligned assignment.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

        Returns:
            target_labels (torch.Tensor): Target labels with shape (bs, num_total_anchors).
            target_bboxes (torch.Tensor): Target bounding boxes with shape (bs, num_total_anchors, 4).
            target_scores (torch.Tensor): Target scores with shape (bs, num_total_anchors, num_classes).
            fg_mask (torch.Tensor): Foreground mask with shape (bs, num_total_anchors).
            target_gt_idx (torch.Tensor): Target ground truth indices with shape (bs, num_total_anchors).

        References:
            https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py
        """
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]
        device = gt_bboxes.device
        self.last_stats = {k: 0.0 for k in self.last_stats}

        if self.n_max_boxes == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.num_classes),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0]),
                torch.zeros_like(pd_scores[..., 0]),
            )

        try:
            return self._forward(
                pd_scores,
                pd_bboxes,
                anc_points,
                gt_labels,
                gt_bboxes,
                mask_gt,
                gate_response,
                stride_tensor,
                gate_valid,
                current_epoch,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Move tensors to CPU, compute, then move back to original device
                LOGGER.warning("CUDA OutOfMemoryError in TaskAlignedAssigner, using CPU")
                cpu_tensors = [
                    t.cpu() if t is not None else None
                    for t in (pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, gate_response, stride_tensor, gate_valid)
                ]
                result = self._forward(*cpu_tensors, current_epoch=current_epoch)
                return tuple(t.to(device) for t in result)
            raise

    def _forward(
        self,
        pd_scores,
        pd_bboxes,
        anc_points,
        gt_labels,
        gt_bboxes,
        mask_gt,
        gate_response=None,
        stride_tensor=None,
        gate_valid=None,
        current_epoch: int | float | None = None,
    ):
        """Compute the task-aligned assignment.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

        Returns:
            target_labels (torch.Tensor): Target labels with shape (bs, num_total_anchors).
            target_bboxes (torch.Tensor): Target bounding boxes with shape (bs, num_total_anchors, 4).
            target_scores (torch.Tensor): Target scores with shape (bs, num_total_anchors, num_classes).
            fg_mask (torch.Tensor): Foreground mask with shape (bs, num_total_anchors).
            target_gt_idx (torch.Tensor): Target ground truth indices with shape (bs, num_total_anchors).
        """
        mask_pos, metric_for_topk, metric_base, overlaps, quality = self.get_pos_mask(
            pd_scores,
            pd_bboxes,
            gt_labels,
            gt_bboxes,
            anc_points,
            mask_gt,
            gate_response,
            stride_tensor,
            gate_valid,
            current_epoch,
        )

        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(
            mask_pos, quality["match_quality"], self.n_max_boxes, metric_for_topk
        )

        # Assigned target
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        if self.sfqa_enabled and self.sfqa_cfg.get("use_mixed_quality_target", True):
            matched_q_geo = (quality["q_geo"] * mask_pos).amax(-2).unsqueeze(-1)
            target_scores = target_scores * matched_q_geo
        else:
            metric_for_target = metric_base * mask_pos
            pos_align_metrics = metric_for_target.amax(dim=-1, keepdim=True)  # b, max_num_obj
            pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # b, max_num_obj
            norm_align_metric = (metric_for_target * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
            target_scores = target_scores * norm_align_metric

        fg_mask_bool = fg_mask.bool()
        if fg_mask_bool.any():
            matched_gt_idx = target_gt_idx.clamp(0, self.n_max_boxes - 1)
            matched_sizes = quality["gt_sizes"].squeeze(-1).gather(1, matched_gt_idx)
            matched_iou = (quality["iou"] * mask_pos).amax(-2)
            matched_nwd = (quality["nwd"] * mask_pos).amax(-2)
            matched_q_geo = (quality["q_geo"] * mask_pos).amax(-2)
            matched_q_assign = (quality["q_assign"] * mask_pos).amax(-2)
            matched_alpha = (quality["alpha_s"].expand_as(quality["q_geo"]) * mask_pos).amax(-2)
            tiny_thr = float(self.sfqa_cfg.get("tiny_thr", 32.0))
            self.last_stats.update(
                {
                "mean_iou": float(matched_iou[fg_mask_bool].mean().item()),
                "mean_nwd": float(matched_nwd[fg_mask_bool].mean().item()),
                "mean_q_geo": float(matched_q_geo[fg_mask_bool].mean().item()),
                "mean_q_assign": float(matched_q_assign[fg_mask_bool].mean().item()),
                "mean_alpha_s": float(matched_alpha[fg_mask_bool].mean().item()),
                "num_pos_tiny": float((fg_mask_bool & (matched_sizes < tiny_thr)).sum().item()),
                "num_pos_all": float(fg_mask_bool.sum().item()),
                }
            )

        return target_labels, target_bboxes, target_scores, fg_mask_bool, target_gt_idx

    def get_pos_mask(
        self,
        pd_scores,
        pd_bboxes,
        gt_labels,
        gt_bboxes,
        anc_points,
        mask_gt,
        gate_response=None,
        stride_tensor=None,
        gate_valid=None,
        current_epoch: int | float | None = None,
    ):
        """Get positive mask for each ground truth box.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

        Returns:
            mask_pos (torch.Tensor): Positive mask with shape (bs, max_num_obj, h*w).
            align_metric (torch.Tensor): Alignment metric with shape (bs, max_num_obj, h*w).
            overlaps (torch.Tensor): Overlaps between predicted vs ground truth boxes with shape (bs, max_num_obj, h*w).
        """
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes, mask_gt, stride_tensor=stride_tensor)
        # Get anchor_align metric, (b, max_num_obj, h*w)
        metric_base, overlaps, quality = self.get_box_metrics(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt, gate_response, gate_valid
        )
        metric_for_topk = self._get_metric_for_topk(
            metric_base,
            gt_bboxes=gt_bboxes,
            mask_gt=mask_gt,
            gate_response=gate_response,
            gate_valid=gate_valid,
            current_epoch=current_epoch,
        )
        # Get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(metric_for_topk, gt_bboxes=gt_bboxes, mask_gt=mask_gt)
        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, metric_for_topk, metric_base, overlaps, quality

    def _build_tiny_gt_mask(self, gt_bboxes: torch.Tensor, mask_gt: torch.Tensor) -> torch.Tensor:
        """Build the broadcastable tiny-GT mask in input-pixel coordinates."""
        gt_sizes = box_area_sqrt(gt_bboxes)
        valid_gt = mask_gt.squeeze(-1).bool()
        if self.fqa_cfg.get("apply_to_tiny_only", True):
            tiny_mask = (gt_sizes < float(self.fqa_cfg.get("tiny_thr", 32.0))) & valid_gt
        else:
            tiny_mask = valid_gt
        return tiny_mask.unsqueeze(-1)

    def _get_metric_for_topk(
        self,
        metric_base: torch.Tensor,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
        gate_response: torch.Tensor | None = None,
        gate_valid: torch.Tensor | None = None,
        current_epoch: int | float | None = None,
    ) -> torch.Tensor:
        """Return the ranking metric used for top-k candidate selection."""
        stats = {
            "fq_lam": 0.0,
            "fq_g": 0.0,
            "fq_bst": 0.0,
            "fq_tiny": 0.0,
        }
        if not self.fqa_enabled:
            self.last_stats.update(stats)
            return metric_base

        tiny_gt_mask = self._build_tiny_gt_mask(gt_bboxes, mask_gt)
        tiny_count = int(tiny_gt_mask.squeeze(-1).sum().item())
        freq_lambda = compute_fqa_lambda(
            current_epoch=current_epoch,
            lambda_f=float(self.fqa_cfg.get("lambda_f", 0.15)),
            warmup_epochs=int(self.fqa_cfg.get("warmup_epochs", 10)),
        )
        stats["fq_lam"] = float(freq_lambda)
        stats["fq_tiny"] = float(tiny_count)
        stats["fq_bst"] = 1.0

        if tiny_count == 0:
            self.last_stats.update(stats)
            return metric_base

        if gate_response is None or gate_response.ndim != 2 or gate_response.shape != metric_base.shape[::2]:
            self.last_stats.update(stats)
            return metric_base

        gate = gate_response
        if self.fqa_cfg.get("detach_gate", True):
            gate = gate.detach()
        gate = gate.to(device=metric_base.device, dtype=metric_base.dtype)
        valid = None
        if gate_valid is not None and gate_valid.ndim == 2 and gate_valid.shape == gate.shape:
            valid = gate_valid.detach() if isinstance(gate_valid, torch.Tensor) else gate_valid
            valid = valid.to(device=metric_base.device, dtype=torch.bool)
        gate_clean = torch.nan_to_num(gate, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        if valid is not None:
            gate_clean = torch.where(valid, gate_clean, torch.zeros_like(gate_clean))

        if float(freq_lambda) > 0.0:
            metric_for_topk = apply_fqa_ranking_boost(
                metric_base=metric_base,
                gate_flat=gate_clean,
                tiny_gt_mask=tiny_gt_mask,
                freq_lambda=freq_lambda,
                gate_valid=valid,
            )
        else:
            metric_for_topk = metric_base

        used_mask = valid if valid is not None else torch.ones_like(gate_clean, dtype=torch.bool)
        if used_mask.any() and float(freq_lambda) > 0.0:
            stats["fq_g"] = float(gate_clean[used_mask].mean().item())
            stats["fq_bst"] = float((1.0 + float(freq_lambda) * gate_clean[used_mask]).mean().item())

        self.last_stats.update(stats)
        return metric_for_topk

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt, gate_response=None, gate_valid=None):
        """Compute alignment metric given predicted and ground truth bounding boxes.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, h*w).

        Returns:
            align_metric (torch.Tensor): Alignment metric combining classification and localization.
            overlaps (torch.Tensor): IoU overlaps between predicted and ground truth boxes.
        """
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        nwd = torch.zeros_like(overlaps)
        q_geo = torch.zeros_like(overlaps)
        q_assign = torch.zeros_like(overlaps)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)
        gt_sizes = box_area_sqrt(gt_bboxes).unsqueeze(-1)
        alpha_s = (
            compute_scale_alpha(
                gt_sizes,
                self.sfqa_cfg.get("s0", 24.0),
                self.sfqa_cfg.get("gamma", 1.0),
                self.sfqa_cfg.get("alpha_max", 0.70),
            )
            if self.sfqa_enabled
            else torch.zeros_like(gt_sizes)
        )

        batch_ind = torch.arange(end=self.bs, device=gt_labels.device, dtype=torch.long).view(-1, 1).expand(-1, self.n_max_boxes)
        ind = torch.stack((batch_ind, gt_labels.squeeze(-1).long()))  # 2, b, max_num_obj
        # Get the scores of each grid for each gt cls
        bbox_scores_selected = pd_scores[ind[0], :, ind[1]][mask_gt]
        bbox_scores = bbox_scores.masked_scatter(mask_gt, bbox_scores_selected)  # b, max_num_obj, h*w

        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlap_vals = self.iou_calculation(gt_boxes, pd_boxes)
        overlaps = overlaps.masked_scatter(mask_gt, overlap_vals)
        q_geo = overlaps
        q_assign = q_geo

        if self.sfqa_enabled and mask_gt.any():
            alpha_full = alpha_s.expand(-1, -1, na)
            alpha_vals = alpha_full[mask_gt]
            nwd_vals = normalized_wasserstein_similarity(
                pd_boxes, gt_boxes, nwd_c=self.sfqa_cfg.get("nwd_c", 12.8), eps=self.eps
            ).to(overlaps.dtype)
            nwd = nwd.masked_scatter(mask_gt, nwd_vals)
            q_geo_vals = (1.0 - alpha_vals) * overlap_vals + alpha_vals * nwd_vals
            q_geo_vals = q_geo_vals.clamp(0.0, 1.0)
            q_geo = q_geo.masked_scatter(mask_gt, q_geo_vals)
            q_assign = q_geo
            if (
                self.sfqa_cfg.get("use_gate_in_assign", False)
                and gate_response is not None
                and gate_valid is not None
                and gate_valid.any()
            ):
                gate_values = gate_response.unsqueeze(1).expand(-1, self.n_max_boxes, -1)
                gate_valid_full = gate_valid.unsqueeze(1).expand(-1, self.n_max_boxes, -1)
                beta_s = (
                    (gt_sizes < float(self.sfqa_cfg.get("tiny_thr", 32.0))).to(q_geo.dtype)
                    * float(self.sfqa_cfg.get("beta0", 0.20))
                ).expand(-1, -1, na)
                beta_s = torch.where(gate_valid_full, beta_s, torch.zeros_like(beta_s))
                gated_q = (1.0 - beta_s) * q_geo + beta_s * gate_values
                q_assign = torch.where(gate_valid_full, gated_q, q_geo)
                q_assign = q_assign.clamp(0.0, 1.0)

        align_metric = bbox_scores.pow(self.alpha) * q_assign.pow(self.beta)
        return align_metric, overlaps, {
            "iou": overlaps,
            "nwd": nwd,
            "q_geo": q_geo,
            "q_assign": q_assign,
            "alpha_s": alpha_s,
            "gt_sizes": gt_sizes,
            "match_quality": q_assign if self.sfqa_enabled else overlaps,
        }

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """Calculate IoU for horizontal bounding boxes.

        Args:
            gt_bboxes (torch.Tensor): Ground truth boxes.
            pd_bboxes (torch.Tensor): Predicted boxes.

        Returns:
            (torch.Tensor): IoU values between each pair of boxes.
        """
        return bbox_iou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)

    def select_topk_candidates(self, metrics, topk_mask=None, gt_bboxes=None, mask_gt=None):
        """Select the top-k candidates based on the given metrics.

        Args:
            metrics (torch.Tensor): A tensor of shape (b, max_num_obj, h*w), where b is the batch size, max_num_obj is
                the maximum number of objects, and h*w represents the total number of anchor points.
            topk_mask (torch.Tensor, optional): An optional boolean tensor of shape (b, max_num_obj, topk), where topk
                is the number of top candidates to consider. If not provided, the top-k values are automatically
                computed based on the given metrics.

        Returns:
            (torch.Tensor): A tensor of shape (b, max_num_obj, h*w) containing the selected top-k candidates.
        """
        dynamic_topk = None
        if (
            self.sfqa_enabled
            and self.sfqa_cfg.get("enable_tiny_candidate_relax", False)
            and gt_bboxes is not None
            and mask_gt is not None
        ):
            gt_sizes = box_area_sqrt(gt_bboxes)
            dynamic_topk = torch.full(
                gt_sizes.shape,
                self.topk,
                dtype=torch.long,
                device=metrics.device,
            )
            tiny_mask = (gt_sizes < float(self.sfqa_cfg.get("tiny_thr", 32.0))) & mask_gt.squeeze(-1).bool()
            if tiny_mask.any():
                topk_tiny = self.topk + int(self.sfqa_cfg.get("topk_tiny_extra", 3))
                topk_floor = max(int(self.sfqa_cfg.get("pos_floor_tiny", 5)), topk_tiny)
                dynamic_topk = torch.where(tiny_mask, torch.full_like(dynamic_topk, topk_floor), dynamic_topk)
        max_topk = int(dynamic_topk.max().item()) if dynamic_topk is not None else self.topk

        # (b, max_num_obj, topk)
        topk_metrics, topk_idxs = torch.topk(metrics, max_topk, dim=-1, largest=True)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        elif topk_mask.shape[-1] != max_topk:
            topk_mask = mask_gt.expand(-1, -1, max_topk).bool() if mask_gt is not None else topk_mask[..., :1].expand_as(topk_idxs)
        if dynamic_topk is not None:
            rank_mask = torch.arange(max_topk, device=metrics.device).view(1, 1, -1) < dynamic_topk.unsqueeze(-1)
            topk_mask = topk_mask & rank_mask & (topk_metrics > self.eps)
        # (b, max_num_obj, topk)
        topk_idxs.masked_fill_(~topk_mask, 0)

        # (b, max_num_obj, topk, h*w) -> (b, max_num_obj, h*w)
        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device)
        for k in range(max_topk):
            # Expand topk_idxs for each value of k and add 1 at the specified positions
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k : k + 1], ones)
        # Filter invalid bboxes
        count_tensor.masked_fill_(count_tensor > 1, 0)

        return count_tensor.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """Compute target labels, target bounding boxes, and target scores for the positive anchor points.

        Args:
            gt_labels (torch.Tensor): Ground truth labels of shape (b, max_num_obj, 1), where b is the batch size and
                max_num_obj is the maximum number of objects.
            gt_bboxes (torch.Tensor): Ground truth bounding boxes of shape (b, max_num_obj, 4).
            target_gt_idx (torch.Tensor): Indices of the assigned ground truth objects for positive anchor points, with
                shape (b, h*w), where h*w is the total number of anchor points.
            fg_mask (torch.Tensor): A boolean tensor of shape (b, h*w) indicating the positive (foreground) anchor
                points.

        Returns:
            target_labels (torch.Tensor): Target labels for positive anchor points with shape (b, h*w).
            target_bboxes (torch.Tensor): Target bounding boxes for positive anchor points with shape (b, h*w, 4).
            target_scores (torch.Tensor): Target scores for positive anchor points with shape (b, h*w, num_classes).
        """
        # Assigned target labels, (b, 1)
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, h*w)
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w)

        # Assigned target boxes, (b, max_num_obj, 4) -> (b, h*w, 4)
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]

        # Assigned target scores
        target_labels.clamp_(0)

        # 10x faster than F.one_hot()
        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),
            dtype=torch.int64,
            device=target_labels.device,
        )  # (b, h*w, 80)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)

        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, 80)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        return target_labels, target_bboxes, target_scores

    def select_candidates_in_gts(self, xy_centers, gt_bboxes, mask_gt, stride_tensor=None, eps=1e-9):
        """Select positive anchor centers within ground truth bounding boxes.

        Args:
            xy_centers (torch.Tensor): Anchor center coordinates, shape (h*w, 2).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes, shape (b, n_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes, shape (b, n_boxes, 1).
            eps (float, optional): Small value for numerical stability.

        Returns:
            (torch.Tensor): Boolean mask of positive anchors, shape (b, n_boxes, h*w).

        Notes:
            - b: batch size, n_boxes: number of ground truth boxes, h: height, w: width.
            - Bounding box format: [x_min, y_min, x_max, y_max].
        """
        gt_bboxes_xywh = xyxy2xywh(gt_bboxes)
        wh_mask = gt_bboxes_xywh[..., 2:] < self.stride[0]  # the smallest stride
        gt_bboxes_xywh[..., 2:] = torch.where(
            (wh_mask * mask_gt).bool(),
            torch.tensor(self.stride_val, dtype=gt_bboxes_xywh.dtype, device=gt_bboxes_xywh.device),
            gt_bboxes_xywh[..., 2:],
        )
        gt_bboxes = xywh2xyxy(gt_bboxes_xywh)

        n_anchors = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
        bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
        candidate_mask = bbox_deltas.amin(3).gt(eps)

        if (
            self.sfqa_enabled
            and self.sfqa_cfg.get("enable_tiny_candidate_relax", False)
            and stride_tensor is not None
            and mask_gt.any()
        ):
            centers = ((gt_bboxes[..., :2] + gt_bboxes[..., 2:4]) * 0.5).unsqueeze(2)
            gt_sizes = box_area_sqrt(gt_bboxes)
            tiny_mask = (gt_sizes < float(self.sfqa_cfg.get("tiny_thr", 32.0))) & mask_gt.squeeze(-1).bool()
            if tiny_mask.any():
                radius = float(self.sfqa_cfg.get("tiny_center_radius_base", 2.0)) * (
                    float(self.sfqa_cfg.get("tiny_thr", 32.0)) / (gt_sizes + float(self.sfqa_cfg.get("tiny_thr", 32.0)))
                )
                radius = radius.clamp(
                    float(self.sfqa_cfg.get("tiny_center_radius_min", 1.5)),
                    float(self.sfqa_cfg.get("tiny_center_radius_max", 3.0)),
                )
                stride_vals = stride_tensor.view(1, 1, n_anchors).to(gt_bboxes.dtype)
                dist2 = ((xy_centers.view(1, 1, n_anchors, 2) - centers) ** 2).sum(dim=-1)
                relax_mask = dist2 <= (radius.unsqueeze(-1) * stride_vals).pow(2)
                candidate_mask = candidate_mask | (relax_mask & tiny_mask.unsqueeze(-1))
        return candidate_mask

    def select_highest_overlaps(self, mask_pos, overlaps, n_max_boxes, align_metric):
        """Select anchor boxes with highest IoU when assigned to multiple ground truths.

        Args:
            mask_pos (torch.Tensor): Positive mask, shape (b, n_max_boxes, h*w).
            overlaps (torch.Tensor): IoU overlaps, shape (b, n_max_boxes, h*w).
            n_max_boxes (int): Maximum number of ground truth boxes.
            align_metric (torch.Tensor): Alignment metric for selecting best matches.

        Returns:
            target_gt_idx (torch.Tensor): Indices of assigned ground truths, shape (b, h*w).
            fg_mask (torch.Tensor): Foreground mask, shape (b, h*w).
            mask_pos (torch.Tensor): Updated positive mask, shape (b, n_max_boxes, h*w).
        """
        # Convert (b, n_max_boxes, h*w) -> (b, h*w)
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)  # (b, n_max_boxes, h*w)

            max_overlaps_idx = overlaps.argmax(1)  # (b, h*w)
            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()  # (b, n_max_boxes, h*w)

            fg_mask = mask_pos.sum(-2)

        if self.topk2 != self.topk:
            align_metric = align_metric * mask_pos  # update overlaps
            max_overlaps_idx = torch.topk(align_metric, self.topk2, dim=-1, largest=True).indices  # (b, n_max_boxes)
            topk_idx = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)  # update mask_pos
            topk_idx.scatter_(-1, max_overlaps_idx, 1.0)
            mask_pos *= topk_idx
            fg_mask = mask_pos.sum(-2)
        # Find each grid serve which gt(index)
        target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
        return target_gt_idx, fg_mask, mask_pos


class RotatedTaskAlignedAssigner(TaskAlignedAssigner):
    """Assigns ground-truth objects to rotated bounding boxes using a task-aligned metric."""

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """Calculate IoU for rotated bounding boxes."""
        return probiou(gt_bboxes, pd_bboxes).squeeze(-1).clamp_(0)

    def select_candidates_in_gts(self, xy_centers, gt_bboxes, mask_gt, stride_tensor=None):
        """Select the positive anchor center in gt for rotated bounding boxes.

        Args:
            xy_centers (torch.Tensor): Anchor center coordinates with shape (h*w, 2).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes with shape (b, n_boxes, 5).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (b, n_boxes, 1).

        Returns:
            (torch.Tensor): Boolean mask of positive anchors with shape (b, n_boxes, h*w).
        """
        wh_mask = gt_bboxes[..., 2:4] < self.stride[0]
        gt_bboxes[..., 2:4] = torch.where(
            (wh_mask * mask_gt).bool(),
            torch.tensor(self.stride_val, dtype=gt_bboxes.dtype, device=gt_bboxes.device),
            gt_bboxes[..., 2:4],
        )

        # (b, n_boxes, 5) --> (b, n_boxes, 4, 2)
        corners = xywhr2xyxyxyxy(gt_bboxes)
        # (b, n_boxes, 1, 2)
        a, b, _, d = corners.split(1, dim=-2)
        ab = b - a
        ad = d - a

        # (b, n_boxes, h*w, 2)
        ap = xy_centers - a
        norm_ab = (ab * ab).sum(dim=-1)
        norm_ad = (ad * ad).sum(dim=-1)
        ap_dot_ab = (ap * ab).sum(dim=-1)
        ap_dot_ad = (ap * ad).sum(dim=-1)
        return (ap_dot_ab >= 0) & (ap_dot_ab <= norm_ab) & (ap_dot_ad >= 0) & (ap_dot_ad <= norm_ad)  # is_in_box


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i in range(len(feats)):  # use len(feats) to avoid TracerWarning from iterating over strides tensor
        stride = strides[i]
        h, w = feats[i].shape[2:] if isinstance(feats, list) else (int(feats[i][0]), int(feats[i][1]))
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_11 else torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat([c_xy, wh], dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def bbox2dist(anchor_points: torch.Tensor, bbox: torch.Tensor, reg_max: int | None = None) -> torch.Tensor:
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    dist = torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1)
    if reg_max is not None:
        dist = dist.clamp_(0, reg_max - 0.01)  # dist (lt, rb)
    return dist


def dist2rbox(pred_dist, pred_angle, anchor_points, dim=-1):
    """Decode predicted rotated bounding box coordinates from anchor points and distribution.

    Args:
        pred_dist (torch.Tensor): Predicted rotated distance with shape (bs, h*w, 4).
        pred_angle (torch.Tensor): Predicted angle with shape (bs, h*w, 1).
        anchor_points (torch.Tensor): Anchor points with shape (h*w, 2).
        dim (int, optional): Dimension along which to split.

    Returns:
        (torch.Tensor): Predicted rotated bounding boxes with shape (bs, h*w, 4).
    """
    lt, rb = pred_dist.split(2, dim=dim)
    cos, sin = torch.cos(pred_angle), torch.sin(pred_angle)
    # (bs, h*w, 1)
    xf, yf = ((rb - lt) / 2).split(1, dim=dim)
    x, y = xf * cos - yf * sin, xf * sin + yf * cos
    xy = torch.cat([x, y], dim=dim) + anchor_points
    return torch.cat([xy, lt + rb], dim=dim)


def rbox2dist(
    target_bboxes: torch.Tensor,
    anchor_points: torch.Tensor,
    target_angle: torch.Tensor,
    dim: int = -1,
    reg_max: int | None = None,
):
    """Transform rotated bounding box (xywh) to distance (ltrb). This is the inverse of dist2rbox.

    Args:
        target_bboxes (torch.Tensor): Target rotated bounding boxes with shape (bs, h*w, 4), format [x, y, w, h].
        anchor_points (torch.Tensor): Anchor points with shape (h*w, 2).
        target_angle (torch.Tensor): Target angle with shape (bs, h*w, 1).
        dim (int, optional): Dimension along which to split.
        reg_max (int, optional): Maximum regression value for clamping.

    Returns:
        (torch.Tensor): Rotated distance with shape (bs, h*w, 4), format [l, t, r, b].
    """
    xy, wh = target_bboxes.split(2, dim=dim)
    offset = xy - anchor_points  # (bs, h*w, 2)
    offset_x, offset_y = offset.split(1, dim=dim)
    cos, sin = torch.cos(target_angle), torch.sin(target_angle)
    xf = offset_x * cos + offset_y * sin
    yf = -offset_x * sin + offset_y * cos

    w, h = wh.split(1, dim=dim)
    target_l = w / 2 - xf
    target_t = h / 2 - yf
    target_r = w / 2 + xf
    target_b = h / 2 + yf

    dist = torch.cat([target_l, target_t, target_r, target_b], dim=dim)
    if reg_max is not None:
        dist = dist.clamp_(0, reg_max - 0.01)

    return dist
