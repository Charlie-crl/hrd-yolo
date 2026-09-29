"""Shared helpers for FreqRRR Phase 1 configs, logging, and aux cleanup."""

from __future__ import annotations

import math
from copy import deepcopy

import torch
import torch.nn.functional as F

FREQRRR_LOG_KEY_MAP = {
    "fprb_gain_stride8": "fp_g8",
    "fprb_high_abs_mean_stride8": "fp_h8",
    "fprb_res_abs_mean_stride8": "fp_r8",
    "fprb_gain_stride4": "fp_g4",
    "fprb_high_abs_mean_stride4": "fp_h4",
    "fprb_res_abs_mean_stride4": "fp_r4",
    "loss_gate": "tg_l",
    "gate_pos_mean": "tg_p",
    "gate_neg_mean": "tg_n",
    "num_tiny_gt": "tg_nt",
    "fqa_lambda": "fq_lam",
    "fqa_gate_mean": "fq_g",
    "fqa_boost_mean": "fq_bst",
    "fqa_num_tiny_gt": "fq_tiny",
    "rg_l": "rg_l",
    "rg_p": "rg_p",
    "rg_n": "rg_n",
    "rg_gap": "rg_gap",
    "rg_gain": "rg_gain",
    "rg_nt": "rg_nt",
    "ta_l": "ta_l",
    "ta_p": "ta_p",
    "ta_n": "ta_n",
    "ta_num": "ta_num",
    "ta_sum": "ta_sum",
    "ta_mean": "ta_mean",
    "matched_target_bboxes_min": "matched_target_bboxes_min",
    "matched_target_bboxes_max": "matched_target_bboxes_max",
    "matched_area_mean": "matched_area_mean",
    "fa_l": "fa_l",
    "fa_p": "fa_p",
    "fa_n": "fa_n",
    "fa_gap": "fa_gap",
    "fa_e": "fa_e",
    "fa_c": "fa_c",
    "taf_l": "taf_l",
    "taf_p": "taf_p",
    "taf_n": "taf_n",
    "taf_gap": "taf_gap",
    "dt_l": "dt_l",
    "df_l": "df_l",
    "df_w": "df_w",
    "df_p": "df_p",
    "df_n": "df_n",
    "df_gap": "df_gap",
    "df_num": "df_num",
    "df_sum": "df_sum",
    "td_l": "td_l",
    "td_p": "td_p",
    "td_n": "td_n",
    "td_gap": "td_gap",
    "td_g": "td_g",
    "td_nt": "td_nt",
    "fg_l": "fg_l",
    "fg_p": "fg_p",
    "fg_n": "fg_n",
    "fg_gap": "fg_gap",
    "fg_gain": "fg_gain",
    "fg_nt": "fg_nt",
    "fg_e": "fg_e",
    "fg_c": "fg_c",
}

FREQRRR_LOG_KEY_HELP = {
    "fp_g8": "FPRB gain at stride 8",
    "fp_h8": "FPRB high-pass response mean at stride 8",
    "fp_r8": "FPRB residual response mean at stride 8",
    "fp_g4": "FPRB gain at stride 4",
    "fp_h4": "FPRB high-pass response mean at stride 4",
    "fp_r4": "FPRB residual response mean at stride 4",
    "tg_l": "TGF gate loss",
    "tg_p": "TGF positive gate mean",
    "tg_n": "TGF negative gate mean",
    "tg_nt": "number of tiny GTs",
    "fq_lam": "FQA warmup-scaled lambda",
    "fq_g": "mean detached gate used by FQA ranking",
    "fq_bst": "mean FQA frequency boost",
    "fq_tiny": "number of tiny GTs affected by FQA",
    "rg_l": "RRFusion learned spatial gate loss",
    "rg_p": "RRFusion tiny-region spatial gate mean",
    "rg_n": "RRFusion background spatial gate mean",
    "rg_gap": "RRFusion tiny/background gate gap",
    "rg_gain": "RRFusion scalar detail gain",
    "rg_nt": "number of tiny GTs for RRFusion gate",
    "ta_l": "task-aligned RRFusion auxiliary gate loss",
    "ta_p": "task-aligned positive gate mean",
    "ta_n": "task-aligned negative gate mean",
    "ta_num": "number of positive task-aligned gate targets",
    "ta_sum": "sum of the task-aligned gate target map",
    "ta_mean": "mean positive task-aligned gate target value",
    "matched_target_bboxes_min": "minimum matched task-aligned target bbox coordinate",
    "matched_target_bboxes_max": "maximum matched task-aligned target bbox coordinate",
    "matched_area_mean": "mean matched task-aligned target bbox area",
    "fa_l": "frequency-aware auxiliary RRFusion gate loss",
    "fa_p": "frequency-aware auxiliary positive gate mean",
    "fa_n": "frequency-aware auxiliary negative gate mean",
    "fa_gap": "frequency-aware auxiliary positive/negative gate gap",
    "fa_e": "frequency-aware auxiliary energy mean",
    "fa_c": "frequency-aware auxiliary contrast mean",
    "taf_l": "task-aligned frequency-aware auxiliary gate loss",
    "taf_p": "task-aligned frequency-aware auxiliary positive gate mean",
    "taf_n": "task-aligned frequency-aware auxiliary negative gate mean",
    "taf_gap": "task-aligned frequency-aware auxiliary positive/negative gate gap",
    "dt_l": "dual-target tiny-mask RRFusion gate loss",
    "df_l": "dual-target task-aligned frequency auxiliary gate loss",
    "df_w": "dual-target current task-aligned frequency auxiliary loss weight",
    "df_p": "dual-target frequency auxiliary positive gate mean",
    "df_n": "dual-target frequency auxiliary negative gate mean",
    "df_gap": "dual-target frequency auxiliary positive/negative gate gap",
    "df_num": "dual-target task-aligned frequency auxiliary positive target count",
    "df_sum": "dual-target task-aligned frequency auxiliary target sum",
    "td_l": "TGF-detail spatial gate loss",
    "td_p": "TGF-detail tiny-region gate mean",
    "td_n": "TGF-detail background gate mean",
    "td_gap": "TGF-detail tiny/background gate gap",
    "td_g": "TGF-detail scalar high-frequency gain",
    "td_nt": "number of tiny GTs for TGF-detail",
    "fg_l": "frequency-aware RRFusion spatial gate loss",
    "fg_p": "frequency-aware tiny-region spatial gate mean",
    "fg_n": "frequency-aware background spatial gate mean",
    "fg_gap": "frequency-aware tiny/background gate gap",
    "fg_gain": "frequency-aware RRFusion scalar detail gain",
    "fg_nt": "number of tiny GTs for frequency-aware RRFusion",
    "fg_e": "frequency-aware RRFusion energy mean",
    "fg_c": "frequency-aware RRFusion contrast mean",
}

FREQRRR_BASE_LOSS_KEYS = ("box_loss", "cls_loss", "dfl_loss")
FREQRRR_TGF_LOG_KEYS = ("loss_gate", "gate_pos_mean", "gate_neg_mean", "num_tiny_gt")
FREQRRR_FQA_LOG_KEYS = ("fqa_lambda", "fqa_gate_mean", "fqa_boost_mean", "fqa_num_tiny_gt")
FREQRRR_RR_GATE_LOG_KEYS = ("rg_l", "rg_p", "rg_n", "rg_gap", "rg_gain", "rg_nt")
FREQRRR_TA_GATE_LOG_KEYS = ("ta_l", "ta_p", "ta_n", "ta_num", "ta_sum")
FREQRRR_TA_GATE_DEBUG_LOG_KEYS = (
    "matched_target_bboxes_min",
    "matched_target_bboxes_max",
    "matched_area_mean",
    "ta_mean",
)
FREQRRR_FREQ_AUX_LOG_KEYS = ("fa_l", "fa_p", "fa_n", "fa_gap", "fa_e", "fa_c")
FREQRRR_TA_FREQ_AUX_LOG_KEYS = ("taf_l", "taf_p", "taf_n", "taf_gap")
FREQRRR_DUAL_FREQ_AUX_LOG_KEYS = ("dt_l", "df_l", "df_w", "df_p", "df_n", "df_gap", "df_num", "df_sum")
FREQRRR_TGF_DETAIL_LOG_KEYS = ("td_l", "td_p", "td_n", "td_gap", "td_g", "td_nt")
FREQRRR_FREQ_GATE_LOG_KEYS = ("fg_l", "fg_p", "fg_n", "fg_gap", "fg_gain", "fg_nt", "fg_e", "fg_c")

DEFAULT_FREQRRR_FPRB_CFG = {
    "enabled": False,
    "strides": [8],
}

DEFAULT_FREQRRR_TGF_CFG = {
    "enabled": False,
    "strides": [8],
    "gate_loss_weight": 0.05,
}

DEFAULT_FREQRRR_FQA_CFG = {
    "enabled": False,
    "lambda_f": 0.15,
    "warmup_epochs": 10,
    "apply_to_tiny_only": True,
    "detach_gate": True,
}

DEFAULT_FREQRRR_RR_GATE_CFG = {
    "enabled": False,
    "target_mode": "tiny_mask",
    "target_value_mode": "score",
    "loss_weight": 0.10,
    "tiny_only": True,
    "use_learned_spatial_gate": True,
    "use_detail_gain": True,
    "use_freq_aux": False,
    "freq_aux_input_mode": "full",
    "freq_aux_target_mode": "inherit",
    "freq_aux_loss_weight": None,
    "freq_aux_start_epoch": 0,
    "freq_aux_warmup_epochs": 0,
    "detach_freq_aux": True,
}

DEFAULT_FREQRRR_TGF_DETAIL_CFG = {
    "enabled": False,
    "loss_weight": 0.10,
    "use_scalar_gain": True,
}

DEFAULT_FREQRRR_FREQ_GATE_CFG = {
    "enabled": False,
    "loss_weight": 0.10,
    "use_energy": True,
    "use_contrast": True,
    "use_learned_spatial_gate": True,
    "use_detail_gain": True,
}

DEFAULT_FREQRRR_CFG = {
    "enabled": False,
    "tiny_thr": 32.0,
    "rrr": {
        "enabled": True,
        "use_existing_rrr": True,
    },
    "fprb": DEFAULT_FREQRRR_FPRB_CFG,
    "tgf": DEFAULT_FREQRRR_TGF_CFG,
    "fqa": DEFAULT_FREQRRR_FQA_CFG,
    "rr_gate": DEFAULT_FREQRRR_RR_GATE_CFG,
    "tgf_detail": DEFAULT_FREQRRR_TGF_DETAIL_CFG,
    "freq_gate": DEFAULT_FREQRRR_FREQ_GATE_CFG,
    "log": {
        "short_names": True,
        "debug_metrics": False,
    },
}

# Backward-compatible alias for older imports.
FREQRRR_DEFAULTS = DEFAULT_FREQRRR_CFG


def _deep_update(dst: dict, src: dict) -> dict:
    """Recursively merge nested dictionaries."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = deepcopy(v)
    return dst


def _as_int_list(values) -> list[int]:
    """Normalize a list-like stride field to a sorted integer list."""
    if values is None:
        return []
    if not isinstance(values, (list, tuple)):
        values = [values]
    return sorted({int(v) for v in values})


def _ordered_log_strides(values) -> list[int]:
    """Return log strides with stable priority for the common stride-8 view."""
    strides = _as_int_list(values)
    ordered = []
    for preferred in (8, 4):
        if preferred in strides:
            ordered.append(preferred)
            strides.remove(preferred)
    ordered.extend(strides)
    return ordered


def _normalize_freqrrr_schema(cfg: dict | None) -> dict:
    """Accept both the simplified public schema and the legacy nested schema."""
    normalized = deepcopy(cfg or {})
    rrr_cfg = normalized.get("rrr")
    legacy_fprb = None
    if isinstance(rrr_cfg, dict):
        legacy_fprb = rrr_cfg.pop("freq_preserve", None)

    if isinstance(legacy_fprb, dict):
        merged_fprb = deepcopy(legacy_fprb)
        if isinstance(normalized.get("fprb"), dict):
            _deep_update(merged_fprb, normalized["fprb"])
        normalized["fprb"] = merged_fprb

    legacy_sfqa = normalized.pop("sfqa", None)
    if isinstance(legacy_sfqa, dict):
        merged_fqa = deepcopy(legacy_sfqa)
        if isinstance(normalized.get("fqa"), dict):
            _deep_update(merged_fqa, normalized["fqa"])
        normalized["fqa"] = merged_fqa

    normalized.pop("legacy_scl", None)

    if "tiny_thr" not in normalized:
        for section in ("fqa", "tgf"):
            section_cfg = normalized.get(section)
            if isinstance(section_cfg, dict) and "tiny_thr" in section_cfg:
                normalized["tiny_thr"] = section_cfg["tiny_thr"]
                break
    return normalized


def parse_freqrrr_cfg(cfg: dict | None) -> dict:
    """Merge a model freqrrr config with defaults and normalize types."""
    merged = deepcopy(DEFAULT_FREQRRR_CFG)
    normalized_cfg = _normalize_freqrrr_schema(cfg)
    if normalized_cfg:
        _deep_update(merged, normalized_cfg)

    merged["enabled"] = bool(merged["enabled"])
    merged["tiny_thr"] = float(merged["tiny_thr"])
    merged["rrr"]["enabled"] = bool(merged["rrr"]["enabled"])
    merged["rrr"]["use_existing_rrr"] = bool(merged["rrr"]["use_existing_rrr"])
    fp = merged["fprb"]
    fp["enabled"] = bool(fp["enabled"])
    fp["strides"] = _as_int_list(fp["strides"])

    tgf = merged["tgf"]
    tgf["enabled"] = bool(tgf["enabled"])
    tgf["strides"] = _as_int_list(tgf["strides"])
    tgf["gate_loss_weight"] = float(tgf["gate_loss_weight"])

    fqa = merged["fqa"]
    fqa["enabled"] = bool(fqa["enabled"])
    fqa["lambda_f"] = float(fqa["lambda_f"])
    fqa["warmup_epochs"] = int(fqa["warmup_epochs"])
    fqa["apply_to_tiny_only"] = bool(fqa["apply_to_tiny_only"])
    fqa["detach_gate"] = bool(fqa["detach_gate"])
    rr_gate = merged["rr_gate"]
    rr_gate["enabled"] = bool(rr_gate["enabled"])
    rr_gate["target_mode"] = str(rr_gate.get("target_mode", "tiny_mask")).lower()
    if rr_gate["target_mode"] not in {"tiny_mask", "task_aligned"}:
        rr_gate["target_mode"] = "tiny_mask"
    rr_gate["target_value_mode"] = str(rr_gate.get("target_value_mode", "score")).lower()
    if rr_gate["target_value_mode"] not in {"score", "binary"}:
        rr_gate["target_value_mode"] = "score"
    rr_gate["loss_weight"] = float(rr_gate["loss_weight"])
    rr_gate["tiny_only"] = bool(rr_gate.get("tiny_only", True))
    rr_gate["use_learned_spatial_gate"] = bool(rr_gate["use_learned_spatial_gate"])
    rr_gate["use_detail_gain"] = bool(rr_gate["use_detail_gain"])
    rr_gate["use_freq_aux"] = bool(rr_gate.get("use_freq_aux", False))
    input_mode = rr_gate.get("freq_aux_input_mode", "full")
    if input_mode not in ("full", "zero_cues"):
        raise ValueError(f"freqrrr.rr_gate.freq_aux_input_mode must be 'full' or 'zero_cues', got {input_mode!r}.")
    rr_gate["freq_aux_input_mode"] = input_mode
    rr_gate["freq_aux_target_mode"] = str(rr_gate.get("freq_aux_target_mode", "inherit")).lower()
    if rr_gate["freq_aux_target_mode"] not in {"inherit", "tiny_mask", "task_aligned"}:
        rr_gate["freq_aux_target_mode"] = "inherit"
    freq_aux_loss_weight = rr_gate.get("freq_aux_loss_weight", None)
    rr_gate["freq_aux_loss_weight"] = (
        float(rr_gate["loss_weight"]) if freq_aux_loss_weight is None else float(freq_aux_loss_weight)
    )
    rr_gate["freq_aux_start_epoch"] = int(rr_gate.get("freq_aux_start_epoch", 0))
    rr_gate["freq_aux_warmup_epochs"] = int(rr_gate.get("freq_aux_warmup_epochs", 0))
    rr_gate["detach_freq_aux"] = bool(rr_gate.get("detach_freq_aux", True))
    tgf_detail = merged["tgf_detail"]
    tgf_detail["enabled"] = bool(tgf_detail["enabled"])
    tgf_detail["loss_weight"] = float(tgf_detail["loss_weight"])
    tgf_detail["use_scalar_gain"] = bool(tgf_detail["use_scalar_gain"])
    freq_gate = merged["freq_gate"]
    freq_gate["enabled"] = bool(freq_gate["enabled"])
    freq_gate["loss_weight"] = float(freq_gate["loss_weight"])
    freq_gate["use_energy"] = bool(freq_gate["use_energy"])
    freq_gate["use_contrast"] = bool(freq_gate["use_contrast"])
    freq_gate["use_learned_spatial_gate"] = bool(freq_gate["use_learned_spatial_gate"])
    freq_gate["use_detail_gain"] = bool(freq_gate["use_detail_gain"])
    if input_mode == "zero_cues" and (
        not merged["enabled"]
        or not rr_gate["enabled"]
        or not rr_gate["use_freq_aux"]
        or tgf_detail["enabled"]
        or freq_gate["enabled"]
    ):
        raise ValueError(
            "freq_aux_input_mode='zero_cues' requires freqrrr.enabled=true, rr_gate.enabled=true, "
            "use_freq_aux=true and an active RRFusionTinyGate; tgf_detail/freq_gate must be disabled."
        )
    merged["log"]["short_names"] = bool(merged["log"]["short_names"])
    merged["log"]["debug_metrics"] = bool(merged["log"]["debug_metrics"])
    return merged


def get_freqrrr_log_name(name: str, short_names: bool = True) -> str:
    """Map an internal FreqRRR metric name to a shorter external logger key."""
    return FREQRRR_LOG_KEY_MAP.get(name, name) if short_names else name


def get_freqrrr_display_loss_names(loss_keys, freqrrr_cfg: dict | None) -> tuple[str, ...]:
    """Return display names for the selected logged loss keys."""
    cfg = parse_freqrrr_cfg(freqrrr_cfg)
    short_names = bool(cfg["log"]["short_names"])
    return tuple(get_freqrrr_log_name(name, short_names=short_names) for name in loss_keys)


def _get_fprb_log_keys(freqrrr_cfg: dict) -> list[str]:
    """Return active FPRB log keys in the requested display order."""
    if not (freqrrr_cfg["rrr"]["enabled"] and freqrrr_cfg["fprb"]["enabled"]):
        return []

    keys = []
    for stride in _ordered_log_strides(freqrrr_cfg["fprb"].get("strides", [8])):
        keys.extend(
            [
                f"fprb_gain_stride{stride}",
                f"fprb_high_abs_mean_stride{stride}",
                f"fprb_res_abs_mean_stride{stride}",
            ]
        )
    return keys


def _get_freqrrr_debug_loss_keys(freqrrr_cfg: dict) -> list[str]:
    """Return the full ordered set of active FreqRRR debug metrics."""
    if not freqrrr_cfg["enabled"]:
        return []

    keys = _get_fprb_log_keys(freqrrr_cfg)
    if freqrrr_cfg["tgf"]["enabled"]:
        keys.extend(FREQRRR_TGF_LOG_KEYS)
    if freqrrr_cfg["fqa"]["enabled"]:
        keys.extend(FREQRRR_FQA_LOG_KEYS)
    if freqrrr_cfg["rr_gate"]["enabled"]:
        rr_gate = freqrrr_cfg["rr_gate"]
        keys.extend(FREQRRR_TA_GATE_LOG_KEYS if rr_gate.get("target_mode") == "task_aligned" else FREQRRR_RR_GATE_LOG_KEYS)
        if rr_gate.get("target_mode") == "task_aligned":
            keys.extend(FREQRRR_TA_GATE_DEBUG_LOG_KEYS)
        if rr_gate.get("use_freq_aux", False):
            freq_aux_target_mode = rr_gate.get("freq_aux_target_mode", "inherit")
            if rr_gate.get("target_mode") == "tiny_mask" and freq_aux_target_mode == "task_aligned":
                keys.extend(FREQRRR_DUAL_FREQ_AUX_LOG_KEYS)
            else:
                keys.extend(
                    FREQRRR_TA_FREQ_AUX_LOG_KEYS
                    if rr_gate.get("target_mode") == "task_aligned"
                    else FREQRRR_FREQ_AUX_LOG_KEYS
                )
    if freqrrr_cfg["tgf_detail"]["enabled"]:
        keys.extend(FREQRRR_TGF_DETAIL_LOG_KEYS)
    if freqrrr_cfg["freq_gate"]["enabled"]:
        keys.extend(FREQRRR_FREQ_GATE_LOG_KEYS)
    return keys


def _get_freqrrr_core_loss_keys(freqrrr_cfg: dict) -> list[str]:
    """Return the compact default set of active FreqRRR metrics."""
    if not freqrrr_cfg["enabled"]:
        return []

    keys = []
    fprb_keys = _get_fprb_log_keys(freqrrr_cfg)
    if "fprb_gain_stride8" in fprb_keys:
        keys.append("fprb_gain_stride8")
    if freqrrr_cfg["tgf"]["enabled"]:
        keys.extend(("loss_gate", "gate_pos_mean", "gate_neg_mean"))
    if freqrrr_cfg["fqa"]["enabled"]:
        keys.extend(("fqa_lambda", "fqa_gate_mean", "fqa_boost_mean"))
    if freqrrr_cfg["rr_gate"]["enabled"]:
        rr_gate = freqrrr_cfg["rr_gate"]
        if rr_gate.get("target_mode") == "task_aligned":
            keys.extend(("ta_l", "ta_p", "ta_n", "ta_num", "ta_sum"))
        else:
            keys.extend(("rg_l", "rg_p", "rg_n", "rg_gap", "rg_gain"))
        if rr_gate.get("use_freq_aux", False):
            freq_aux_target_mode = rr_gate.get("freq_aux_target_mode", "inherit")
            if rr_gate.get("target_mode") == "tiny_mask" and freq_aux_target_mode == "task_aligned":
                keys = [key for key in keys if key not in {"rg_l", "rg_p", "rg_n", "rg_gap", "rg_gain"}]
                keys.extend(FREQRRR_DUAL_FREQ_AUX_LOG_KEYS)
            elif rr_gate.get("target_mode") == "task_aligned":
                keys.extend(("taf_l", "taf_p", "taf_n", "taf_gap"))
            else:
                keys.extend(("fa_l", "fa_p", "fa_n", "fa_gap", "fa_e", "fa_c"))
    if freqrrr_cfg["tgf_detail"]["enabled"]:
        keys.extend(("td_l", "td_p", "td_n", "td_gap", "td_g"))
    if freqrrr_cfg["freq_gate"]["enabled"]:
        keys.extend(("fg_l", "fg_p", "fg_n", "fg_gap", "fg_gain"))
    return keys


def get_freqrrr_logged_loss_keys(
    loss_names,
    freqrrr_cfg: dict | None,
    include_standard_losses: bool = True,
) -> tuple[str, ...]:
    """Return the ordered raw loss keys that should be emitted to loggers."""
    cfg = parse_freqrrr_cfg(freqrrr_cfg)
    available = set(loss_names)
    extra_keys = _get_freqrrr_debug_loss_keys(cfg) if cfg["log"]["debug_metrics"] else _get_freqrrr_core_loss_keys(cfg)
    selected = [name for name in extra_keys if name in available]
    if include_standard_losses:
        selected = [name for name in FREQRRR_BASE_LOSS_KEYS if name in available] + selected
    return tuple(selected)


def get_freqrrr_progress_loss_keys(loss_names, freqrrr_cfg: dict | None) -> tuple[str, ...]:
    """Return the ordered raw loss keys that should appear in the training progress bar."""
    cfg = parse_freqrrr_cfg(freqrrr_cfg)
    selected = list(get_freqrrr_logged_loss_keys(loss_names, cfg, include_standard_losses=False))
    if selected:
        return tuple(selected)
    fallback = [name for name in FREQRRR_BASE_LOSS_KEYS if name in set(loss_names)]
    return tuple(fallback or tuple(loss_names))


def get_stride_key(stride: int | float) -> str:
    """Return a stable string key for a stride-specific module."""
    return str(int(stride))


FREQRRR_AUX_TENSOR_ATTRS = (
    "last_rr_spatial_gate_logits",
    "last_rr_spatial_gate",
    "last_rr_gate_logits",
    "last_rr_gate",
    "last_freq_aux_spatial_gate_logits",
    "last_freq_aux_spatial_gate",
    "last_tgf_detail_logits",
    "last_tgf_detail_gate",
    "last_freq_spatial_gate_logits",
    "last_freq_spatial_gate",
    "last_freq_gate_logits",
    "last_freq_gate",
    "last_gate_logits",
    "last_gate",
    "last_gate_detached",
    "last_high",
    "last_res",
    "last_energy",
    "last_contrast",
)
FREQRRR_AUX_CONTAINER_ATTRS = ("tgf_aux", "freqrrr_aux", "_tgf_aux", "_freqrrr_aux", "rrgate_aux")


def clear_freqrrr_aux(model):
    """Clear temporary FreqRRR tensors before deepcopy, EMA, checkpoint save, or recovery."""
    if model is None:
        return None

    modules = model.modules() if hasattr(model, "modules") else [model]
    for module in modules:
        for attr in FREQRRR_AUX_TENSOR_ATTRS:
            if hasattr(module, attr):
                setattr(module, attr, None)
        for attr in FREQRRR_AUX_CONTAINER_ATTRS:
            if hasattr(module, attr):
                value = (
                    {"fprb": {}, "tgf": {}, "rr_gate": {}, "freq_aux": {}, "tgf_detail": {}, "freq_gate": {}}
                    if attr == "freqrrr_aux"
                    else None
                )
                setattr(module, attr, value)
    return model


def box_area_sqrt(boxes_xyxy: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Return sqrt(area) from xyxy boxes."""
    wh = (boxes_xyxy[..., 2:4] - boxes_xyxy[..., 0:2]).clamp_min(eps)
    return (wh[..., 0] * wh[..., 1]).clamp_min(eps).sqrt()


def compute_scale_alpha(size_sqrt: torch.Tensor, s0: float, gamma: float, alpha_max: float) -> torch.Tensor:
    """Compute scale-adaptive IoU/NWD mixing weights."""
    size_sqrt = size_sqrt.clamp_min(0)
    alpha = (float(s0) / (size_sqrt + float(s0))).pow(float(gamma))
    return alpha.clamp(0.0, float(alpha_max))


def normalized_wasserstein_similarity(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    nwd_c: float = 12.8,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Compute normalized Wasserstein similarity for xyxy boxes with stable fp16 support."""
    orig_dtype = boxes1.dtype
    compute_dtype = torch.float32
    b1 = boxes1.to(compute_dtype)
    b2 = boxes2.to(compute_dtype)

    c1 = (b1[..., :2] + b1[..., 2:4]) * 0.5
    c2 = (b2[..., :2] + b2[..., 2:4]) * 0.5
    wh1 = (b1[..., 2:4] - b1[..., 0:2]).clamp_min(eps)
    wh2 = (b2[..., 2:4] - b2[..., 0:2]).clamp_min(eps)

    center_d2 = (c1 - c2).pow(2).sum(dim=-1)
    shape_d2 = (wh1 - wh2).pow(2).sum(dim=-1) * 0.25
    d2 = center_d2 + shape_d2
    dist = torch.sqrt(d2.clamp(min=0.0) + float(eps))
    nwd = torch.exp(-dist / max(float(nwd_c), float(eps)))
    nwd = torch.nan_to_num(nwd, nan=0.0, posinf=1.0, neginf=0.0)
    nwd = nwd.clamp(0.0, 1.0)
    return nwd.to(orig_dtype)


def focal_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Compute focal BCE loss for dense gate supervision."""
    logits_f = logits.float()
    target_f = target.float()
    prob = logits_f.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits_f, target_f, reduction="none")
    pt = prob * target_f + (1.0 - prob) * (1.0 - target_f)
    alpha_t = alpha * target_f + (1.0 - alpha) * (1.0 - target_f)
    loss = alpha_t * (1.0 - pt).pow(gamma) * ce
    return loss.mean().to(logits.dtype)


def _target_device(targets) -> torch.device:
    """Infer the target tensor device for dense tiny mask construction."""
    if isinstance(targets, torch.Tensor):
        return targets.device
    if isinstance(targets, dict):
        for key in ("bboxes", "boxes", "gt_bboxes", "batch_idx"):
            value = targets.get(key)
            if isinstance(value, torch.Tensor):
                return value.device
    if isinstance(targets, (list, tuple)):
        for value in targets:
            if isinstance(value, torch.Tensor):
                return value.device
    return torch.device("cpu")


def _image_hw_tensor(image_size, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize image_size to scalar height/width tensors."""
    if isinstance(image_size, torch.Tensor):
        flat = image_size.to(device=device, dtype=dtype).flatten()
        if flat.numel() == 0:
            h = w = torch.tensor(1.0, device=device, dtype=dtype)
        elif flat.numel() == 1:
            h = w = flat[0].clamp_min(1.0)
        else:
            h, w = flat[0].clamp_min(1.0), flat[1].clamp_min(1.0)
    elif isinstance(image_size, (list, tuple)):
        if len(image_size) == 0:
            h = w = torch.tensor(1.0, device=device, dtype=dtype)
        elif len(image_size) == 1:
            h = w = torch.tensor(float(image_size[0]), device=device, dtype=dtype).clamp_min(1.0)
        else:
            h = torch.tensor(float(image_size[0]), device=device, dtype=dtype).clamp_min(1.0)
            w = torch.tensor(float(image_size[1]), device=device, dtype=dtype).clamp_min(1.0)
    else:
        h = w = torch.tensor(float(image_size), device=device, dtype=dtype).clamp_min(1.0)
    return h, w


def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert center xywh boxes to xyxy boxes."""
    xy = boxes[..., :2]
    wh = boxes[..., 2:4]
    return torch.cat((xy - wh * 0.5, xy + wh * 0.5), dim=-1)


def _boxes_to_xyxy_pixels(
    boxes: torch.Tensor,
    image_size,
    box_format: str | None = None,
    normalized: bool | None = None,
) -> torch.Tensor:
    """Convert normalized/pixel xywh/xyxy boxes to pixel-space xyxy boxes."""
    if boxes.numel() == 0:
        return boxes.reshape(-1, 4)

    boxes = torch.nan_to_num(boxes[..., :4].float(), nan=0.0, posinf=0.0, neginf=0.0)
    device, dtype = boxes.device, boxes.dtype
    img_h, img_w = _image_hw_tensor(image_size, device, dtype)
    if normalized is None:
        normalized = bool(boxes.detach().abs().max().item() <= 1.5)
    if box_format is None:
        if normalized:
            box_format = "xywh"
        else:
            valid_xyxy = (boxes[:, 2] >= boxes[:, 0]) & (boxes[:, 3] >= boxes[:, 1])
            box_format = "xyxy" if bool(valid_xyxy.float().mean().item() >= 0.75) else "xywh"
    box_format = str(box_format).lower()

    if normalized:
        scale = torch.stack((img_w, img_h, img_w, img_h))
        boxes = boxes * scale
    boxes_xyxy = _xywh_to_xyxy(boxes) if box_format == "xywh" else boxes
    boxes_xyxy = torch.nan_to_num(boxes_xyxy, nan=0.0, posinf=0.0, neginf=0.0)
    boxes_xyxy[:, 0::2] = boxes_xyxy[:, 0::2].clamp(0.0, float(img_w.item()))
    boxes_xyxy[:, 1::2] = boxes_xyxy[:, 1::2].clamp(0.0, float(img_h.item()))
    return boxes_xyxy


def _tiny_boxes_by_batch(
    targets,
    batch_size: int,
    image_size,
    tiny_thr: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], int]:
    """Extract tiny pixel-space xyxy boxes grouped by batch index."""
    grouped: list[list[torch.Tensor]] = [[] for _ in range(int(batch_size))]
    box_format = None
    normalized = None

    if isinstance(targets, (list, tuple)) and len(targets) >= 2 and isinstance(targets[0], torch.Tensor):
        gt_bboxes = targets[0].to(device=device)
        mask_gt = targets[1].to(device=device).bool()
        if gt_bboxes.ndim == 3:
            for b in range(min(int(batch_size), gt_bboxes.shape[0])):
                valid = mask_gt[b].reshape(-1) if mask_gt.ndim >= 2 else torch.ones(gt_bboxes.shape[1], device=device).bool()
                if valid.numel() != gt_bboxes.shape[1]:
                    valid = valid[: gt_bboxes.shape[1]]
                boxes = _boxes_to_xyxy_pixels(gt_bboxes[b, valid, :4], image_size, box_format="xyxy")
                if boxes.numel():
                    grouped[b].append(boxes)
    elif isinstance(targets, dict):
        boxes = targets.get("bboxes", targets.get("boxes", targets.get("gt_bboxes")))
        if boxes is not None:
            boxes = boxes.to(device=device)
            batch_idx = targets.get("batch_idx")
            box_format = targets.get("bbox_format", targets.get("box_format", "xywh"))
            normalized = targets.get("normalized", None)
            if isinstance(normalized, torch.Tensor):
                normalized = bool(normalized.item())
            if boxes.ndim == 3:
                for b in range(min(int(batch_size), boxes.shape[0])):
                    valid = boxes[b].sum(dim=-1).ne(0.0)
                    grouped[b].append(_boxes_to_xyxy_pixels(boxes[b, valid], image_size, box_format, normalized))
            else:
                if batch_idx is None:
                    batch_idx = torch.zeros(boxes.shape[0], device=device, dtype=torch.long)
                batch_idx = batch_idx.to(device=device).long().reshape(-1)
                boxes_xyxy = _boxes_to_xyxy_pixels(boxes, image_size, box_format, normalized)
                for b in range(int(batch_size)):
                    select = batch_idx == b
                    if select.any():
                        grouped[b].append(boxes_xyxy[select])
    elif isinstance(targets, torch.Tensor):
        tensor = targets.to(device=device)
        if tensor.ndim == 3:
            boxes = tensor[..., 1:5] if tensor.shape[-1] >= 5 else tensor[..., :4]
            for b in range(min(int(batch_size), boxes.shape[0])):
                valid = boxes[b].sum(dim=-1).ne(0.0)
                grouped[b].append(_boxes_to_xyxy_pixels(boxes[b, valid], image_size, None, None))
        elif tensor.ndim == 2 and tensor.numel():
            if tensor.shape[1] >= 6:
                batch_idx = tensor[:, 0].long()
                boxes = tensor[:, 2:6]
            elif tensor.shape[1] >= 5:
                batch_idx = tensor[:, 0].long()
                boxes = tensor[:, 1:5]
            else:
                batch_idx = torch.zeros(tensor.shape[0], device=device, dtype=torch.long)
                boxes = tensor[:, :4]
            boxes_xyxy = _boxes_to_xyxy_pixels(boxes, image_size, None, None)
            for b in range(int(batch_size)):
                select = batch_idx == b
                if select.any():
                    grouped[b].append(boxes_xyxy[select])

    out: list[tuple[torch.Tensor, torch.Tensor]] = []
    total_tiny = 0
    for boxes_list in grouped:
        non_empty = [b for b in boxes_list if b.numel()]
        if non_empty:
            boxes = torch.cat(non_empty, dim=0)
        else:
            boxes = torch.zeros(0, 4, device=device, dtype=torch.float32)
        boxes = boxes.to(device=device, dtype=dtype)
        wh = (boxes[:, 2:4] - boxes[:, 0:2]).clamp_min(0.0)
        sizes = (wh[:, 0] * wh[:, 1]).clamp_min(0.0).sqrt()
        tiny = sizes < float(tiny_thr)
        total_tiny += int(tiny.sum().item())
        out.append((boxes[tiny], sizes[tiny]))
    return out, total_tiny


def build_tiny_mask_and_count(
    targets,
    stride: int | float,
    feature_hw: tuple[int, int],
    batch_size: int,
    image_size,
    tiny_thr: float = 32.0,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, int]:
    """Build a dense tiny-object Gaussian mask and return the tiny-GT count."""
    device = device or _target_device(targets)
    dtype = dtype or torch.float32
    h, w = int(feature_hw[0]), int(feature_hw[1])
    tiny_mask = torch.zeros(int(batch_size), 1, h, w, device=device, dtype=dtype)
    tiny_boxes, num_tiny_gt = _tiny_boxes_by_batch(targets, batch_size, image_size, tiny_thr, device, dtype)
    if h <= 0 or w <= 0 or num_tiny_gt == 0:
        return tiny_mask, num_tiny_gt

    stride_f = max(float(stride), 1.0)
    for b, (boxes, sizes) in enumerate(tiny_boxes):
        if boxes.numel() == 0:
            continue
        centers = (boxes[:, :2] + boxes[:, 2:4]) * 0.5
        for center, size in zip(centers, sizes):
            cx = center[0] / stride_f
            cy = center[1] / stride_f
            sigma = ((size / stride_f) * 0.25).clamp(1.0, 4.0)
            radius = int(math.ceil(3.0 * float(sigma.detach().item())))
            cx_floor = int(math.floor(float(cx.detach().item())))
            cy_floor = int(math.floor(float(cy.detach().item())))
            x0 = max(cx_floor - radius, 0)
            x1 = min(cx_floor + radius, w - 1)
            y0 = max(cy_floor - radius, 0)
            y1 = min(cy_floor + radius, h - 1)
            if x1 < x0 or y1 < y0:
                continue
            yy = torch.arange(y0, y1 + 1, device=device, dtype=dtype)
            xx = torch.arange(x0, x1 + 1, device=device, dtype=dtype)
            yy, xx = torch.meshgrid(yy, xx, indexing="ij")
            gaussian = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma.to(dtype) ** 2))
            tiny_mask[b, 0, y0 : y1 + 1, x0 : x1 + 1] = torch.maximum(
                tiny_mask[b, 0, y0 : y1 + 1, x0 : x1 + 1], gaussian
            )
    return torch.nan_to_num(tiny_mask, nan=0.0, posinf=1.0, neginf=0.0), num_tiny_gt


def build_tiny_mask(
    targets,
    stride: int | float,
    feature_hw: tuple[int, int],
    batch_size: int,
    image_size,
    tiny_thr: float = 32.0,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Build a [B, 1, H, W] Gaussian mask for tiny-object gate supervision."""
    tiny_mask, _ = build_tiny_mask_and_count(
        targets, stride, feature_hw, batch_size, image_size, tiny_thr, device=device, dtype=dtype
    )
    return tiny_mask


def build_tiny_gaussian_masks(
    gt_bboxes: torch.Tensor,
    mask_gt: torch.Tensor,
    feature_shapes: dict[int, tuple[int, int]],
    tiny_thr: float,
    sigma_scale: float,
    min_sigma: float,
    max_sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[int, torch.Tensor], int]:
    """Build per-level Gaussian masks for tiny-object gate supervision."""
    bs = gt_bboxes.shape[0]
    masks = {
        int(stride): torch.zeros(bs, 1, shape[0], shape[1], device=device, dtype=dtype)
        for stride, shape in feature_shapes.items()
    }
    if not masks:
        return masks, 0

    total_tiny = 0
    for b in range(bs):
        valid = mask_gt[b, :, 0].bool()
        if not valid.any():
            continue
        boxes = gt_bboxes[b, valid]
        sizes = box_area_sqrt(boxes)
        tiny_mask = sizes < float(tiny_thr)
        if not tiny_mask.any():
            continue
        boxes = boxes[tiny_mask]
        sizes = sizes[tiny_mask]
        total_tiny += int(tiny_mask.sum().item())
        centers = (boxes[:, :2] + boxes[:, 2:4]) * 0.5
        for stride, feat_mask in masks.items():
            h, w = feat_mask.shape[-2:]
            for center, size in zip(centers, sizes):
                cx = center[0] / float(stride)
                cy = center[1] / float(stride)
                sigma = ((size / float(stride)) * float(sigma_scale)).clamp(float(min_sigma), float(max_sigma))
                radius = int(math.ceil(3.0 * float(sigma.item())))
                cx_floor = int(math.floor(float(cx.item())))
                cy_floor = int(math.floor(float(cy.item())))
                x0 = max(cx_floor - radius, 0)
                x1 = min(cx_floor + radius, w - 1)
                y0 = max(cy_floor - radius, 0)
                y1 = min(cy_floor + radius, h - 1)
                if x1 < x0 or y1 < y0:
                    continue
                yy = torch.arange(y0, y1 + 1, device=device, dtype=dtype)
                xx = torch.arange(x0, x1 + 1, device=device, dtype=dtype)
                yy, xx = torch.meshgrid(yy, xx, indexing="ij")
                gaussian = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma.to(dtype) ** 2))
                feat_mask[b, 0, y0 : y1 + 1, x0 : x1 + 1] = torch.maximum(
                    feat_mask[b, 0, y0 : y1 + 1, x0 : x1 + 1], gaussian
                )
    return masks, total_tiny
