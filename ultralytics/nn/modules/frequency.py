"""Frequency-aware lightweight feature modules."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, DWConv

__all__ = ("FrequencyPreservingReallocationBlock", "TinyGuidedFrequencyGate")


def _reflect_blur(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Apply a local mean blur with reflect padding and stride 1."""
    if kernel_size <= 1:
        return x
    pad = kernel_size // 2
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    return F.avg_pool2d(x_pad, kernel_size=kernel_size, stride=1)


def _zero_init_conv(module: nn.Module) -> None:
    """Zero-initialize the last convolutional projection when requested."""
    if isinstance(module, Conv):
        nn.init.zeros_(module.conv.weight)
        if module.conv.bias is not None:
            nn.init.zeros_(module.conv.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.zeros_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class FrequencyPreservingReallocationBlock(nn.Module):
    """Preserve high-frequency detail with a lightweight residual projector."""

    def __init__(
        self,
        c1: int,
        highpass_kernel: int = 3,
        use_depthwise: bool = True,
        zero_init_proj: bool = True,
        init_gain: float = 1.0,
        max_gain: float = 1.0,
        enabled: bool = True,
        stride: int | None = None,
    ):
        """Initialize the lightweight frequency-preserving block."""
        super().__init__()
        self.enabled = enabled
        self.stride = stride
        self.highpass_kernel = int(highpass_kernel)
        self.max_gain = float(max_gain)
        self.gain = nn.Parameter(torch.tensor(float(init_gain)))
        if use_depthwise:
            self.res_proj = nn.Sequential(DWConv(c1, c1, 3), Conv(c1, c1, 1, act=False))
            proj = self.res_proj[-1]
        else:
            self.res_proj = Conv(c1, c1, 3, act=False)
            proj = self.res_proj
        if zero_init_proj:
            _zero_init_conv(proj)
        self.last_high = None
        self.last_res = None
        self.last_high_abs_mean = 0.0
        self.last_res_abs_mean = 0.0
        self.last_gain = float(init_gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a residual high-frequency correction without changing tensor shape."""
        if not self.enabled:
            return x
        self.last_high = None
        self.last_res = None
        high = x - _reflect_blur(x, self.highpass_kernel)
        res = self.res_proj(high)
        gain = self.gain.clamp(min=-self.max_gain, max=self.max_gain)
        y = x + gain * res
        with torch.no_grad():
            self.last_high_abs_mean = float(high.detach().abs().mean().item())
            self.last_res_abs_mean = float(res.detach().abs().mean().item())
            self.last_gain = float(gain.detach().item())
        return y

    def __getstate__(self):
        """Drop transient tensors before deepcopy/checkpoint serialization."""
        state = self.__dict__.copy()
        for attr in (
            "last_gate_logits",
            "last_gate",
            "last_gate_detached",
            "last_high",
            "last_res",
            "last_energy",
            "last_contrast",
        ):
            if attr in state:
                state[attr] = None
        return state


class TinyGuidedFrequencyGate(nn.Module):
    """Selectively enhance tiny-object high-frequency detail with a supervised gate."""

    def __init__(
        self,
        c1: int,
        reduction: int = 4,
        highpass_kernel: int = 3,
        local_kernel: int = 7,
        init_gain: float = 0.10,
        max_gain: float = 1.00,
        light_proj: bool = False,
        enabled: bool = True,
        stride: int | None = None,
        eps: float = 1e-6,
    ):
        """Initialize the tiny-guided frequency gate."""
        super().__init__()
        self.enabled = enabled
        self.stride = stride
        self.highpass_kernel = int(highpass_kernel)
        self.local_kernel = int(local_kernel)
        self.max_gain = float(max_gain)
        self.eps = float(eps)

        hidden_channels = max(c1 // max(int(reduction), 1), 16)
        self.gate_conv1 = Conv(2 * c1 + 1, hidden_channels, 1)
        self.gate_conv2 = nn.Conv2d(hidden_channels, 1, 1)
        if light_proj:
            self.high_proj = nn.Sequential(DWConv(c1, c1, 3), Conv(c1, c1, 1, act=False))
        else:
            self.high_proj = Conv(c1, c1, 1, act=False)

        init_ratio = min(max(init_gain / max(self.max_gain, self.eps), self.eps), 1.0 - self.eps)
        self.gain_raw = nn.Parameter(torch.tensor(math.log(init_ratio / (1.0 - init_ratio)), dtype=torch.float32))

        self.last_gate_logits: torch.Tensor | None = None
        self.last_gate: torch.Tensor | None = None
        self.last_gate_detached: torch.Tensor | None = None
        self.last_high: torch.Tensor | None = None
        self.last_energy: torch.Tensor | None = None
        self.last_contrast: torch.Tensor | None = None
        self.last_stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a spatial gate over projected high-frequency responses."""
        if not self.enabled:
            self.last_gate_logits = None
            self.last_gate = None
            self.last_gate_detached = None
            self.last_high = None
            self.last_energy = None
            self.last_contrast = None
            return x

        high = x - _reflect_blur(x, self.highpass_kernel)
        energy = high.abs().mean(dim=1, keepdim=True)
        local_energy = _reflect_blur(energy, self.local_kernel)
        contrast = (energy - local_energy) / (energy + local_energy + self.eps)

        gate_input = torch.cat((x, high, contrast), dim=1)
        gate_logits = self.gate_conv2(self.gate_conv1(gate_input))
        gate = torch.sigmoid(gate_logits)
        hp = self.high_proj(high)
        gain = self.max_gain * torch.sigmoid(self.gain_raw)
        y = x + gain * gate * hp

        if self.training and torch.is_grad_enabled():
            self.last_gate_logits = gate_logits
            self.last_gate = gate
            self.last_gate_detached = gate.detach()
        else:
            self.last_gate_logits = None
            self.last_gate = None
            self.last_gate_detached = None
        self.last_high = None
        self.last_energy = None
        self.last_contrast = None
        self.last_stride = self.stride
        return y

    def __getstate__(self):
        """Drop transient tensors before deepcopy/checkpoint serialization."""
        state = self.__dict__.copy()
        for attr in (
            "last_gate_logits",
            "last_gate",
            "last_gate_detached",
            "last_high",
            "last_energy",
            "last_contrast",
        ):
            if attr in state:
                state[attr] = None
        return state
