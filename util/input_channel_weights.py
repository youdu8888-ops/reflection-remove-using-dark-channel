"""
Read-only statistics on *trained* conv1 weights (no parameters are modified).

Input channel order matches models.errnet_model.ERRNetModel.forward:
RGB(3) [+ dark-channel prior DCP(1)] [+ VGG hypercolumn 1472ch].

Each reported scalar is the **mean of absolute weights** mean(|W|): first mean
over output channels and spatial kernel positions for each input channel, then
mean over all input channels in that group (RGB / DCP / hyper). This is not a
signed average, so negative and positive weights both count as magnitude.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Must match errnet_model.ERRNetModel.initialize: in_channels += 1472 when --hyper
HYPER_COLUMN_CHANNELS = 1472


def first_conv_group_mean_abs_weight(
    net_i: nn.Module, opt, hyper_ch: int = HYPER_COLUMN_CHANNELS
) -> dict:
    """
    Summarize existing learned weights in the first conv (read-only tensor read).

    Returns keys: w_mean_abs_RGB, w_mean_abs_dark, w_mean_abs_hyper
    (missing branches are None / JSON null — not used in that checkpoint).
    """
    w = net_i.conv1.conv2d.weight.detach()
    if w.dim() != 4:
        raise ValueError("expected 4D conv1 weight")
    in_c = w.shape[1]
    # per input channel: mean |W| over (out, kh, kw)
    m = w.abs().mean(dim=(0, 2, 3))  # [in_c]
    d: dict = {
        "w_mean_abs_RGB": m[0:3].mean().item(),
        "w_mean_abs_dark": None,
        "w_mean_abs_hyper": None,
    }
    i = 3
    if getattr(opt, "dark_channel", False):
        d["w_mean_abs_dark"] = m[i].item()
        i += 1
    if getattr(opt, "hyper", False):
        d["w_mean_abs_hyper"] = m[i : i + hyper_ch].mean().item()
        i += hyper_ch
    if i != in_c:
        raise ValueError(
            "input channel layout mismatch: opt implies %d ch, conv1 has %d in ch"
            % (i, in_c)
        )
    return d


def ch_weight_ratio_reg_loss(
    net_i: nn.Module,
    opt,
    hyper_ch: int = HYPER_COLUMN_CHANNELS,
    eps: float = 1e-8,
    alpha: float = 0.15,
) -> torch.Tensor:
    """
    Differentiable regularizer on first-conv *mean |W|* per input group (same grouping as
    first_conv_group_mean_abs_weight).     For each enabled extra group (DCP and/or hyper) define

        t_g = max(0, w_group / (w_rgb + eps) - alpha) ** 2

    w_rgb and w_group come from mean(|W|), so the ratio is already nonnegative; no extra abs.

    L_reg = mean of {t_g} over enabled groups (1 or 2 terms), not the sum, so "both" does not
    double the penalty weight versus a single extra branch.
    """
    w = net_i.conv1.conv2d.weight
    if w.dim() != 4:
        raise ValueError("expected 4D conv1 weight")
    m = w.abs().mean(dim=(0, 2, 3))  # [in_c]
    w_rgb = m[0:3].mean()
    i = 3
    terms = []
    if getattr(opt, "dark_channel", False):
        w_dcp = m[i]
        i += 1
        r = w_dcp / (w_rgb + eps) - alpha
        terms.append(F.relu(r) ** 2)
    if getattr(opt, "hyper", False):
        w_hyper = m[i : i + hyper_ch].mean()
        i += hyper_ch
        r = w_hyper / (w_rgb + eps) - alpha
        terms.append(F.relu(r) ** 2)
    in_c = w.shape[1]
    if i != in_c:
        raise ValueError(
            "input channel layout mismatch: opt implies %d ch, conv1 has %d in ch"
            % (i, in_c)
        )
    if not terms:
        return w_rgb * 0
    return torch.stack(terms).mean()
