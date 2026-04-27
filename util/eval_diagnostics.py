"""
Eval-only: saliency (|dL/dx| L1) on input to net_i by group, activation at conv1
(mean |F| and global L2), and pooled feature covariance across images (N>=2).
Read-only; does not change checkpoint weights.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from models.errnet_model import dark_channel_map
from util.input_channel_weights import HYPER_COLUMN_CHANNELS


def _build_input_to_net_i(model) -> torch.Tensor:
    """Match ERRNetModel.forward tensor fed to net_i (VGG features detached as in training)."""
    input_i = model.input
    opt = model.opt
    if getattr(opt, "dark_channel", False):
        k = getattr(opt, "dcp_kernel_size", 15)
        dcp = dark_channel_map(model.input, kernel_size=k)
        input_i = torch.cat([input_i, dcp], dim=1)
    if getattr(opt, "hyper", False) and model.vgg is not None:
        hypercolumn = model.vgg(model.input)
        _, _, H, W = model.input.shape
        hypercolumn = [
            F.interpolate(
                feature.detach(),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )
            for feature in hypercolumn
        ]
        input_i = [input_i]
        input_i.extend(hypercolumn)
        input_i = torch.cat(input_i, dim=1)
    return input_i


def _group_sums_from_ch_vec(sc: torch.Tensor, opt) -> dict:
    """sc: (C,) per-channel L1 (or any per-channel stat). Return mean within group."""
    d = {
        "sal_grad_l1_RGB": sc[0:3].mean().item(),
        "sal_grad_l1_DCP": None,
        "sal_grad_l1_hyper": None,
    }
    i = 3
    if getattr(opt, "dark_channel", False):
        d["sal_grad_l1_DCP"] = sc[i].item()
        i += 1
    if getattr(opt, "hyper", False):
        d["sal_grad_l1_hyper"] = sc[i : i + HYPER_COLUMN_CHANNELS].mean().item()
        i += HYPER_COLUMN_CHANNELS
    if i != sc.numel():
        raise ValueError("channel count mismatch in saliency grouping")
    return d


def _align_out_target_spatially(out, target):
    """
    net_i out and GT can differ by 1px on an edge (stride/padding), same as eval vs quality_assess.
    Crop both to the overlapping region so L1/MSE and grad are well-defined.
    """
    h = min(out.shape[2], target.shape[2])
    w = min(out.shape[3], target.shape[3])
    if h < 1 or w < 1:
        return out, target
    return out[:, :, :h, :w], target[:, :, :h, :w]


def _one_saliency(
    model, loss_fn="l1"
):
    """One image; gradient of loss w.r.t. first-layer *input* (concat, as in forward)."""
    model._eval()
    input_i = _build_input_to_net_i(model)
    x = input_i.detach().clone().requires_grad_(True)
    if model.target_t is None:
        return None
    out = model.net_i(x)
    tgt = model.target_t
    out, tgt = _align_out_target_spatially(out, tgt)
    if loss_fn == "l1":
        L = F.l1_loss(out, tgt)
    else:
        L = F.mse_loss(out, tgt)
    g = torch.autograd.grad(L, x, retain_graph=False, create_graph=False)[0]
    # S_c = sum_{h,w} |g_c|  (L1 over space per channel)
    sc = g[0].abs().sum(dim=(1, 2))  # (C,)
    return _group_sums_from_ch_vec(sc, model.opt)


@torch.no_grad()
def _one_conv1_activation_pooled(model):
    """
    One full forward. From conv1 output y: (1) A_c = mean_spatial|F_c|, then
    mean_c A_c, (2) ||F||_2, (3) f = spatial mean of F (for covariance on N images).
    """
    model._eval()
    input_i = _build_input_to_net_i(model)
    hook_out = {}

    def _hook(m, inp, o):
        hook_out["y"] = o

    h = model.net_i.conv1.register_forward_hook(_hook)
    try:
        _ = model.net_i(input_i)
    finally:
        h.remove()
    y = hook_out["y"]
    ac = y.abs().mean(dim=(0, 2, 3))  # (C_f,) mean over space per c
    mean_l1 = ac.mean().item()
    l2 = y.detach().to(torch.float64).view(-1).norm().item()
    vec = y.mean(dim=(0, 2, 3)).float().cpu()  # (C_f,) raw activation mean
    return mean_l1, l2, vec


def run_eval_diagnostics(model, dataloader) -> dict:
    """
    Second pass over val_loader. Only meaningful when model.aligned and GT exists.
    Returns flat dict of averaged scalars; missing branches None; cov fields None if N<2.
    """
    if not hasattr(model, "net_i") or not hasattr(model, "opt"):
        return {}

    n_sal = 0
    sum_sal = {"sal_grad_l1_RGB": 0.0, "sal_grad_l1_DCP": 0.0, "sal_grad_l1_hyper": 0.0}
    has_sal = {"sal_grad_l1_DCP": False, "sal_grad_l1_hyper": False}

    sum_mean_l1 = 0.0
    sum_l2 = 0.0
    n_act = 0

    vecs = []

    was_train = [p.requires_grad for p in model.net_i.parameters()]

    for data in dataloader:
        model.set_input(data, "eval")
        if not model.aligned or model.target_t is None:
            continue

        # Saliency (grad w.r.t. first-layer input)
        for p in model.net_i.parameters():
            p.requires_grad_(False)
        try:
            with torch.enable_grad():
                s = _one_saliency(model, loss_fn="l1")
            if s is not None:
                n_sal += 1
                sum_sal["sal_grad_l1_RGB"] += s["sal_grad_l1_RGB"]
                if s.get("sal_grad_l1_DCP") is not None:
                    has_sal["sal_grad_l1_DCP"] = True
                    sum_sal["sal_grad_l1_DCP"] += s["sal_grad_l1_DCP"]
                if s.get("sal_grad_l1_hyper") is not None:
                    has_sal["sal_grad_l1_hyper"] = True
                    sum_sal["sal_grad_l1_hyper"] += s["sal_grad_l1_hyper"]
        finally:
            for p, rg in zip(model.net_i.parameters(), was_train):
                p.requires_grad_(rg)

        # One forward: activation + pooled vector for covariance
        with torch.no_grad():
            m1, l2, vec = _one_conv1_activation_pooled(model)
            sum_mean_l1 += m1
            sum_l2 += l2
            n_act += 1
            vecs.append(vec)

    out: dict = {}

    if n_sal > 0:
        out["sal_grad_l1_RGB"] = sum_sal["sal_grad_l1_RGB"] / n_sal
        out["sal_grad_l1_DCP"] = (
            None
            if not has_sal["sal_grad_l1_DCP"]
            else sum_sal["sal_grad_l1_DCP"] / n_sal
        )
        out["sal_grad_l1_hyper"] = (
            None
            if not has_sal["sal_grad_l1_hyper"]
            else sum_sal["sal_grad_l1_hyper"] / n_sal
        )
    else:
        out["sal_grad_l1_RGB"] = None
        out["sal_grad_l1_DCP"] = None
        out["sal_grad_l1_hyper"] = None

    if n_act > 0:
        out["act_conv1_mean_l1"] = sum_mean_l1 / n_act
        out["act_conv1_fro_l2"] = sum_l2 / n_act
    else:
        out["act_conv1_mean_l1"] = None
        out["act_conv1_fro_l2"] = None

    N = len(vecs)
    if N >= 2:
        V = torch.stack(vecs, dim=0)  # (N, C)
        Vc = V - V.mean(dim=0, keepdim=True)
        cov = (Vc.t() @ Vc) / (N - 1)
        c = cov.to(torch.float64)
        d = torch.diag(c)
        out["cov_pooled_tr"] = d.mean().item()
        out["cov_pooled_fro"] = c.norm().item()
        c_off = c.clone()
        c_off.fill_diagonal_(0)
        out["cov_pooled_off_fro"] = c_off.norm().item()
    else:
        out["cov_pooled_tr"] = None
        out["cov_pooled_fro"] = None
        out["cov_pooled_off_fro"] = None

    return out
