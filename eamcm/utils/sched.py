"""Learning-rate schedule (linear warmup + cosine decay) and optimizer factory."""

import math
from typing import Optional

import torch


def adjust_lr(
    optimizer: torch.optim.Optimizer,
    base_lr: float,
    step: int,
    total_steps: int,
    warmup_steps: int,
    min_lr: float = 1e-6,
) -> float:
    """Set per-param-group LR for warmup+cosine; returns the applied LR."""
    if step < warmup_steps:
        lr = base_lr * step / max(1, warmup_steps)
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = lr * group.get("lr_scale", 1.0)
    return lr


def build_optimizer(
    model: torch.nn.Module, lr: float, weight_decay: float, betas=(0.9, 0.95)
) -> torch.optim.AdamW:
    """AdamW excluding bias / 1-D params / positional & token parameters from decay."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            param.ndim <= 1
            or name.endswith(".pos_embed")
            or name.endswith(".decoder_pos_embed")
            or "cls_token" in name
            or "mask_token" in name
        ):
            no_decay.append(param)
        else:
            decay.append(param)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=betas)


class GradScalerAMP:
    """Thin wrapper so pretrain/finetune scripts share one AMP code path."""

    def __init__(self, enabled: bool) -> None:
        self.scaler = torch.amp.GradScaler("cuda", enabled=enabled)
        self.enabled = enabled

    def backward(self, loss: torch.Tensor) -> None:
        self.scaler.scale(loss).backward()

    def step(
        self,
        optimizer: torch.optim.Optimizer,
        clip_grad: Optional[float] = None,
        model: Optional[torch.nn.Module] = None,
    ) -> float:
        norm = 0.0
        if clip_grad is not None and model is not None:
            self.scaler.unscale_(optimizer)
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad))
        self.scaler.step(optimizer)
        self.scaler.update()
        return norm
