"""Loss functions used by the P-HLPL-HCC neural head."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class FocalLoss(torch.nn.Module):
    def __init__(
        self,
        gamma: float = 1.5,
        class_weights: list[float] | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be one of: none, mean, sum")
        self.gamma = gamma
        self.reduction = reduction
        if class_weights is None:
            self.register_buffer("class_weights", None)
        else:
            self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weights = self.class_weights
        if weights is not None:
            weights = weights.to(logits.device)
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        target_log_probs = log_probs.gather(1, targets[:, None]).squeeze(1)
        target_probs = probs.gather(1, targets[:, None]).squeeze(1)
        ce = -target_log_probs
        if weights is not None:
            ce = ce * weights[targets]
        loss = ((1.0 - target_probs).clamp_min(1e-8) ** self.gamma) * ce
        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()
        return loss.mean()

