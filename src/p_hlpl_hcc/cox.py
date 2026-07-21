"""Cox elastic-net hazard layer using PyTorch partial likelihood."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


@dataclass
class CoxElasticNetTorch:
    epochs: int = 300
    learning_rate: float = 0.03
    l1: float = 1e-3
    l2: float = 1e-3
    beta_: np.ndarray | None = None
    scaler_: StandardScaler | None = None

    def fit(self, x: np.ndarray, times: np.ndarray, events: np.ndarray) -> "CoxElasticNetTorch":
        self.scaler_ = StandardScaler().fit(x)
        xs = self.scaler_.transform(x).astype(np.float32)
        event = np.asarray(events, dtype=np.float32)
        if event.sum() < 1:
            self.beta_ = np.zeros(xs.shape[1], dtype=np.float32)
            return self
        order = np.argsort(-np.asarray(times, dtype=float))
        x_t = torch.tensor(xs[order], dtype=torch.float32)
        e_t = torch.tensor(event[order], dtype=torch.float32)
        beta = torch.zeros(xs.shape[1], dtype=torch.float32, requires_grad=True)
        opt = torch.optim.Adam([beta], lr=self.learning_rate)
        denom = torch.clamp(e_t.sum(), min=1.0)
        for _ in range(self.epochs):
            opt.zero_grad(set_to_none=True)
            risk = x_t @ beta
            log_risk = torch.logcumsumexp(risk, dim=0)
            neg_partial = -torch.sum(e_t * (risk - log_risk)) / denom
            penalty = self.l1 * torch.abs(beta).sum() + 0.5 * self.l2 * torch.square(beta).sum()
            loss = neg_partial + penalty
            loss.backward()
            opt.step()
        self.beta_ = beta.detach().cpu().numpy()
        return self

    def predict_log_hazard(self, x: np.ndarray) -> np.ndarray:
        if self.beta_ is None or self.scaler_ is None:
            raise RuntimeError("CoxElasticNetTorch is not fitted")
        return self.scaler_.transform(x) @ self.beta_

    def hazard_ratios(self) -> np.ndarray:
        if self.beta_ is None:
            raise RuntimeError("CoxElasticNetTorch is not fitted")
        return np.exp(self.beta_)

