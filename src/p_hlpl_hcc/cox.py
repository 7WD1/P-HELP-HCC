"""Cox elastic-net hazard layer with grouped Breslow ties."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


def breslow_negative_partial_log_likelihood(
    log_risk: torch.Tensor,
    times: np.ndarray | torch.Tensor,
    events: np.ndarray | torch.Tensor,
) -> torch.Tensor:
    """Return the negative Cox partial log-likelihood with Breslow ties."""

    if log_risk.ndim != 1:
        raise ValueError("log_risk must be one-dimensional")
    time_t = torch.as_tensor(times, dtype=log_risk.dtype, device=log_risk.device)
    event_t = torch.as_tensor(events, dtype=log_risk.dtype, device=log_risk.device)
    if time_t.shape != log_risk.shape or event_t.shape != log_risk.shape:
        raise ValueError("times, events, and log_risk must be aligned")
    if not torch.all((event_t == 0) | (event_t == 1)):
        raise ValueError("events must contain only 0/1 values")
    event_times = torch.unique(time_t[event_t == 1], sorted=True)
    if event_times.numel() == 0:
        return log_risk.sum() * 0.0

    event_rows = (time_t.unsqueeze(0) == event_times.unsqueeze(1)) & (
        event_t.unsqueeze(0) == 1
    )
    risk_rows = time_t.unsqueeze(0) >= event_times.unsqueeze(1)
    deaths = event_rows.sum(dim=1).to(log_risk.dtype)
    event_score = (event_rows.to(log_risk.dtype) * log_risk.unsqueeze(0)).sum(dim=1)
    risk_score = log_risk.unsqueeze(0).expand(event_times.numel(), -1)
    log_denominator = torch.logsumexp(
        risk_score.masked_fill(~risk_rows, -torch.inf), dim=1
    )
    partial = (event_score - deaths * log_denominator).sum()
    return -partial


def breslow_baseline_hazard(
    log_risk: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return event times, Breslow increments, and cumulative baseline hazard.

    Risk-set denominators are evaluated on the log scale. This preserves the
    Breslow formula for large linear predictors without clipping risk scores.
    """

    score = np.asarray(log_risk, dtype=np.float64)
    time = np.asarray(times, dtype=np.float64)
    event = np.asarray(events, dtype=np.float64)
    if score.ndim != 1 or time.shape != score.shape or event.shape != score.shape:
        raise ValueError("log_risk, times, and events must be aligned vectors")
    if not set(np.unique(event)).issubset({0.0, 1.0}):
        raise ValueError("events must contain only 0/1 values")
    event_times = np.sort(np.unique(time[event == 1]))
    if event_times.size == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty.copy(), empty.copy()

    increments: list[float] = []
    for event_time in event_times:
        deaths = float(np.sum((time == event_time) & (event == 1)))
        log_denominator = float(np.logaddexp.reduce(score[time >= event_time]))
        increments.append(float(np.exp(np.log(deaths) - log_denominator)))
    increment_array = np.asarray(increments, dtype=np.float64)
    return event_times, increment_array, np.cumsum(increment_array)


@dataclass
class CoxElasticNetTorch:
    epochs: int = 300
    learning_rate: float = 0.03
    l1: float = 1e-3
    l2: float = 1e-3
    beta_: np.ndarray | None = None
    scaler_: StandardScaler | None = None
    baseline_event_times_: np.ndarray | None = None
    baseline_hazard_increments_: np.ndarray | None = None
    baseline_cumulative_hazard_: np.ndarray | None = None

    def fit(self, x: np.ndarray, times: np.ndarray, events: np.ndarray) -> "CoxElasticNetTorch":
        self.scaler_ = StandardScaler().fit(x)
        xs = self.scaler_.transform(x).astype(np.float32)
        event = np.asarray(events, dtype=np.float32)
        time = np.asarray(times, dtype=float)
        if time.shape != event.shape or xs.shape[0] != len(time):
            raise ValueError("x, times, and events must contain the same number of rows")
        if not set(np.unique(event)).issubset({0.0, 1.0}):
            raise ValueError("events must contain only 0/1 values")
        if event.sum() < 1:
            self.beta_ = np.zeros(xs.shape[1], dtype=np.float32)
            self.baseline_event_times_ = np.array([], dtype=float)
            self.baseline_hazard_increments_ = np.array([], dtype=float)
            self.baseline_cumulative_hazard_ = np.array([], dtype=float)
            return self
        x_t = torch.tensor(xs, dtype=torch.float32)
        beta = torch.zeros(xs.shape[1], dtype=torch.float32, requires_grad=True)
        opt = torch.optim.Adam([beta], lr=self.learning_rate)
        for _ in range(self.epochs):
            opt.zero_grad(set_to_none=True)
            risk = x_t @ beta
            neg_partial = breslow_negative_partial_log_likelihood(risk, time, event)
            penalty = self.l1 * torch.abs(beta).sum() + self.l2 * torch.square(beta).sum()
            loss = neg_partial + penalty
            loss.backward()
            opt.step()
        self.beta_ = beta.detach().cpu().numpy()
        event_times, increments, cumulative = breslow_baseline_hazard(
            xs.astype(np.float64) @ self.beta_.astype(np.float64), time, event
        )
        self.baseline_event_times_ = event_times.astype(float)
        self.baseline_hazard_increments_ = increments
        self.baseline_cumulative_hazard_ = cumulative
        return self

    def baseline_cumulative_hazard_at(self, times: np.ndarray | list[float]) -> np.ndarray:
        """Evaluate the fitted Breslow baseline cumulative hazard."""

        if self.baseline_event_times_ is None or self.baseline_cumulative_hazard_ is None:
            raise RuntimeError("CoxElasticNetTorch is not fitted")
        query = np.asarray(times, dtype=float)
        indices = np.searchsorted(self.baseline_event_times_, query, side="right") - 1
        result = np.zeros(query.shape, dtype=float)
        valid = indices >= 0
        result[valid] = self.baseline_cumulative_hazard_[indices[valid]]
        return result

    def predict_log_hazard(self, x: np.ndarray) -> np.ndarray:
        if self.beta_ is None or self.scaler_ is None:
            raise RuntimeError("CoxElasticNetTorch is not fitted")
        return self.scaler_.transform(x) @ self.beta_

    def hazard_ratios(self) -> np.ndarray:
        if self.beta_ is None:
            raise RuntimeError("CoxElasticNetTorch is not fitted")
        return np.exp(self.beta_)

