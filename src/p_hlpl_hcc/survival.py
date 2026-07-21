"""Censoring-aware discrete-time survival utilities.

The helpers in this module deliberately keep right-censored observations out
of hard interval labels.  They construct the person-period likelihood masks
needed to use all observed follow-up without pretending that an immature
censored patient has reached a definitive survival class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F


def _validated_inputs(
    times: Iterable[float], events: Iterable[int], cutpoints: Iterable[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time = np.asarray(list(times), dtype=float)
    event = np.asarray(list(events), dtype=int)
    cuts = np.asarray(list(cutpoints), dtype=float)
    if time.ndim != 1 or event.ndim != 1 or len(time) != len(event):
        raise ValueError("times and events must be aligned one-dimensional arrays")
    if len(cuts) == 0 or not np.all(np.isfinite(cuts)):
        raise ValueError("cutpoints must contain at least one finite value")
    if not np.all(np.diff(cuts) > 0) or cuts[0] <= 0:
        raise ValueError("cutpoints must be strictly increasing and positive")
    if not np.all(np.isfinite(time)) or np.any(time < 0):
        raise ValueError("survival times must be finite and non-negative")
    if not set(np.unique(event)).issubset({0, 1}):
        raise ValueError("events must contain only 0/1 indicators")
    return time, event, cuts


@dataclass(frozen=True)
class DiscreteTimeTargets:
    """Person-period representation for a discrete hazard likelihood.

    ``likelihood_mask`` includes an event's failure interval.  For a censored
    observation it includes only intervals whose upper boundary was actually
    reached, so partial follow-up is never converted into a known non-event.
    ``risk_set_mask`` is supplied separately for cohort/time audit counts.
    """

    event_targets: np.ndarray
    likelihood_mask: np.ndarray
    risk_set_mask: np.ndarray
    event_interval: np.ndarray
    censor_interval: np.ndarray
    cutpoints: np.ndarray


@dataclass(frozen=True)
class CensoringKaplanMeier:
    """Reverse-KM censoring survival fitted on a training fold only."""

    timeline: np.ndarray
    survival_at_time: np.ndarray

    def lookup(self, times: Iterable[float], *, floor: float = 1e-3) -> np.ndarray:
        """Evaluate the right-continuous censoring survival G(t)."""

        query = np.asarray(list(times), dtype=float)
        indices = np.searchsorted(self.timeline, query, side="right") - 1
        result = np.ones(query.shape, dtype=float)
        observed = indices >= 0
        result[observed] = self.survival_at_time[indices[observed]]
        return np.clip(result, floor, 1.0)


def fit_censoring_kaplan_meier(
    times: Iterable[float], events: Iterable[int]
) -> CensoringKaplanMeier:
    """Fit G(t)=P(C>=t) using death as censoring for reverse KM."""

    time = np.asarray(list(times), dtype=float)
    event = np.asarray(list(events), dtype=int)
    if time.ndim != 1 or event.shape != time.shape:
        raise ValueError("times and events must be aligned one-dimensional arrays")
    if not set(np.unique(event)).issubset({0, 1}):
        raise ValueError("events must contain only 0/1 indicators")
    survival = 1.0
    timeline: list[float] = []
    survival_at_time: list[float] = []
    for observed_time in np.unique(time):
        at_risk = int(np.sum(time >= observed_time))
        censored = int(np.sum((time == observed_time) & (event == 0)))
        if at_risk > 0 and censored:
            survival *= 1.0 - censored / at_risk
        timeline.append(float(observed_time))
        survival_at_time.append(float(survival))
    return CensoringKaplanMeier(np.asarray(timeline), np.asarray(survival_at_time))


def ipcw_likelihood_cell_weights(
    times: Iterable[float],
    events: Iterable[int],
    targets: DiscreteTimeTargets,
    censoring_km: CensoringKaplanMeier,
    *,
    floor: float = 1e-3,
) -> np.ndarray:
    """Return Eq.(dt-lik) cell weights: 1/G(T) or 1/G(U_c).

    Observed failure cells use ``1/G(T_i)``.  Each preceding survival cell
    uses ``1/G(U_c)`` at that interval's upper boundary.  Unobserved cells
    remain zero.  Thus weights vary by both patient and interval rather than
    approximating the likelihood with a single patient-level weight.
    The reverse-KM estimator must be fitted on the training fold and reused
    unchanged for validation and test evaluation.
    """

    time = np.asarray(list(times), dtype=float)
    event = np.asarray(list(events), dtype=int)
    if time.ndim != 1 or event.shape != time.shape:
        raise ValueError("times and events must be aligned one-dimensional arrays")
    if targets.likelihood_mask.shape[0] != len(time):
        raise ValueError("targets must align with times/events")
    weights = np.zeros_like(targets.event_targets, dtype=np.float32)
    survival_weights = 1.0 / censoring_km.lookup(targets.cutpoints, floor=floor)
    for interval_index, weight in enumerate(survival_weights):
        survival_cell = targets.likelihood_mask[:, interval_index] & (
            targets.event_targets[:, interval_index] == 0
        )
        weights[survival_cell, interval_index] = float(weight)
    event_rows, event_columns = np.nonzero(targets.event_targets == 1)
    if len(event_rows):
        event_g = censoring_km.lookup(time[event_rows], floor=floor)
        weights[event_rows, event_columns] = (1.0 / event_g).astype(np.float32)
    return weights


def make_discrete_time_targets(
    times: Iterable[float], events: Iterable[int], cutpoints: Iterable[float]
) -> DiscreteTimeTargets:
    time, event, cuts = _validated_inputs(times, events, cutpoints)
    starts = np.concatenate(([0.0], cuts[:-1]))
    n, k = len(time), len(cuts)

    # Intervals are [0,c1), [c1,c2), ...; an event exactly on a boundary is
    # assigned to the interval beginning at that boundary.
    interval = np.searchsorted(cuts, time, side="right")
    event_interval = np.where((event == 1) & (interval < k), interval, -1)
    censor_interval = np.where(event == 0, np.minimum(interval, k - 1), -1)

    event_targets = np.zeros((n, k), dtype=np.float32)
    event_rows = np.flatnonzero(event_interval >= 0)
    event_targets[event_rows, event_interval[event_rows]] = 1.0

    risk_set_mask = time[:, None] >= starts[None, :]
    likelihood_mask = np.zeros((n, k), dtype=bool)
    for row in range(n):
        if event_interval[row] >= 0:
            likelihood_mask[row, : event_interval[row] + 1] = True
        elif event[row] == 0:
            # A censored row contributes survival only through complete bins.
            likelihood_mask[row] = cuts <= time[row]
        else:
            # Event after the analysis horizon: known event-free through it.
            likelihood_mask[row] = cuts <= time[row]

    return DiscreteTimeTargets(
        event_targets=event_targets,
        likelihood_mask=likelihood_mask,
        risk_set_mask=risk_set_mask,
        event_interval=event_interval.astype(int),
        censor_interval=censor_interval.astype(int),
        cutpoints=cuts,
    )


def risk_event_censor_counts(
    times: Iterable[float], events: Iterable[int], cutpoints: Iterable[float]
) -> list[dict[str, object]]:
    """Return cohort/time-specific risk-set, event, and censoring counts."""

    time, event, cuts = _validated_inputs(times, events, cutpoints)
    starts = np.concatenate(([0.0], cuts[:-1]))
    interval = np.searchsorted(cuts, time, side="right")
    counts: list[dict[str, object]] = []
    for idx, (start, stop) in enumerate(zip(starts, cuts)):
        counts.append(
            {
                "interval": idx,
                "start_month": float(start),
                "stop_month": float(stop),
                "risk_set": int(np.sum(time >= start)),
                "events": int(np.sum((event == 1) & (interval == idx))),
                "censored": int(np.sum((event == 0) & (interval == idx))),
            }
        )
    tail_at_risk = time >= cuts[-1]
    counts.append(
        {
            "interval": len(cuts),
            "start_month": float(cuts[-1]),
            "stop_month": None,
            "risk_set": int(np.sum(tail_at_risk)),
            "events": 0,
            "censored": int(np.sum(tail_at_risk)),
            "administrative_censoring_at_horizon": int(np.sum(tail_at_risk)),
            "observed_events_after_horizon": int(np.sum(tail_at_risk & (event == 1))),
        }
    )
    return counts


def discrete_time_negative_log_likelihood(
    logits: torch.Tensor,
    event_targets: torch.Tensor,
    likelihood_mask: torch.Tensor,
    *,
    cell_weights: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Bernoulli discrete-hazard negative log likelihood."""

    if logits.shape != event_targets.shape or logits.shape != likelihood_mask.shape:
        raise ValueError("logits, event_targets, and likelihood_mask must have equal shapes")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be one of: none, mean, sum")
    mask = likelihood_mask.to(dtype=logits.dtype)
    if cell_weights is None:
        weights = torch.ones_like(logits)
    else:
        if cell_weights.shape != logits.shape:
            raise ValueError("cell_weights must contain one IPCW value per patient/interval")
        weights = cell_weights.to(dtype=logits.dtype, device=logits.device)
        observed_weights = weights[likelihood_mask]
        if torch.any(observed_weights <= 0) or not torch.all(torch.isfinite(observed_weights)):
            raise ValueError("observed cell_weights must be finite and positive")
    per_cell = F.binary_cross_entropy_with_logits(logits, event_targets, reduction="none")
    per_row = (per_cell * mask * weights).sum(dim=1)
    if reduction == "none":
        return per_row
    if reduction == "sum":
        return per_row.sum()
    observed_rows = (mask.sum(dim=1) > 0).to(dtype=logits.dtype)
    return per_row.sum() / observed_rows.sum().clamp_min(1.0)


def hazard_logits_to_class_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Convert K hazard logits into K+1 ordered interval probabilities."""

    hazards = torch.sigmoid(logits).clamp(1e-7, 1.0 - 1e-7)
    survival_before = torch.cumprod(
        torch.cat([torch.ones_like(hazards[:, :1]), 1.0 - hazards[:, :-1]], dim=1),
        dim=1,
    )
    event_prob = survival_before * hazards
    tail_prob = torch.prod(1.0 - hazards, dim=1, keepdim=True)
    probabilities = torch.cat([event_prob, tail_prob], dim=1)
    return probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-12)
