"""IoMT-deployment safeguards (paper Section V Discussion).

This module provides reference implementations of the four deployment-time
safeguards declared in the paper's Discussion block:

1. Federated learning under DP-SGD with the manifest budget
   ``(epsilon, delta) = (4.0, 1e-5)`` per round, gradient clip ``C = 1``,
   and noise multiplier ``sigma = 1.1``, with Bonawitz-style secure
   aggregation as the cross-site protocol.
2. Mahalanobis-distance OOD detection at the per-class training-centroid
   99th percentile.
3. Hash-chained signed-append audit log primitive, anchored daily to a
   hospital HSM-signed root.
4. Silent-shadow logging gate: Phase-P writes to the audit log without
   influencing bedside outputs unless an explicit IRB / SaMD review has
   been recorded.

The federated path is intentionally framed as Future Work: nothing here
schedules a real cross-site round. The classes are designed to make the
deployment envelope auditable from configuration alone.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .constants import (
    DP_SGD_DEFAULTS,
    MAHALANOBIS_OOD_PERCENTILE,
)


# ---------------------------------------------------------------------------
# Federated DP-SGD envelope (Future Work).
# ---------------------------------------------------------------------------


@dataclass
class DPSGDConfig:
    """DP-SGD round budget transcribed from paper Section V Discussion."""

    epsilon_per_round: float = DP_SGD_DEFAULTS["epsilon_per_round"]
    delta: float = DP_SGD_DEFAULTS["delta"]
    noise_multiplier_sigma: float = DP_SGD_DEFAULTS["noise_multiplier_sigma"]
    l2_clip_C: float = DP_SGD_DEFAULTS["l2_clip_C"]

    def clip_gradient(self, gradient: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(gradient))
        scale = min(1.0, self.l2_clip_C / max(norm, 1e-12))
        return gradient * scale

    def noisy_gradient(self, clipped_gradient: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        noise = rng.normal(
            loc=0.0,
            scale=self.noise_multiplier_sigma * self.l2_clip_C,
            size=clipped_gradient.shape,
        )
        return clipped_gradient + noise

    def step(self, raw_gradient: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        return self.noisy_gradient(self.clip_gradient(raw_gradient), rng=rng)


@dataclass
class BonawitzSecureAggregator:
    """Reference stub for Bonawitz secure aggregation (Future Work).

    The real protocol uses pairwise additive masks and threshold secret
    sharing. This stub records the intended invariants so tests and
    deployment audits can verify that the federated path was not silently
    bypassed.
    """

    site_count: int = 4
    threshold: int = 3
    masking: str = "pairwise_additive"
    secret_sharing: str = "shamir_threshold"
    framing: str = "future_work"


# ---------------------------------------------------------------------------
# Mahalanobis OOD detection.
# ---------------------------------------------------------------------------


@dataclass
class MahalanobisOOD:
    """Per-class Mahalanobis-distance OOD detector at the 99th percentile.

    Fit one centroid and shared inverse-covariance per training class,
    threshold each class's distances at the configured percentile, and
    declare a sample OOD if its minimum class distance exceeds the
    matching threshold.
    """

    percentile: float = float(MAHALANOBIS_OOD_PERCENTILE)
    centroids_: dict[int, np.ndarray] = field(default_factory=dict)
    inv_cov_: np.ndarray | None = None
    thresholds_: dict[int, float] = field(default_factory=dict)
    ridge: float = 1e-6

    def fit(self, x: np.ndarray, y: np.ndarray) -> "MahalanobisOOD":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=int)
        cov = np.cov(x, rowvar=False)
        cov = cov + self.ridge * np.eye(cov.shape[0])
        self.inv_cov_ = np.linalg.pinv(cov)
        self.centroids_ = {}
        self.thresholds_ = {}
        for cls in np.unique(y):
            members = x[y == cls]
            if members.shape[0] == 0:
                continue
            mu = members.mean(axis=0)
            self.centroids_[int(cls)] = mu
            distances = self._distance_to_class(members, int(cls))
            self.thresholds_[int(cls)] = float(np.percentile(distances, self.percentile))
        return self

    def _distance_to_class(self, x: np.ndarray, cls: int) -> np.ndarray:
        if self.inv_cov_ is None or cls not in self.centroids_:
            raise RuntimeError("MahalanobisOOD is not fitted")
        delta = x - self.centroids_[cls]
        return np.einsum("ij,jk,ik->i", delta, self.inv_cov_, delta)

    def score(self, x: np.ndarray) -> dict[str, np.ndarray]:
        if self.inv_cov_ is None or not self.centroids_:
            raise RuntimeError("MahalanobisOOD is not fitted")
        x = np.asarray(x, dtype=float)
        classes = sorted(self.centroids_.keys())
        per_class = np.column_stack([self._distance_to_class(x, cls) for cls in classes])
        nearest_idx = np.argmin(per_class, axis=1)
        nearest_distance = per_class[np.arange(len(x)), nearest_idx]
        thresholds = np.asarray([self.thresholds_[classes[i]] for i in nearest_idx])
        return {
            "nearest_class": np.asarray([classes[i] for i in nearest_idx], dtype=int),
            "distance": nearest_distance,
            "threshold": thresholds,
            "is_ood": nearest_distance > thresholds,
        }


# ---------------------------------------------------------------------------
# Hash-chained signed-append audit log.
# ---------------------------------------------------------------------------


def _stable_serialize(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


@dataclass
class HashChainedAuditLog:
    """Append-only hash-chained audit log.

    Each entry stores a deterministic hash over the previous chain head
    concatenated with the entry payload. ``daily_anchor`` is intended to
    be the most recent HSM-signed root for the day; production deployments
    inject the live HSM signature here.
    """

    chain_id: str = "p-help-hcc"
    daily_anchor: str = "hospital_HSM_signed_root"
    head: str = ""
    entries: list[dict[str, Any]] = field(default_factory=list)

    def append(self, payload: dict[str, Any]) -> dict[str, Any]:
        timestamp = float(time.time())
        body = {"chain_id": self.chain_id, "ts": timestamp, "payload": payload}
        prev = self.head or self.daily_anchor
        digest = hashlib.sha256()
        digest.update(prev.encode("utf-8"))
        digest.update(_stable_serialize(body).encode("utf-8"))
        head = digest.hexdigest()
        record = {**body, "prev_hash": prev, "hash": head}
        self.entries.append(record)
        self.head = head
        return record

    def verify(self) -> bool:
        prev = self.daily_anchor
        for record in self.entries:
            digest = hashlib.sha256()
            digest.update(prev.encode("utf-8"))
            body = {k: record[k] for k in ("chain_id", "ts", "payload")}
            digest.update(_stable_serialize(body).encode("utf-8"))
            if digest.hexdigest() != record["hash"] or record["prev_hash"] != prev:
                return False
            prev = record["hash"]
        return True


# ---------------------------------------------------------------------------
# Silent-shadow gate.
# ---------------------------------------------------------------------------


@dataclass
class SilentShadowGate:
    """Phase-P silent-shadow logging vs.\\ live closed-loop gate.

    The paper's Discussion specifies that Phase-P operates as silent
    shadow logging until an explicit IRB and SaMD review allows live
    closed-loop influence. This gate centralises that policy.
    """

    irb_approval_id: str | None = None
    samd_class: str = "IMDRF_Class_II"
    live_loop_enabled: bool = False

    def can_influence_bedside(self) -> bool:
        return bool(self.live_loop_enabled and self.irb_approval_id)

    def envelope(self) -> dict[str, Any]:
        return {
            "irb_approval_id": self.irb_approval_id,
            "samd_class": self.samd_class,
            "live_loop_enabled": self.live_loop_enabled,
            "mode": "live_closed_loop" if self.can_influence_bedside() else "silent_shadow",
        }


__all__ = [
    "DPSGDConfig",
    "BonawitzSecureAggregator",
    "MahalanobisOOD",
    "HashChainedAuditLog",
    "SilentShadowGate",
]
