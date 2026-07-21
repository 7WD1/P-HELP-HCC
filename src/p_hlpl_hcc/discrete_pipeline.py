"""Executable censoring-aware discrete-time survival pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from .data import validate_and_prepare_dataframe
from .metrics import classification_metrics
from .preprocessing import PHlplPreprocessor
from .society import SocietyTransformer
from .survival import (
    CensoringKaplanMeier,
    discrete_time_negative_log_likelihood,
    fit_censoring_kaplan_meier,
    hazard_logits_to_class_probabilities,
    ipcw_likelihood_cell_weights,
    make_discrete_time_targets,
    risk_event_censor_counts,
)
from .utils import seed_everything


class DiscreteTimeNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float, n_intervals: int):
        super().__init__()
        layers: list[nn.Module] = []
        width = input_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout)])
            width = hidden
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.hazard_head = nn.Linear(width, n_intervals)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hazard_head(self.backbone(x))


@dataclass
class DiscreteTimeSurvivalPipeline:
    """MLP discrete-hazard model that never hard-labels immature censoring."""

    config: dict[str, Any]
    seed: int = 42
    preprocessor: PHlplPreprocessor | None = None
    society: SocietyTransformer | None = None
    model_state_: dict[str, torch.Tensor] | None = None
    input_dim_: int = 0
    cutpoints_: list[float] = field(default_factory=list)
    best_validation_nll_: float = float("nan")
    mechanism_trace_: dict[str, Any] = field(default_factory=dict)
    censoring_km_: CensoringKaplanMeier | None = None

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        data_cfg = self.config["data"]
        prepared = validate_and_prepare_dataframe(
            df,
            time_col=data_cfg["target_time_col"],
            event_col=data_cfg["event_col"],
            label_col=data_cfg["label_col"],
            require_unambiguous_hard_labels=False,
        )
        bins = list(
            map(
                float,
                data_cfg.get(
                    "survival_bins_months", [0, 6, 12, 24, 36, 48, 60, 72]
                ),
            )
        )
        horizon = float(max(bins))
        time_col = data_cfg["target_time_col"]
        event_col = data_cfg["event_col"]
        label_col = data_cfg["label_col"]
        immature_censored = (prepared[event_col] == 0) & (
            prepared[time_col] < horizon
        )
        # A caller-supplied label does not make an immature censored endpoint
        # definitive. Such rows remain in the likelihood but never enter hard
        # eight-class metrics.
        prepared[label_col] = prepared[label_col].astype("Int64")
        prepared.loc[immature_censored, label_col] = pd.NA
        return prepared

    def _build_model(self) -> DiscreteTimeNetwork:
        if not self.cutpoints_ or self.input_dim_ <= 0:
            raise RuntimeError("Discrete-time pipeline is not initialized")
        mlp = self.config.get("phase_c", {}).get("mlp", {})
        model = DiscreteTimeNetwork(
            self.input_dim_,
            list(mlp.get("hidden_dims", [256, 128, 64])),
            float(mlp.get("dropout", 0.2)),
            len(self.cutpoints_),
        )
        if self.model_state_ is not None:
            model.load_state_dict(self.model_state_)
        return model

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> "DiscreteTimeSurvivalPipeline":
        seed_everything(self.seed)
        train = self._prepare(train_df)
        val = self._prepare(val_df)
        data_cfg = self.config["data"]
        time_col, event_col = data_cfg["target_time_col"], data_cfg["event_col"]
        bins = list(map(float, data_cfg.get("survival_bins_months", [0, 6, 12, 24, 36, 48, 60, 72])))
        self.cutpoints_ = bins[1:] if bins and bins[0] == 0 else bins

        self.preprocessor = PHlplPreprocessor(curated_dim=int(data_cfg.get("curated_dim", 67)))
        x_train = self.preprocessor.fit_transform(train)
        x_val = self.preprocessor.transform(val)
        self.society = SocietyTransformer(process_noise_std=0.0)
        state_train = self.society.fit_transform(x_train)
        state_val = self.society.transform(x_val)
        self.input_dim_ = int(state_train.shape[1])

        train_targets = make_discrete_time_targets(
            train[time_col].to_numpy(), train[event_col].to_numpy(), self.cutpoints_
        )
        val_targets = make_discrete_time_targets(
            val[time_col].to_numpy(), val[event_col].to_numpy(), self.cutpoints_
        )
        self.censoring_km_ = fit_censoring_kaplan_meier(
            train[time_col].to_numpy(), train[event_col].to_numpy()
        )
        train_ipcw = ipcw_likelihood_cell_weights(
            train[time_col], train[event_col], train_targets, self.censoring_km_
        )
        val_ipcw = ipcw_likelihood_cell_weights(
            val[time_col], val[event_col], val_targets, self.censoring_km_
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self._build_model().to(device)
        mlp_cfg = self.config.get("phase_c", {}).get("mlp", {})
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(mlp_cfg.get("learning_rate", 1e-3)),
            weight_decay=float(mlp_cfg.get("weight_decay", 0.0)),
        )
        batch_size = max(1, int(mlp_cfg.get("batch_size", 32)))
        epochs = max(1, int(mlp_cfg.get("epochs", 100)))
        patience = max(1, int(mlp_cfg.get("patience", 15)))

        x_t = torch.tensor(state_train, dtype=torch.float32)
        y_t = torch.tensor(train_targets.event_targets, dtype=torch.float32)
        m_t = torch.tensor(train_targets.likelihood_mask, dtype=torch.bool)
        w_t = torch.tensor(train_ipcw, dtype=torch.float32)
        x_val_t = torch.tensor(state_val, dtype=torch.float32, device=device)
        y_val_t = torch.tensor(val_targets.event_targets, dtype=torch.float32, device=device)
        m_val_t = torch.tensor(val_targets.likelihood_mask, dtype=torch.bool, device=device)
        w_val_t = torch.tensor(val_ipcw, dtype=torch.float32, device=device)
        generator = torch.Generator().manual_seed(self.seed)
        best_loss, waited, best_state = float("inf"), 0, None
        for _epoch in range(epochs):
            model.train()
            for indices in torch.randperm(len(x_t), generator=generator).split(batch_size):
                xb, yb, mb, wb = (
                    x_t[indices].to(device),
                    y_t[indices].to(device),
                    m_t[indices].to(device),
                    w_t[indices].to(device),
                )
                optimizer.zero_grad(set_to_none=True)
                loss = discrete_time_negative_log_likelihood(model(xb), yb, mb, cell_weights=wb)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val_loss = float(
                    discrete_time_negative_log_likelihood(
                        model(x_val_t), y_val_t, m_val_t, cell_weights=w_val_t
                    ).cpu()
                )
            if val_loss + 1e-7 < best_loss:
                best_loss = val_loss
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                waited = 0
            else:
                waited += 1
                if waited >= patience:
                    break
        if best_state is None:
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        self.model_state_ = best_state
        self.best_validation_nll_ = best_loss
        self.mechanism_trace_ = {
            "endpoint_objective": "censoring_aware_discrete_time",
            "hard_labels_for_immature_censoring": False,
            "interval_cutpoints_months": self.cutpoints_,
            "training_rows": len(train),
            "training_likelihood_cells": int(train_targets.likelihood_mask.sum()),
            "validation_likelihood_cells": int(val_targets.likelihood_mask.sum()),
            "ipcw_censoring_estimator": "training_fold_reverse_kaplan_meier",
            "ipcw_training_weight_min": float(train_ipcw[train_targets.likelihood_mask].min()),
            "ipcw_training_weight_max": float(train_ipcw.max()),
        }
        return self

    def _state(self, df: pd.DataFrame, *, already_prepared: bool = False) -> np.ndarray:
        if self.preprocessor is None or self.society is None:
            raise RuntimeError("Discrete-time pipeline is not fitted")
        prepared = df if already_prepared else self._prepare(df)
        return self.society.transform(self.preprocessor.transform(prepared))

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        state = self._state(df)
        model = self._build_model().eval()
        with torch.no_grad():
            return hazard_logits_to_class_probabilities(
                model(torch.tensor(state, dtype=torch.float32))
            ).cpu().numpy()

    def predict_hazards(self, df: pd.DataFrame) -> np.ndarray:
        """Return seven interval hazards; the eighth class is the 72-month tail."""

        state = self._state(df)
        model = self._build_model().eval()
        with torch.no_grad():
            return torch.sigmoid(model(torch.tensor(state, dtype=torch.float32))).cpu().numpy()

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.argmax(self.predict_proba(df), axis=1)

    def evaluate(self, df: pd.DataFrame) -> dict[str, Any]:
        prepared = self._prepare(df)
        data_cfg = self.config["data"]
        time_col, event_col, label_col = (
            data_cfg["target_time_col"],
            data_cfg["event_col"],
            data_cfg["label_col"],
        )
        targets = make_discrete_time_targets(
            prepared[time_col].to_numpy(), prepared[event_col].to_numpy(), self.cutpoints_
        )
        state = self._state(prepared, already_prepared=True)
        model = self._build_model().eval()
        if self.censoring_km_ is None:
            raise RuntimeError("Training-fold censoring KM is unavailable")
        evaluation_ipcw = ipcw_likelihood_cell_weights(
            prepared[time_col], prepared[event_col], targets, self.censoring_km_
        )
        with torch.no_grad():
            logits = model(torch.tensor(state, dtype=torch.float32))
            nll = float(
                discrete_time_negative_log_likelihood(
                    logits,
                    torch.tensor(targets.event_targets),
                    torch.tensor(targets.likelihood_mask),
                    cell_weights=torch.tensor(evaluation_ipcw, dtype=torch.float32),
                )
            )
            proba = hazard_logits_to_class_probabilities(logits).cpu().numpy()
        result: dict[str, Any] = {
            "discrete_time_nll": nll,
            "n_rows": int(len(prepared)),
            "n_immature_censored": int(
                ((prepared[event_col] == 0) & (prepared[time_col] < self.cutpoints_[-1])).sum()
            ),
            "ipcw_weight_min": float(evaluation_ipcw[targets.likelihood_mask].min()),
            "ipcw_weight_max": float(evaluation_ipcw.max()),
            "risk_event_censor_counts": risk_event_censor_counts(
                prepared[time_col], prepared[event_col], self.cutpoints_
            ),
        }
        auditable = prepared[label_col].notna().to_numpy()
        result["n_hard_label_auditable"] = int(auditable.sum())
        result["n_hard_label_excluded"] = int((~auditable).sum())
        if auditable.any():
            result.update(
                classification_metrics(
                    prepared.loc[auditable, label_col].to_numpy(dtype=int),
                    proba[auditable],
                    times=prepared.loc[auditable, time_col].to_numpy(dtype=float),
                    events=prepared.loc[auditable, event_col].to_numpy(dtype=int),
                )
            )
        return result
