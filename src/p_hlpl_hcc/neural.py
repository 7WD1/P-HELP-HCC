"""PyTorch MLP classifier used as the DNN backbone."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .constants import CLASS_WEIGHTS, N_CLASSES
from .losses import FocalLoss
from .parallel import phase_p_residual_score
from .utils import seed_everything


class PHlplMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.2,
        n_classes: int = N_CLASSES,
        scenario_classes: int = 0,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for width in hidden_dims:
            layers.extend([nn.Linear(prev, width), nn.GELU(), nn.Dropout(dropout)])
            prev = width
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.survival_head = nn.Linear(prev, n_classes)
        self.scenario_head = nn.Linear(prev, scenario_classes) if scenario_classes > 0 else None

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.survival_head(self.forward_features(x))

    def forward_with_scenario(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.forward_features(x)
        if self.scenario_head is None:
            raise RuntimeError("Scenario auxiliary head is disabled")
        return self.survival_head(features), self.scenario_head(features)


@dataclass
class MLPClassifierWrapper:
    input_dim: int
    hidden_dims: list[int]
    dropout: float
    n_classes: int = N_CLASSES
    scenario_classes: int = 0
    state_dict: dict | None = None
    training_metadata: dict[str, float | bool] = field(default_factory=dict)

    def _build(self) -> PHlplMLP:
        model = PHlplMLP(
            self.input_dim,
            self.hidden_dims,
            self.dropout,
            self.n_classes,
            self.scenario_classes,
        )
        if self.state_dict is not None:
            model.load_state_dict(self.state_dict)
        model.eval()
        return model

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        model = self._build()
        with torch.no_grad():
            logits = model(torch.tensor(x, dtype=torch.float32))
            return torch.softmax(logits, dim=1).cpu().numpy()

    def predict_scenario_proba(self, x: np.ndarray) -> np.ndarray:
        if self.scenario_classes <= 0:
            raise RuntimeError("Scenario auxiliary head is disabled")
        model = self._build()
        with torch.no_grad():
            _survival_logits, scenario_logits = model.forward_with_scenario(
                torch.tensor(x, dtype=torch.float32)
            )
            return torch.softmax(scenario_logits, dim=1).cpu().numpy()


def train_mlp_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    hidden_dims: list[int],
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    patience: int,
    gamma: float,
    class_weights: list[float] | None = None,
    sample_weights: np.ndarray | None = None,
    scenario_train: np.ndarray | None = None,
    scenario_val: np.ndarray | None = None,
    scenario_loss_weight: float = 0.0,
    n_scenario_classes: int = 0,
    phase_p_model_selection_enabled: bool = False,
    phase_p_residual_weight: float = 0.0,
    val_events: np.ndarray | None = None,
    phase_p_calibration_mix: float = 0.5,
    phase_e_lambda_cal: float = 0.0,
    phase_e_lambda_exp: float = 0.0,
    phase_e_lambda_clin: float = 0.0,
    phase_e_kappa: float = 5.0,
    cox_direction: np.ndarray | None = None,
    clinical_monotonic_indices: list[int] | None = None,
    clinical_monotonic_signs: list[float] | None = None,
    seed: int = 42,
) -> MLPClassifierWrapper:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scenario_enabled = bool(
        scenario_loss_weight > 0
        and n_scenario_classes > 1
        and scenario_train is not None
        and scenario_val is not None
    )
    scenario_classes = int(n_scenario_classes) if scenario_enabled else 0
    model = PHlplMLP(
        x_train.shape[1],
        hidden_dims,
        dropout,
        scenario_classes=scenario_classes,
    ).to(device)
    criterion = FocalLoss(
        gamma=gamma,
        class_weights=class_weights or CLASS_WEIGHTS,
        reduction="none",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    if sample_weights is None:
        sample_weights = np.ones(len(x_train), dtype=float)
    sample_weights = np.asarray(sample_weights, dtype=float)
    if sample_weights.shape != (len(x_train),):
        raise ValueError("sample_weights must have one value per training row")
    cox_anchor_t: torch.Tensor | None = None
    if phase_e_lambda_exp > 0:
        if cox_direction is None or np.asarray(cox_direction).shape != (x_train.shape[1],):
            raise ValueError("A state-aligned Cox direction is required when L_exp is enabled")
        anchor = np.asarray(cox_direction, dtype=float)
        anchor = anchor / max(float(np.linalg.norm(anchor)), 1e-12)
        cox_anchor_t = torch.tensor(anchor, dtype=torch.float32, device=device).reshape(1, -1)
    monotonic_indices = list(clinical_monotonic_indices or [])
    if phase_e_lambda_clin > 0 and not monotonic_indices:
        raise ValueError("At least one clinical monotonic index is required when L_clin is enabled")
    if phase_e_lambda_clin > 0 and any(
        index < 0 or index >= x_train.shape[1] for index in monotonic_indices
    ):
        raise ValueError("clinical monotonic index is outside the model input")
    monotonic_signs = list(clinical_monotonic_signs or [1.0] * len(monotonic_indices))
    if len(monotonic_signs) != len(monotonic_indices):
        raise ValueError("clinical_monotonic_signs must align with indices")
    monotonic_signs_t = torch.tensor(monotonic_signs, dtype=torch.float32, device=device).reshape(1, -1)
    if scenario_enabled:
        scenario_train = np.asarray(scenario_train, dtype=int)
        scenario_val = np.asarray(scenario_val, dtype=int)
        if scenario_train.shape != (len(x_train),) or scenario_val.shape != (len(x_val),):
            raise ValueError("scenario targets must align with train/validation rows")
    else:
        scenario_train = np.full(len(x_train), -1, dtype=int)
        scenario_val = np.full(len(x_val), -1, dtype=int)
    ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(sample_weights, dtype=torch.float32),
        torch.tensor(scenario_train, dtype=torch.long),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    x_val_t = torch.tensor(x_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.long, device=device)
    scenario_val_t = torch.tensor(scenario_val, dtype=torch.long, device=device)
    best_state = None
    best_loss = float("inf")
    waited = 0
    phase_e_epoch_losses = {"pred_focal": 0.0, "cal": 0.0, "exp": 0.0, "clin": 0.0}
    for _epoch in range(epochs):
        model.train()
        phase_e_running = {"pred_focal": [], "cal": [], "exp": [], "clin": []}
        for xb, yb, wb, sb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)
            sb = sb.to(device)
            attribution_enabled = phase_e_lambda_exp > 0 or phase_e_lambda_clin > 0
            xb.requires_grad_(attribution_enabled)
            optimizer.zero_grad(set_to_none=True)
            if scenario_enabled:
                survival_logits, scenario_logits = model.forward_with_scenario(xb)
            else:
                survival_logits = model(xb)
                scenario_logits = None
            survival_loss = criterion(survival_logits, yb)
            pred_focal_loss = (survival_loss * wb).sum() / wb.sum().clamp_min(1e-12)
            loss = pred_focal_loss
            phase_e_running["pred_focal"].append(float(pred_focal_loss.detach().cpu()))
            probabilities = torch.softmax(survival_logits, dim=1)
            one_hot = nn.functional.one_hot(yb, num_classes=probabilities.shape[1]).to(probabilities.dtype)
            brier = torch.square(probabilities - one_hot).sum(dim=1)
            cal_loss = (brier * wb).sum() / wb.sum().clamp_min(1e-12)
            loss = loss + float(phase_e_lambda_cal) * cal_loss
            phase_e_running["cal"].append(float(cal_loss.detach().cpu()))
            exp_loss = torch.zeros((), dtype=loss.dtype, device=device)
            clin_loss = torch.zeros((), dtype=loss.dtype, device=device)
            if attribution_enabled:
                class_order = torch.arange(probabilities.shape[1], dtype=probabilities.dtype, device=device)
                short_survival_risk = -(probabilities * class_order.reshape(1, -1)).sum(dim=1)
                input_gradient = torch.autograd.grad(
                    short_survival_risk.sum(), xb, create_graph=True, retain_graph=True
                )[0]
                if phase_e_lambda_exp > 0 and cox_anchor_t is not None:
                    exp_loss = torch.abs(
                        torch.tanh(float(phase_e_kappa) * input_gradient)
                        - torch.tanh(float(phase_e_kappa) * cox_anchor_t)
                    ).mean()
                    loss = loss + float(phase_e_lambda_exp) * exp_loss
                if phase_e_lambda_clin > 0:
                    selected_gradient = input_gradient[:, monotonic_indices]
                    clin_loss = torch.relu(-selected_gradient * monotonic_signs_t).mean()
                    loss = loss + float(phase_e_lambda_clin) * clin_loss
            phase_e_running["exp"].append(float(exp_loss.detach().cpu()))
            phase_e_running["clin"].append(float(clin_loss.detach().cpu()))
            if scenario_logits is not None:
                scenario_loss = nn.functional.cross_entropy(scenario_logits, sb, reduction="none")
                loss = loss + float(scenario_loss_weight) * (
                    (scenario_loss * wb).sum() / wb.sum().clamp_min(1e-12)
                )
            loss.backward()
            optimizer.step()
        scheduler.step()
        phase_e_epoch_losses = {
            key: float(np.mean(values)) if values else 0.0
            for key, values in phase_e_running.items()
        }
        model.eval()
        with torch.no_grad():
            if scenario_enabled:
                val_logits, val_scenario_logits = model.forward_with_scenario(x_val_t)
            else:
                val_logits = model(x_val_t)
                val_scenario_logits = None
            val_loss_tensor = criterion(val_logits, y_val_t).mean()
            if val_scenario_logits is not None:
                val_loss_tensor = val_loss_tensor + float(scenario_loss_weight) * nn.functional.cross_entropy(
                    val_scenario_logits, scenario_val_t
                )
            val_loss = float(val_loss_tensor.cpu())
            val_proba = torch.softmax(val_logits, dim=1).cpu().numpy()
        residual = 0.0
        if phase_p_model_selection_enabled and val_events is not None:
            residual = phase_p_residual_score(
                val_proba,
                y_val,
                np.asarray(val_events, dtype=int),
                classification_calibration_mix=phase_p_calibration_mix,
            )
        selection_loss = val_loss + float(phase_p_residual_weight) * residual
        if selection_loss + 1e-6 < best_loss:
            best_loss = selection_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_validation_loss = val_loss
            best_residual = residual
            waited = 0
        else:
            waited += 1
            if waited >= patience:
                break
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_validation_loss = float("nan")
        best_residual = float("nan")
    return MLPClassifierWrapper(
        input_dim=x_train.shape[1],
        hidden_dims=list(hidden_dims),
        dropout=dropout,
        scenario_classes=scenario_classes,
        state_dict=best_state,
        training_metadata={
            "scenario_auxiliary_enabled": scenario_enabled,
            "scenario_loss_weight": float(scenario_loss_weight) if scenario_enabled else 0.0,
            "phase_p_model_selection_enabled": bool(phase_p_model_selection_enabled),
            "phase_p_residual_weight": float(phase_p_residual_weight)
            if phase_p_model_selection_enabled
            else 0.0,
            "best_selection_loss": float(best_loss),
            "best_validation_loss": float(best_validation_loss),
            "best_phase_p_residual": float(best_residual),
            "phase_e_differentiable_training": bool(
                phase_e_lambda_cal > 0 or phase_e_lambda_exp > 0 or phase_e_lambda_clin > 0
            ),
            "phase_e_lambda_cal": float(phase_e_lambda_cal),
            "phase_e_lambda_exp": float(phase_e_lambda_exp),
            "phase_e_lambda_clin": float(phase_e_lambda_clin),
            "phase_e_last_loss_pred_focal": phase_e_epoch_losses["pred_focal"],
            "phase_e_last_loss_cal": phase_e_epoch_losses["cal"],
            "phase_e_last_loss_exp": phase_e_epoch_losses["exp"],
            "phase_e_last_loss_clin": phase_e_epoch_losses["clin"],
            "cox_direction_connected": cox_anchor_t is not None,
            "clinical_monotonic_indices": monotonic_indices,
        },
    )
