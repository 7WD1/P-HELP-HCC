"""PyTorch MLP classifier used as the DNN backbone."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .constants import CLASS_WEIGHTS, N_CLASSES
from .losses import FocalLoss
from .utils import seed_everything


class PHelpMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.2,
        n_classes: int = N_CLASSES,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for width in hidden_dims:
            layers.extend([nn.Linear(prev, width), nn.GELU(), nn.Dropout(dropout)])
            prev = width
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class MLPClassifierWrapper:
    input_dim: int
    hidden_dims: list[int]
    dropout: float
    n_classes: int = N_CLASSES
    state_dict: dict | None = None

    def _build(self) -> PHelpMLP:
        model = PHelpMLP(self.input_dim, self.hidden_dims, self.dropout, self.n_classes)
        if self.state_dict is not None:
            model.load_state_dict(self.state_dict)
        model.eval()
        return model

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        model = self._build()
        with torch.no_grad():
            logits = model(torch.tensor(x, dtype=torch.float32))
            return torch.softmax(logits, dim=1).cpu().numpy()


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
    seed: int = 42,
) -> MLPClassifierWrapper:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PHelpMLP(x_train.shape[1], hidden_dims, dropout).to(device)
    criterion = FocalLoss(gamma=gamma, class_weights=class_weights or CLASS_WEIGHTS)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    ds = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    x_val_t = torch.tensor(x_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.long, device=device)
    best_state = None
    best_loss = float("inf")
    waited = 0
    for _epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(x_val_t), y_val_t).cpu())
        if val_loss + 1e-6 < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            waited = 0
        else:
            waited += 1
            if waited >= patience:
                break
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return MLPClassifierWrapper(
        input_dim=x_train.shape[1],
        hidden_dims=list(hidden_dims),
        dropout=dropout,
        state_dict=best_state,
    )

