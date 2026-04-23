"""
Standard ERM training loop for the baseline Image2Biomass model.

Loss: weighted SmoothL1 on raw target scales.
  Rationale: the competition metric's denominator (ss_tot) is constant w.r.t.
  model parameters, so minimising weighted squared error on raw scales is
  equivalent to maximising weighted global R². SmoothL1 instead of MSE for
  robustness to label noise on a small dataset (~285 train samples).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..shared.metrics import WEIGHT_VECTOR, weighted_global_r2, rmse_per_target

# Competition target weights (raw-scale loss)
_LOSS_WEIGHTS = torch.tensor(WEIGHT_VECTOR, dtype=torch.float32)


def _weighted_smooth_l1(pred: torch.Tensor, target: torch.Tensor,
                         weights: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """Per-target weighted SmoothL1, averaged over the batch."""
    loss_per_target = nn.functional.smooth_l1_loss(
        pred, target, beta=beta, reduction="none"
    )  # (B, 5)
    return (loss_per_target * weights.to(pred.device)).mean()


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = _weighted_smooth_l1(pred, y, _LOSS_WEIGHTS)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    all_pred, all_true = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = _weighted_smooth_l1(pred, y, _LOSS_WEIGHTS)
        total_loss += loss.item() * len(X)
        all_pred.append(pred.cpu().numpy())
        all_true.append(y.cpu().numpy())
    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_true)
    r2 = weighted_global_r2(y_true, y_pred)
    rmse = rmse_per_target(y_true, y_pred)
    return total_loss / len(loader.dataset), r2, rmse, y_pred, y_true


def train(
    model,
    train_ds,
    val_ds,
    epochs: int = 60,
    batch_size: int = 32,
    lr: float = 3e-4,
    weight_decay: float = 1e-3,
    seed: int = 42,
    device: str = "cpu",
    verbose: bool = True,
):
    """Standard ERM training loop (baseline). Returns history dict."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr / 100
    )

    history = {"train_loss": [], "val_loss": [], "val_r2": [], "val_rmse": []}

    for epoch in range(1, epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_r2, val_rmse, _, _ = eval_epoch(model, val_loader, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_r2"].append(val_r2)
        history["val_rmse"].append(val_rmse)

        if verbose and (epoch % 10 == 0 or epoch == 1):
            rmse_str = "  ".join(
                f"{t.split('_')[1]}:{v:.2f}" for t, v in val_rmse.items()
            )
            print(
                f"  ep {epoch:3d}  tr_loss={tr_loss:.4f}  "
                f"val_loss={val_loss:.4f}  val_R²={val_r2:.4f}  [{rmse_str}]"
            )

    return history
