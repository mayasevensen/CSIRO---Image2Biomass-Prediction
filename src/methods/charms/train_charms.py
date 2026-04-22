"""
Training loop for CHARMS on the Image2Biomass baseline.

Implements the paper's combined loss (eq. 7–8):

    L = ℓ(f(x^I), y)                                      # image biomass task
      + λ_tab * ℓ(g(x^T), y)                              # tabular biomass task
      + λ_num * Σ_p MSE(T_p · φ(x^I), x_num_p)            # aux numeric
      + λ_cat * Σ_q CE(T_q · φ(x^I), x_cat_q)             # aux categorical

Key training details from the paper:
  * Cost matrix / transport matrix recomputed every cfg.cost_update_every
    epochs (paper uses 5).
  * φ (image encoder) is updated every step — only the cost matrix refresh
    is periodic.
  * Validation uses the image-only path (baseline-compatible forward).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from shared.metrics import WEIGHT_VECTOR, weighted_global_r2, rmse_per_target
from .charms import BiomassCharmsModel, CharmsConfig, CharmsDataset

_LOSS_WEIGHTS = torch.tensor(WEIGHT_VECTOR, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def weighted_smooth_l1(
    pred: torch.Tensor, target: torch.Tensor,
    weights: torch.Tensor, beta: float = 1.0,
) -> torch.Tensor:
    """Per-target weighted SmoothL1 — matches baseline (shared/train.py)."""
    per_target = F.smooth_l1_loss(pred, target, beta=beta, reduction="none")  # (B, 5)
    return (per_target * weights.to(pred.device)).mean()


def aux_numeric_loss(num_preds: List[torch.Tensor], x_num: torch.Tensor) -> torch.Tensor:
    """Σ_p MSE(pred_p, x_num_p). x_num is already z-scored in the dataset."""
    if not num_preds:
        return torch.tensor(0.0, device=x_num.device)
    losses = [F.mse_loss(p, x_num[:, i]) for i, p in enumerate(num_preds)]
    return torch.stack(losses).sum()


def aux_categorical_loss(cat_logits: List[torch.Tensor], x_cat: torch.Tensor) -> torch.Tensor:
    """Σ_q CE(logits_q, x_cat_q)."""
    if not cat_logits:
        return torch.tensor(0.0, device=x_cat.device)
    losses = [F.cross_entropy(logits, x_cat[:, j]) for j, logits in enumerate(cat_logits)]
    return torch.stack(losses).sum()


# ---------------------------------------------------------------------------
# Transport refresh
# ---------------------------------------------------------------------------

@torch.no_grad()
def _collect_refresh_batch(
    loader: DataLoader, model: BiomassCharmsModel, device: str, n_samples: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Concatenate batches from `loader` until we have at least n_samples."""
    xs, nums, cats = [], [], []
    collected = 0
    for batch in loader:
        xs.append(batch["X"].to(device))
        nums.append(batch["num"].to(device))
        cats.append(batch["cat"].to(device))
        collected += batch["X"].shape[0]
        if collected >= n_samples:
            break
    X = torch.cat(xs, dim=0)[:n_samples]
    N = torch.cat(nums, dim=0)[:n_samples]
    C = torch.cat(cats, dim=0)[:n_samples]
    return X, N, C


def refresh_transport_matrix(
    model: BiomassCharmsModel,
    loader: DataLoader,
    device: str,
) -> torch.Tensor:
    """Recompute T from a batch of training samples; updates model.transport in place."""
    X, num, cat = _collect_refresh_batch(
        loader, model, device,
        n_samples=model.cfg.cost_update_samples,
    )
    return model.refresh_transport(X, num, cat)


# ---------------------------------------------------------------------------
# Epoch loops
# ---------------------------------------------------------------------------

def train_epoch(
    model: BiomassCharmsModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> Dict[str, float]:
    model.train()
    totals = {"total": 0.0, "img": 0.0, "tab": 0.0, "num": 0.0, "cat": 0.0}
    n = 0
    for batch in loader:
        X = batch["X"].to(device)
        y = batch["y"].to(device)
        x_num = batch["num"].to(device)
        x_cat = batch["cat"].to(device)

        out = model.forward_train(X, x_num, x_cat)

        L_img = weighted_smooth_l1(out["y_img"], y, _LOSS_WEIGHTS)
        L_tab = weighted_smooth_l1(out["y_tab"], y, _LOSS_WEIGHTS)
        L_num = aux_numeric_loss(out["num_preds"], x_num)
        L_cat = aux_categorical_loss(out["cat_logits"], x_cat)

        cfg = model.cfg
        loss = (
            L_img
            + cfg.lambda_tab    * L_tab
            + cfg.lambda_i2t_num * L_num
            + cfg.lambda_i2t_cat * L_cat
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = X.shape[0]
        totals["total"] += loss.item() * bs
        totals["img"]   += L_img.item() * bs
        totals["tab"]   += L_tab.item() * bs
        totals["num"]   += L_num.item() * bs
        totals["cat"]   += L_cat.item() * bs
        n += bs
    return {k: v / max(n, 1) for k, v in totals.items()}


@torch.no_grad()
def eval_epoch(
    model: BiomassCharmsModel,
    loader: DataLoader,
    device: str,
) -> Tuple[float, float, Dict[str, float], np.ndarray, np.ndarray]:
    """
    Image-only evaluation — the CHARMS tabular branch and aux heads are
    discarded, so this matches baseline evaluation exactly.
    """
    model.eval()
    total_loss = 0.0
    all_pred, all_true = [], []
    for batch in loader:
        X = batch["X"].to(device)
        y = batch["y"].to(device)
        pred = model(X)  # baseline-compatible forward (Encoder → BiomassHead)
        loss = weighted_smooth_l1(pred, y, _LOSS_WEIGHTS)
        total_loss += loss.item() * X.shape[0]
        all_pred.append(pred.cpu().numpy())
        all_true.append(y.cpu().numpy())

    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_true)
    r2 = weighted_global_r2(y_true, y_pred)
    rmse = rmse_per_target(y_true, y_pred)
    return total_loss / len(loader.dataset), r2, rmse, y_pred, y_true


# ---------------------------------------------------------------------------
# Top-level trainer
# ---------------------------------------------------------------------------

def train(
    model: BiomassCharmsModel,
    train_ds: CharmsDataset,
    val_ds: CharmsDataset,
    epochs: int = 80,
    batch_size: int = 32,
    lr: float = 3e-4,
    weight_decay: float = 1e-3,
    seed: int = 42,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict[str, list]:
    """
    CHARMS training loop. Mirrors baseline `shared/train.train` (same optimizer,
    scheduler, batch size, seed handling) so the only difference vs baseline
    is the CHARMS loss/heads.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0)
    # separate unshuffled loader for refreshing the transport matrix
    refresh_loader = DataLoader(train_ds, batch_size=min(256, len(train_ds)),
                                shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr / 100
    )

    history: Dict[str, list] = {
        "train_total": [], "train_img": [], "train_tab": [],
        "train_num": [], "train_cat": [],
        "val_loss": [], "val_r2": [], "val_rmse": [],
        "transport_updates": [],
    }

    # Initial transport — before any gradient step
    refresh_transport_matrix(model, refresh_loader, device)
    history["transport_updates"].append(0)

    for epoch in range(1, epochs + 1):
        tr = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_r2, val_rmse, _, _ = eval_epoch(model, val_loader, device)
        scheduler.step()

        history["train_total"].append(tr["total"])
        history["train_img"].append(tr["img"])
        history["train_tab"].append(tr["tab"])
        history["train_num"].append(tr["num"])
        history["train_cat"].append(tr["cat"])
        history["val_loss"].append(val_loss)
        history["val_r2"].append(val_r2)
        history["val_rmse"].append(val_rmse)

        # Refresh cost/transport matrix every cost_update_every epochs
        if epoch % model.cfg.cost_update_every == 0 and epoch < epochs:
            refresh_transport_matrix(model, refresh_loader, device)
            history["transport_updates"].append(epoch)

        if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == epochs):
            rmse_str = "  ".join(
                f"{t.split('_')[1]}:{v:.2f}" for t, v in val_rmse.items()
            )
            print(
                f"  ep {epoch:3d}  "
                f"L_img={tr['img']:.4f}  L_tab={tr['tab']:.4f}  "
                f"L_num={tr['num']:.4f}  L_cat={tr['cat']:.4f}  "
                f"val_L={val_loss:.4f}  val_R²={val_r2:.4f}  [{rmse_str}]"
            )

    return history
