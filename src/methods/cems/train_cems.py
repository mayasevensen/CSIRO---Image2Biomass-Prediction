"""
CEMS-specific training loop (anchor-based manifold augmentation).

The standard ERM baseline (train, train_epoch, eval_epoch) lives in
shared/train.py. This module contains only the CEMS augmentation loop.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from shared.train import eval_epoch, _weighted_smooth_l1, _LOSS_WEIGHTS


# ---------------------------------------------------------------------------
# CEMS training loop (method 1, anchor-based)
# ---------------------------------------------------------------------------

def train_cems(
    model,
    X_train: np.ndarray,
    Y_train_raw: np.ndarray,
    val_ds,
    knn_indices: np.ndarray,
    scaler,
    args,
    neigh_size: int,
    epochs: int = 60,
    lr: float = 3e-4,
    weight_decay: float = 1e-3,
    seed: int = 42,
    device: str = "cpu",
    verbose: bool = True,
):
    """Anchor-based CEMS training (method 1 / batch-centred SVD).

    Ported from reference algorithm.py train_cems_batched.
    One anchor per iteration; each training sample serves as anchor exactly
    once per epoch (N_train iterations total per epoch).

    Args:
        model:        BiomassModel with forward_cems.
        X_train:      DINOv2 features  (N_train, 384), numpy float32.
        Y_train_raw:  Raw-scale targets (N_train, 5), numpy float32.
        val_ds:       PyTorch Dataset for validation (unchanged from ERM).
        knn_indices:  Precomputed kNN indices (N_train, neigh_size), int64.
        scaler:       Fitted MinMaxScaler for Y (transform / inverse_transform).
        args:         Namespace: sigma, cems_method, id, use_hessian.
        neigh_size:   Batch size for CEMS (= anchor + neighbours).
        epochs:       Number of training epochs.
        lr, weight_decay, seed, device, verbose: standard hyperparameters.

    Returns:
        history dict with per-epoch:
          train_loss, train_loss_real, train_loss_aug, val_loss, val_r2, val_rmse.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Precompute scaled Y (used inside CEMS joint space)
    Y_train_scaled = scaler.transform(Y_train_raw).astype(np.float32)

    # Pin tensors to device for fast indexing
    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    Y_raw_t = torch.tensor(Y_train_raw, dtype=torch.float32, device=device)
    Y_scaled_t = torch.tensor(Y_train_scaled, dtype=torch.float32, device=device)

    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr / 100
    )

    N_train = len(X_train)
    samples_idx = np.arange(N_train)

    history = {
        "train_loss": [],
        "train_loss_real": [],
        "train_loss_aug": [],
        "val_loss": [],
        "val_r2": [],
        "val_rmse": [],
    }

    mode_str = "CEMS-full" if args.use_hessian else "CEMS-L (linear)"
    if verbose:
        print(f"  [train_cems] mode={mode_str}  neigh_size={neigh_size}  "
              f"sigma={args.sigma}  epochs={epochs}")

    for epoch in range(1, epochs + 1):
        model.train()
        shuffle_idx = np.random.permutation(samples_idx)  # fresh order each epoch
        idx_to_del: list = []

        epoch_loss = epoch_loss_real = epoch_loss_aug = 0.0
        n_steps = 0

        for anchor in shuffle_idx:
            anchor = int(anchor)
            if anchor in idx_to_del:
                continue   # already used as anchor this epoch

            # Gather neigh_size - 1 neighbours (skip self at col 0, skip used)
            neigh_row = knn_indices[anchor]          # (neigh_size,), col 0 = self
            candidates = neigh_row[1:]               # exclude self
            available = candidates[~np.isin(candidates, idx_to_del)]
            n_neigh = min(neigh_size - 1, len(available))
            if n_neigh == 0:
                # Fall back: use whatever is in the original row
                available = candidates[:neigh_size - 1]
                n_neigh = len(available)

            idx_2 = available[:n_neigh]
            idx_all = np.concatenate([[anchor], idx_2])

            X_batch = X_t[idx_all]           # (batch, 384)
            Y_raw_batch = Y_raw_t[idx_all]   # (batch, 5)
            Y_sc_batch = Y_scaled_t[idx_all] # (batch, 5)

            optimizer.zero_grad()

            # Real path — normal forward, gradient flows through encoder
            pred_real = model(X_batch)
            loss_real = _weighted_smooth_l1(pred_real, Y_raw_batch, _LOSS_WEIGHTS)

            # CEMS augmentation path
            pred_aug, y_aug_scaled = model.forward_cems(args, X_batch, Y_sc_batch)
            # Unscale y_aug back to raw target space for a comparable loss
            y_aug_np = scaler.inverse_transform(
                y_aug_scaled.detach().cpu().numpy()
            ).astype(np.float32)
            y_aug_raw = torch.tensor(y_aug_np, dtype=torch.float32, device=device)
            loss_aug = _weighted_smooth_l1(pred_aug, y_aug_raw, _LOSS_WEIGHTS)

            loss = loss_real + loss_aug
            loss.backward()
            optimizer.step()

            b = len(idx_all)
            epoch_loss += loss.item() * b
            epoch_loss_real += loss_real.item() * b
            epoch_loss_aug += loss_aug.item() * b
            n_steps += b

            idx_to_del.append(anchor)   # mark anchor as used; neighbours stay available

        scheduler.step()

        tr_loss = epoch_loss / max(n_steps, 1)
        tr_real = epoch_loss_real / max(n_steps, 1)
        tr_aug = epoch_loss_aug / max(n_steps, 1)

        val_loss, val_r2, val_rmse, _, _ = eval_epoch(model, val_loader, device)

        history["train_loss"].append(tr_loss)
        history["train_loss_real"].append(tr_real)
        history["train_loss_aug"].append(tr_aug)
        history["val_loss"].append(val_loss)
        history["val_r2"].append(val_r2)
        history["val_rmse"].append(val_rmse)

        if verbose and (epoch % 10 == 0 or epoch == 1):
            rmse_str = "  ".join(
                f"{t.split('_')[1]}:{v:.2f}" for t, v in val_rmse.items()
            )
            print(
                f"  ep {epoch:3d}  tr={tr_loss:.4f}  "
                f"real={tr_real:.4f}  aug={tr_aug:.4f}  "
                f"val_loss={val_loss:.4f}  val_R²={val_r2:.4f}  [{rmse_str}]"
            )

    return history
