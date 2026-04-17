"""
Part 2 baseline: ERM training on cached DINOv2 features.

Steps:
  1. Load cached features + build train/val splits.
  2. Train BiomassModel (Encoder 384→128→32, Head 32→32→5) with weighted SmoothL1.
  3. Report final val metrics.
  4. Save weights to src/CEMS/checkpoints/baseline.pt.
  5. Estimate intrinsic dimension of the learned 32-d latent (TwoNN).
  6. Sanity checks: loss decreasing, val R² beats mean-prediction baseline,
     deterministic across two identical seeds.
"""

import sys
import numpy as np
import torch
from pathlib import Path

# Allow running as a script from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.CEMS.dataset import load_datasets
from src.CEMS.model import BiomassModel
from src.CEMS.metrics import (
    weighted_global_r2, rmse_per_target, TARGETS, _verify_metrics
)
from src.CEMS.train import train, eval_epoch
from src.CEMS.estimate_id import estimate_id

from torch.utils.data import DataLoader

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
CACHE_DIR = Path(__file__).resolve().parent / "cache"

SEED = 42
EPOCHS = 80
BATCH_SIZE = 32
LR = 3e-4
WEIGHT_DECAY = 1e-3
LATENT_DIM = 32
DROPOUT = 0.1


def mean_prediction_r2(val_ds):
    """R² when predicting per-target training mean — minimum acceptable bar."""
    all_y = val_ds.y.numpy()
    train_mean = all_y.mean(axis=0, keepdims=True)
    y_mean_pred = np.broadcast_to(train_mean, all_y.shape)
    return weighted_global_r2(all_y, y_mean_pred)


def main():
    # ---- verify metrics ----
    _verify_metrics()

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- data ----
    print("\nLoading datasets...")
    train_ds, val_ds = load_datasets()
    print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}")

    # ---- mean-prediction baseline ----
    mean_r2 = mean_prediction_r2(val_ds)
    print(f"  Mean-prediction val R² (floor): {mean_r2:.4f}")

    # ---- train ----
    print(f"\nTraining baseline (ERM, seed={SEED}, epochs={EPOCHS})...")
    model = BiomassModel(
        input_dim=384, latent_dim=LATENT_DIM, output_dim=5, dropout=DROPOUT
    ).to(device)

    history = train(
        model, train_ds, val_ds,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        seed=SEED,
        device=device,

        verbose=True,
    )

    # ---- final eval ----
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    _, final_r2, final_rmse, y_pred, y_true = eval_epoch(model, val_loader, device)

    print("\n" + "=" * 55)
    print("FINAL VAL METRICS")
    print("=" * 55)
    print(f"  Weighted global R²  : {final_r2:.4f}")
    print(f"  Mean-prediction R²  : {mean_r2:.4f}")
    for t in TARGETS:
        print(f"  RMSE {t:<18}: {final_rmse[t]:.4f}")

    # ---- sanity checks ----
    losses = history["train_loss"]
    first_quarter_mean = np.mean(losses[:len(losses) // 4])
    last_quarter_mean  = np.mean(losses[3 * len(losses) // 4:])
    loss_decreasing = last_quarter_mean < first_quarter_mean
    beats_mean = final_r2 > mean_r2

    print(f"\n  Loss decreasing (first-q vs last-q avg): "
          f"{first_quarter_mean:.4f} → {last_quarter_mean:.4f}  →  "
          f"{'PASS' if loss_decreasing else 'FAIL'}")
    print(f"  Beats mean-prediction R²: "
          f"{final_r2:.4f} > {mean_r2:.4f}  →  "
          f"{'PASS' if beats_mean else 'FAIL'}")

    # ---- determinism check ----
    print("\n  Determinism check (re-running with same seed)...")
    model2 = BiomassModel(
        input_dim=384, latent_dim=LATENT_DIM, output_dim=5, dropout=DROPOUT
    ).to(device)
    history2 = train(
        model2, train_ds, val_ds,
        epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
        weight_decay=WEIGHT_DECAY, seed=SEED, device=device,
        use_cems=False, verbose=False,
    )
    _, r2_run2, _, _, _ = eval_epoch(model2, val_loader, device)
    det_pass = abs(r2_run2 - final_r2) < 1e-4
    print(f"  Run 1 R²={final_r2:.6f}  Run 2 R²={r2_run2:.6f}  "
          f"→  {'PASS' if det_pass else 'FAIL (seed may not fully control MPS)'}")

    # ---- save checkpoint ----
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / "baseline.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "history": history,
        "final_r2": final_r2,
        "final_rmse": final_rmse,
        "hyperparams": {
            "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "lr": LR, "weight_decay": WEIGHT_DECAY,
            "latent_dim": LATENT_DIM, "dropout": DROPOUT, "seed": SEED,
        },
    }, ckpt_path)
    print(f"\n  Checkpoint saved: {ckpt_path}")

    # ---- latent ID estimation ----
    print("\n--- Latent ID estimation (32-d learned representation) ---")
    model.eval()
    all_feats = torch.tensor(
        np.load(CACHE_DIR / "features_dinov2.npy"), dtype=torch.float32
    ).to(device)
    with torch.no_grad():
        all_latents = model.encode(all_feats).cpu().numpy()
    print(f"  Latent shape: {all_latents.shape}")

    d_latent = estimate_id(all_latents, label="32-d learned latent")

    print("\n" + "=" * 55)
    print("LATENT INTRINSIC DIMENSION — CEMS FEASIBILITY")
    print("=" * 55)
    print(f"  d (32-d latent)  = {d_latent:.2f}")
    print(f"  d²               = {d_latent**2:.1f}")
    for bs in [16, 32, 64]:
        verdict = "OK (≥ d²)" if bs >= d_latent ** 2 else "TIGHT (< d²)"
        print(f"  bs={bs:3d}  → {verdict}")
    print("=" * 55)

    return {
        "final_r2": final_r2,
        "final_rmse": final_rmse,
        "d_latent": d_latent,
        "history": history,
    }


if __name__ == "__main__":
    main()
