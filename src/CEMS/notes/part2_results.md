# Part 2 Results — Baseline Pipeline & Latent ID Estimation

- N = 357 total (train=285, val=72, GroupKFold fold 0)
- Architecture: DINOv2 (frozen, 384-d) → Encoder 384→128→32 → Head 32→32→5
- Optimizer: AdamW lr=3e-4, wd=1e-3, cosine LR
- Loss: weighted SmoothL1 (raw scales, competition weights)
- Epochs: 80, batch_size=32, seed=42

## Loss Curve Summary

| Epoch | Train Loss | Val Loss | Val R² |
|-------|-----------|----------|--------|
| 1     | 6.534     | 6.933    | -1.063 |
| 10    | 3.295     | 3.437    | 0.194  |
| 20    | 2.362     | 2.370    | 0.582  |
| 30    | 2.140     | 2.193    | 0.635  |
| 40    | 1.997     | 2.111    | 0.652  |
| 50    | 1.912     | 2.010    | 0.684  |
| 60    | 1.884     | 2.024    | 0.679  |
| 70    | 1.859     | 2.034    | 0.675  |
| 80    | 1.827     | 2.025    | 0.677  |

Loss is decreasing throughout training (first-quarter avg 3.776 → last-quarter avg 1.865). Best val R² around epoch 50.

## Final Val Metrics (epoch 80)

- Weighted global R²: **0.6772**
- Mean-prediction R² (floor): 0.3508 → model clearly beats the floor ✓

| Target         | RMSE   |
|----------------|--------|
| Dry_Green_g    | 16.30  |
| Dry_Dead_g     | 10.92  |
| Dry_Clover_g   | 13.53  |
| GDM_g          | 13.80  |
| Dry_Total_g    | 14.90  |

## Sanity Checks

- Loss decreasing: PASS
- Beats mean-prediction baseline: PASS (0.677 > 0.351)
- Determinism: small run-to-run difference (0.6772 vs 0.6761) — MPS (Apple Silicon) does not guarantee bit-exact reproducibility even with fixed seed; difference <0.2%, acceptable

## Latent Intrinsic Dimension — CEMS Feasibility (decisive estimate)

- Representation: learned 32-d encoder output, all 357 training images
- Estimator: TwoNN (skdim)

| | |
|---|---|
| d (32-d latent) | **4.71** |
| d²              | **22.2** |

| Batch size | d² = 22.2 | Feasible? |
|-----------|-----------|-----------|
| 16        | < d²      | Tight     |
| 32        | ≥ d²      | OK ✓      |
| 64        | ≥ d²      | OK ✓      |

**Verdict: CEMS is feasible with batch size ≥ 32.** The bottleneck reduces the effective intrinsic dimension from ~12.5 (raw DINOv2) to ~4.7, making the d² neighbourhood requirement tractable at standard batch sizes.

## Deviations from Plan

- None significant. Architecture, loss, and split are exactly as specified.
