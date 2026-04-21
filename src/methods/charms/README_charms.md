# CSIRO Image2Biomass — CHARMS Pipeline

**Competition:** [CSIRO Image2Biomass Prediction (Kaggle)](https://www.kaggle.com/competitions/csiro-biomass)  
**Task:** Predict 5 pasture biomass fractions from top-view RGB images  
**Method:** CHARMS — Cross-modal auxiliary supervision with metadata at train time, image-only at test time  
**Metric:** Weighted global R² (mean R² across 5 targets)

---

## Project Structure

```
.
├── README.md                          ← This file
│
├── notebooks/
│   ├── pipeline_CHARMS.ipynb          ← Original baseline notebook (no plots)
│   └── pipeline_CHARMS_v2_plots.ipynb ← Full CHARMS pipeline with diagnostic plots 
│
├── paper/
│   └── PAPER_OUTLINE.md               ← NLDL-aligned paper structure with writing guidance
│
└── outputs/
    ├── submission_charms_v2.csv        ← Final Kaggle submission file
    └── plots/
        ├── plot_01_target_distributions.png
        ├── plot_02_correlations.png
        ├── plot_03_dataset_structure.png
        ├── plot_04_training_curves.png
        ├── plot_05_oof_scatter.png
        ├── plot_06_residuals.png
        ├── plot_07_fold_agreement.png
        └── plot_08_train_test_distribution.png
```

---

## What is CHARMS?

**CHARMS** (Cross-modal Heuristic Auxiliary Representation via Metadata Supervision) solves a specific challenge: during training we have rich sensor metadata (NDVI, canopy height, geographic state, season) paired with every image. At test/deployment time, we have images only.

**The key idea:** train the CNN with two simultaneous objectives:
1. **Primary:** Predict the 5 biomass targets (main task)
2. **Auxiliary:** Reconstruct the metadata values from visual features alone

The auxiliary supervision forces the backbone to develop feature maps that encode physiologically meaningful information — vegetation greenness, canopy density, spectral properties — rather than potentially spurious shortcuts. The auxiliary heads are discarded at inference; only the improved backbone representations remain.

```
TRAINING                              INFERENCE
─────────────────────────────────     ────────────────────────────
Image → Backbone → visual features    Image → Backbone → visual features
                       │                                     │
             ┌─────────┤                          metadata = 0
             │         │                                     │
        Metadata  visual features                  Biomass head → 5 predictions
        embedding      │                          (aux heads discarded)
             │         │
             └──concat─┤
                        │
               ┌────────┴──────────┐
               ▼                   ▼
         Biomass heads         Aux heads
         (5 targets)    (NDVI, height, state, month)
```

---

## Pipeline Overview

### Data
- **1,162** top-view quadrat images (70×30 cm), 19 Australian sites, 2014–2017
- **5 targets:** Dry_Green_g, Dry_Dead_g, Dry_Clover_g, GDM_g, Dry_Total_g
- **Metadata:** Pre-grazing NDVI, canopy height, state (4 classes), month (1–12)

### Model
| Component | Choice | Why |
|---|---|---|
| Backbone | EfficientNet-V2-S | Best accuracy/parameter ratio for texture-heavy images |
| Image size | 384 × 384 | Native EfficientNet-V2-S size; captures fine grass texture |
| Metadata fusion | Feature-level concat + MLP | Metadata directly informs biomass regression |
| Target transform | log1p / expm1 | Biomass is right-skewed; normalises gradients |
| Loss | Uncertainty-weighted SmoothL1 | Adapts to per-target noise levels automatically |
| Auxiliary weight λ | 0.25 | Balances CHARMS supervision without dominating |

### Training
- **5-fold GroupKFold CV** (groups = `sample_id` to prevent leakage between field visits)
- **AdamW** optimiser, lr=2e-4, weight decay=1e-4
- **CosineAnnealingWarmRestarts** scheduler (T₀=20 epochs)
- **Augmentation:** random H+V flip, 90° rotation, ColorJitter, RandomGrayscale
- **Ensemble:** 5 fold models averaged in log-space

### Inference
- **TTA:** 4 flips (original + H-flip + V-flip + both) averaged
- Metadata embedding set to zero (simulates test-time deployment)
- expm1() applied to convert log-space predictions back to grams

---

## How to Run

### 1. Environment setup
```bash
pip install torch torchvision efficientnet_pytorch scikit-learn \
            pandas numpy matplotlib seaborn pillow
```

### 2. Data setup
```
data/
├── tabular/
│   ├── train/train.csv
│   └── test/test.csv
└── image/
    └── [image files as referenced in CSV]
```

### 3. Run the pipeline
Open `notebooks/pipeline_CHARMS_v2_plots.ipynb` in Jupyter and run all cells.

Update the path variables at the top of cell 2:
```python
TRAIN_CSV    = 'path/to/train.csv'
TEST_CSV     = 'path/to/test.csv'
IMAGE_FOLDER = 'path/to/images/'
```

### 4. Expected runtime
- Per fold: ~25–45 min on a T4 GPU (Kaggle free tier)
- Full 5-fold run: ~2–4 hours
- Inference with TTA: ~5 min

### 5. Output
- `submission_charms_v2.csv` — ready to submit to Kaggle
- `plot_0X_*.png` — diagnostic plots saved to working directory
- `best_fold{0-4}.pt` — best model checkpoint per fold

---

## Diagnostic Plots Guide

| Plot | File | What to check |
|---|---|---|
| 1. Target distributions | `plot_01_target_distributions.png` | Right-skew (expected); log1p should normalise |
| 2. Correlations | `plot_02_correlations.png` | NDVI–biomass R² ≈ 0.3–0.6 validates aux task |
| 3. Dataset structure | `plot_03_dataset_structure.png` | Seasonal + geographic imbalance |
| 4. Training curves | `plot_04_training_curves.png` | Train≈valid loss; R² plateauing ≥ 0.55 |
| 5. OOF scatter | `plot_05_oof_scatter.png` | Points near y=x diagonal; green=good |
| 6. Residuals | `plot_06_residuals.png` | Centred near 0; symmetric distribution |
| 7. Fold agreement | `plot_07_fold_agreement.png` | Low std (<0.10) = stable ensemble |
| 8. Train/test distribution | `plot_08_train_test_distribution.png` | Substantial overlap (no major shift) |

---

## Performance Benchmarks

| OOF R² | Interpretation |
|---|---|
| ≥ 0.70 | Excellent — top-quartile Kaggle solution |
| 0.55–0.70 | Good — competitive with leaderboard |
| 0.40–0.55 | Acceptable baseline |
| < 0.40 | Check data pipeline (especially log-transform inversion) |

Per-target expectations:
- **Dry_Total_g**: Easiest — should reach R² ≥ 0.65
- **Dry_Green_g**: Moderate — expect R² 0.55–0.70
- **Dry_Clover_g**: Hardest (sparse presence) — R² ≥ 0.40 is success

---

## Reproducibility

| Item | Value |
|---|---|
| Random seed | 42 |
| CV strategy | GroupKFold(k=5), groups=`sample_id` |
| Target transform | `log1p(clip(x, 0))` → `expm1(clip(x, 0))` |
| Feature normalisation | StandardScaler on NDVI and height |
| State encoding | Alphabetical label encoding |
| Month encoding | 0-indexed (Jan=0, Dec=11) |

---

## Paper

See `paper/PAPER_OUTLINE.md` for the complete NLDL-aligned paper structure, including:
- Section-by-section guidance with word counts
- Suggested ablation study design
- Equations for loss function and architecture
- NLDL submission checklist
- Reproducibility appendix template

---

## References

- Liao et al. (2025). *Estimating Pasture Biomass from Top-View Images: A Dataset for Precision Agriculture.* arXiv:2510.22916
- Kendall & Gal (2017). *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?* NeurIPS.
- Tan & Le (2021). *EfficientNetV2: Smaller Models and Faster Training.* ICML.
- [CHARMS paper — OpenReview v7I5FtL2pV — cite once confirmed]
