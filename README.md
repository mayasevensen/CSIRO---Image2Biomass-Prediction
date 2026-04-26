# CSIRO---Image2Biomass-Prediction
Build models that predict pasture biomass from images, ground-truth measurements, and publicly available datasets. Farmers will use these models to determine when and how to graze their livestock.



## How to Run

### Local

All local notebooks detect the repo root automatically and download DINOv2 weights from HuggingFace on first run - no manual setup required.

Data can be downloaded from Kaggle https://www.kaggle.com/competitions/csiro-biomass/data

| Method | Notebook |
|---|---|
| CEMS | `src/methods/cems/cems_pipeline.ipynb` |
| CHARMS | `src/methods/charms/charms_pipeline.ipynb` |
| CHARMS (legacy) | `src/legacy/pipeline_charms.ipynb` |
| DA-Fusion | `src/methods/da_fusion/da_fusion_pipeline.ipynb` |

For DA-Fusion, synthetic augmented images must exist before running the notebook. If they are not present, generate them first:

```bash
python src/methods/da_fusion/generate_augmented.py \
  --train_csv data/tabular/train/train.csv \
  --image_dir data/image/train \
  --token_dir src/methods/da_fusion/learned_tokens \
  --output_image_dir data/image/train_augmented \
  --output_csv data/tabular/train/train_augmented.csv
```

Or access the generated synthetic images via this Kaggle dataset: https://www.kaggle.com/datasets/ragnhildklette/da-fusion-synthetic-biomass.

### Kaggle

Use the `kaggle_` prefixed version of each notebook. DINOv2 weights must be uploaded as a Kaggle dataset before running:

1. Run `src/shared/download_dino_weights.py` locally - this saves weights into `./dinov2-small/`.
2. Upload the `dinov2-small/` folder as a new Kaggle dataset.
3. Attach that dataset to your notebook and run.

For DA-Fusion, access the generated synthetic images via the above linked Kaggle dataset.

---

## Shared Baseline Pipeline (`src/shared/dinov2_baseline.ipynb`) - group work

A frozen DINOv2 ViT-S/14 backbone extracts 384-d CLS tokens from each image, resized to 504×252 and normalised with ImageNet statistics. Four orientation augmentations (identity, hflip, vflip, hflip+vflip) expand each training image to four variants; GroupKFold (5 folds, grouped by source image) prevents leakage across augmented variants. Extracted features are passed through a two-stage MLP: an Encoder (384→256→64, GELU, Dropout 0.3) followed by a Head (64→32→5, GELU, Dropout 0.3). Training runs for 80 epochs with AdamW (lr=3×10⁻⁴, weight_decay=10⁻³) and cosine annealing; the objective is a per-target weighted SmoothL1 loss and the evaluation metric is the competition's weighted global R².

DINOv2 was chosen over ResNet50 because its self-supervised pre-training on diverse imagery yields richer visual representations out of the box, giving each downstream method a strong enough foundation that measured improvements reflect the method itself rather than baseline weakness.

### Results

| Model | Val R² (fold 0) | Kaggle public | Kaggle private |
|---|---|---|---|
| Baseline | 0.7554 | 0.5533 | 0.5090 |

---

## CEMS (`src/methods/cems/kaggle_cems_pipeline.ipynb`) - Victor's Contribution

Implementation of [CEMS (Curvature-Enhanced Manifold Sampling, Kaufman et al.)](https://arxiv.org/pdf/2506.06853) on top of the DINOv2 baseline. CEMS is a data augmentation method that generates synthetic training samples by estimating the local tangent space of the joint input-label manifold and sampling within it. It was chosen because the dataset is very small (~357 images) and the regression targets lie on a low-dimensional manifold (intrinsic dimensionality ~2.4), making manifold-aware augmentation a principled fit.

### What is the same as the baseline

- Frozen DINOv2 ViT-S/14 backbone extracting 384-d CLS tokens
- Same `BiomassModel` architecture (Linear → GeLU → Dropout → Linear)
- Same 4-orientation augmentation (identity, hflip, vflip, hflip+vflip)
- Same GroupKFold split (5 folds, grouped by source image)
- Same optimizer (AdamW), scheduler (cosine annealing), and training budget (80 epochs)
- Same loss function (`weighted SmoothL1`) and evaluation metric (weighted global R²)

### What CEMS adds

#### 1. Joint [X, Y_scaled] space construction
DINOv2 features are concatenated with MinMax-scaled labels to form a joint representation `zi = [x | y_scaled]`. The intrinsic dimension of this space (~2) is estimated via TwoNN (Facco et al. 2017) and used as the manifold dimension `d` for local SVD decomposition.

#### 2. Sigma calibration
Perturbation noise `σ` is either fixed (`sigma_auto=False`) or derived automatically from the median nearest-neighbour distance in the joint `z`-space, scaled by `sigma_fraction`. Using a single de-duplicated representative per image for the NN computation is controlled by `sigma_dedup`.

#### 3. CEMS training loop replacing standard ERM
Each training step builds a batch around a randomly selected anchor image (one orientation drawn per epoch, without replacement across anchors). Neighbours are selected from the training set (`neigh_type`: `random`, `knn`, or `knnp`). The anchor batch is passed through `get_batch_cems`, which applies local SVD, fits a first- and second-order polynomial in the tangent bundle via ridge regression, samples a perturbation `ν`, and returns synthetic `(x_new, y_new)` pairs. The model is trained on the loss against the inverse-scaled synthetic labels.

### CEMS algorithm implementation

The core CEMS functions were ported from the [azencot-group/CEMS](https://github.com/azencot-group/CEMS) repository. `_estimate_np` and `_solve_ridge_regression` were copied with no changes. The remaining functions - `_adjust_dims`, `_get_projection`, `_estimate_grad_hessian`, `_sample_tangent`, and `get_batch_cems` - were adapted: the `cems_method=2` branch was removed throughout, `xk`/`yk` parameters were dropped, the SVD driver was made conditional on CUDA to support CPU execution, `triu_indices` was given an explicit `device=` argument to fix a CUDA failure, and `get_batch_cems` was extended with a numpy-based intrinsic dimension path and a finite/range fallback guard.

### Results

| Model | Val R² (fold 0) | Kaggle public | Kaggle private |
|---|---|---|---|
| Baseline | 0.7554 | 0.5533 | 0.5090 |
| CEMS | 0.7668 | 0.5497 | 0.5106 |
| Δ | +0.0114 | −0.0036 | +0.0016 |

CEMS improves both validation R² and private leaderboard score over the baseline, while public LB is marginally lower - consistent with the small dataset making single-fold validation noisy.

---

## CHARMS (`kaggle_baseline_pipeline_charms.ipynb`) - Maya's Contribution

Implementation of [CHARMS (Jiang et al., ICML 2024)](https://proceedings.mlr.press/v235/jiang24ab.html)
on top of the DINOv2 baseline. CHARMS uses tabular side-channel attributes
that are available at training time but **not at test time**, steering the
image model towards more discriminative features via an Optimal Transport
alignment step.

### What is the same as the baseline

- Frozen DINOv2 ViT-S/14 backbone extracting 384-d CLS tokens
- Same `BiomassModel` architecture (Linear → GeLU → Dropout → Linear)
- Same 4-orientation augmentation (identity, hflip, vflip, hflip+vflip)
- Same GroupKFold split (5 folds, grouped by source image)
- Same optimizer (AdamW), scheduler (cosine annealing), and training budget (80 epochs)
- Same loss function (`weighted SmoothL1`) and evaluation metric (weighted global R²)
- Test inference is image-only - tabular data is not used at prediction time

### What is added by CHARMS

#### 1. Tabular feature preparation (cell 4b)
Four attributes are extracted from `train.csv` and used only during training:

| Attribute | Type | Preprocessing |
|---|---|---|
| `Pre_GSHH_NDVI` | Numerical | z-score normalised |
| `Height_Ave_cm` | Numerical | z-score normalised |
| `State` | Categorical | label-encoded |
| `Species` | Categorical | label-encoded |

#### 2. K-Means channel clustering (cell 7b)
The 384 DINOv2 feature dimensions are clustered into `k=40` groups using
K-Means. This is an adaptation from the original paper, which clusters CNN
spatial feature maps. Here, dimensions with similar cross-sample activation
patterns are grouped together before the OT step.

#### 3. Three new modules (cell 8b)

**`FTTransformer`** - encodes the 4 tabular attributes into per-attribute
embeddings (used for OT alignment) and produces an auxiliary biomass
prediction from a `[CLS]` token (used to keep the embeddings task-relevant).

**`Li2tHeads`** - one prediction head per tabular attribute. Each head
takes the OT-weighted image features and predicts the attribute value:
MSE loss for numerical attributes, cross-entropy for categorical ones.

**`update_ot_transfer_matrix`** - computes the OT transfer matrix
`T ∈ R^{4 × 384}` by:
1. Computing inter-sample cosine similarity for each of the 40 channel clusters → `S_I`
2. Computing inter-sample cosine similarity for each of the 4 tabular attribute embeddings → `S_T`
3. Building a cost matrix `C[j, k]`
4. Solving exact EMD (via [POT](https://pythonot.github.io/)) with uniform marginals
5. Mapping cluster-level weights back to individual dimensions

The matrix is recomputed every 5 epochs on a random subsample of 512
training examples.

#### 4. Extended training loop (cell 10b)
The CHARMS training loop replaces the baseline loop and optimises a
three-term loss:

L = L_image  +  0.5 × L_tabular  +  0.1 × L_i2t

- `L_image` - weighted SmoothL1 on biomass targets from the image model (same as baseline)
- `L_tabular` - weighted SmoothL1 on biomass targets from the FT-Transformer
- `L_i2t` - attribute prediction loss from the Li2t heads (MSE + cross-entropy)

All three components are optimised jointly with a single AdamW optimizer.
Only the image model weights are saved as the best checkpoint and used at
test time.

#### 5. Additional dependency
CHARMS requires the [POT](https://pythonot.github.io/) library for the exact
EMD solver. It is installed automatically in the notebook if not present:
```python
pip install POT
```

### Results

| Model | Val R² (fold 0) | Kaggle public | Kaggle private |
|---|---|---|---|
| Baseline | 0.7935 | 0.5533 | 0.5090 |
| CHARMS | 0.7774 | 0.5435 | 0.5080 |
| Δ | -0.0161 | −0.0098 | −0.0010 |

CHARMS improves validation R² by -0.0161 and reaches its best checkpoint
later (epoch 74 vs. epoch 62 for the baseline), suggesting the auxiliary
losses slow down overfitting slightly. The gain does not transfer to the
Kaggle leaderboard, likely due to the small dataset making single-fold
validation noisy.


# DA-Fusion Augmentation (`kaggle_baseline_pipeline_charms.ipynb`) - Ragnhild's Contribution

## Method
Adaptation of DA-Fusion (Trabucco et al., ICLR 2024) to the 
Image2Biomass regression task.

## Contribution included:
- Implemented textual inversion pipeline for pasture species tokens
- Generated synthetic images using Stable Diffusion img2img with the learned tokens
- Added 1071 synthetic labeled training samples
- Designed fold-safe synthetic filtering to prevent leakage
- Integrated synthetic data into DINOv2 training pipeline

## Files
- textual_inversion_train.py: trains one token per species
- generate_augmented.py: generates synthetic images via img2img
- da_fusion_pipeline.ipynb: comparison experiment (3 conditions)
- kaggle_da_fusion_pipeline.ipynb: same as above but compatible with Kaggle environment
- learned_tokens/: 9 trained token embeddings

## How to reproduce
1. Run textual_inversion_train.py on a GPU 
2. Run generate_augmented.py to generate synthetic images
3. Run da_fusion_pipeline.ipynb locally or kaggle_da_fusion_pipeline.ipynb on Kaggle to compare methods


The generated synthetic images can be accessed by following this link: https://www.kaggle.com/datasets/ragnhildklette/da-fusion-synthetic-biomass, since they are too large to host on GitHub. The notebooks will automatically download the features extracted from these synthetic images, so you can run the comparison experiment without needing to generate the synthetic data yourself.


More information on the implementation and results can be found in the respective notebooks.

# Combined pipeline (`src/methods/combined/kaggle_combined_pipeline.ipynb`) - group work

All three methods are gated by boolean config flags (`USE_DA_FUSION`, `USE_CEMS`, `USE_CHARMS`); setting all to `False` reproduces the plain baseline.

### What is the same as the baseline

- Frozen DINOv2 ViT-S/14 backbone extracting 384-d CLS tokens
- Same `BiomassModel` architecture (Linear → GeLU → Dropout → Linear)
- Same 4-orientation augmentation (identity, hflip, vflip, hflip+vflip)
- Same GroupKFold split (5 folds, grouped by source image)
- Same optimizer (AdamW), scheduler (cosine annealing), and training budget (80 epochs)
- Same loss function (`weighted SmoothL1`) and evaluation metric (weighted global R²)

### What the combined pipeline adds

- Full 5-fold CV loop with per-fold OOF predictions; test predictions are averaged across all folds
- OOF predictions and fold summaries persisted to disk (`oof_combined.npz`, `fold_summaries.json`)

### Deviations from individual method implementations

#### CEMS

- `sigma_auto` option and `calibrate_sigma()` removed — sigma is always fixed via `cfg.sigma`
- Anchor loop restructured: iterates over a random permutation of all N individual samples with a `used` set (without-replacement at sample level), rather than the standalone's group-indexed loop that draws one sample per unique image group for `N_train // batch_size` steps per epoch
- `build_group_index` removed
- `MAX_ANCHORS_PER_EPOCH` cap added (absent in standalone)

#### CHARMS

- Li2t loss and OT subsample are restricted to real samples only (via `is_real` mask), because DA-Fusion synthetic samples carry no tabular data; standalone CHARMS applies Li2t to all training samples
- `CombinedDataset` (carries `is_real` flag alongside tabular features) replaces the standalone's `CHARMSDataset`

#### DA-Fusion

- Synthetic leakage filter is applied per fold (the set of kept synthetics changes with each fold's training split); standalone filters once for fold 0 only