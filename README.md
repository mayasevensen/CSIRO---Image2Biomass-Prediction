# CSIRO---Image2Biomass-Prediction
Build models that predict pasture biomass from images, ground-truth measurements, and publicly available datasets. Farmers will use these models to determine when and how to graze their livestock.


## CHARMS (`kaggle_baseline_pipeline_charms.ipynb`)

Implementation of [CHARMS (Jiang et al., ICML 2024)](https://proceedings.mlr.press/v235/jiang24ab.html)
on top of the DINOv2 baseline. CHARMS uses tabular side-channel attributes
that are available at training time but **not at test time**, steering the
image model towards more discriminative features via an Optimal Transport
alignment step.

### What is the same as the baseline

- Frozen DINOv2 ViT-S/14 backbone extracting 384-d CLS tokens
- Same `BiomassModel` architecture (Linear → ReLU → Dropout → Linear)
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