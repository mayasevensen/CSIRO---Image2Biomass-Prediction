# Part 1 Results — Feature Extraction & ID Estimation

- N = 357 training images
- Estimator: TwoNN (skdim.id.TwoNN)

## Intrinsic Dimension

| Representation        | Ambient dim | d (TwoNN) |
|-----------------------|-------------|-----------|
| (a) DINOv2 raw feats  | 384         | 12.45     |
| (b) ResNet50 raw feats| 2048        | 11.66     |
| (c) Labels only       | 5           | 2.39      |

## d² vs Batch Sizes (DINOv2 d = 12.45)

| Batch size | d²    | Feasible? |
|-----------|-------|-----------|
| 16        | 155.1 | No (<d²)  |
| 32        | 155.1 | No (<d²)  |
| 64        | 155.1 | No (<d²)  |

## Encoder Comparison Verdict

- ResNet50 has marginally lower raw ID (11.66 vs 12.45 for DINOv2), contrary to the plan's expectation.
- The difference is small (< 1 unit). Both encoders produce similar effective dimensionality in raw feature space.
- DINOv2 remains the preferred encoder for the full pipeline: better-conditioned self-supervised features, no center-crop needed, single forward pass on 2:1 images at native aspect ratio.
- The ID gap at the raw-feature level does not change the encoder choice — it is an input to the paper's analysis, not a decision criterion here.

## Key Note

- Raw-feature d² ≈ 155 is far above any feasible batch size.
- This was expected — CEMS is intended to run on the **learned 32-d latent** (App F), not raw features.
- The decisive feasibility verdict comes from the latent ID estimate at the end of Part 2.
- Labels d ≈ 2.4 is a healthy sanity check (well below the ambient dim of 5).
