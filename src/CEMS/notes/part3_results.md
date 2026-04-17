# Part 3 Results — CEMS Implementation

> Placeholder: fill in ablation table after running pipeline_cems.ipynb on GPU.

## Deliverables Created

- `src/CEMS/cems.py` — core CEMS sampler, port of reference cems.py with `use_hessian` flag
- `src/CEMS/cems_utils.py` — `precompute_knn` + `compute_neigh_size`
- `src/CEMS/model.py` — `BiomassModel.forward_cems` added
- `src/CEMS/train.py` — `train_cems` function added
- `src/CEMS/tests/test_cems_sanity.py` — three sanity checks, all passing
- `src/CEMS/pipeline_cems.ipynb` — notebook wiring all three runs

## Sanity Test Results (CPU, synthetic 2D manifold)

- (a) Full CEMS mean manifold error = 0.083 < 0.15 ✓
- (b) Gradient non-zero, correct shape (32, 1, 2) ✓
- (c) Hessian flag produces different results from linear-only ✓
- (d) aug/real loss ratio ≈ 1.0 in 2-epoch smoke test ✓

## Hyperparameters

| Parameter    | Value | Source |
|---|---|---|
| `sigma`      | `1e-3` | Reference code default |
| `cems_method`| `1`   | Batch-centred SVD |
| `neigh_size` | `32`  | `next_power_of_2(5 + 15 + 1)` at d=5 |
| `d`          | re-estimated per batch via TwoNN (latent=True) | |
| `use_hessian`| True / False | Full CEMS vs CEMS-L ablation |

## Ablation Table

> Run pipeline_cems.ipynb to populate.

| Method      | Val R² | RMSE Green | RMSE Dead | RMSE Clover | RMSE GDM | RMSE Total |
|-------------|--------|------------|-----------|-------------|----------|------------|
| Baseline    | TBD    | TBD        | TBD       | TBD         | TBD      | TBD        |
| CEMS-L      | TBD    | TBD        | TBD       | TBD         | TBD      | TBD        |
| CEMS (full) | TBD    | TBD        | TBD       | TBD         | TBD      | TBD        |

## Deviations from Plan

- `driver="gesvd"` is CUDA-only; code auto-detects and omits on CPU/MPS.
- Training loop marks only the anchor (not its neighbours) as used per epoch
  → N_train iterations per epoch, each sample serves as anchor exactly once.
