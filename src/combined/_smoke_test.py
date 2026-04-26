"""End-to-end smoke test for combined_pipeline.ipynb.

Runs the notebook's cells in-process with:
  - SMOKE_TEST=True  (NUM_EPOCHS=1, NUM_FOLDS=1)
  - USE_DA_FUSION=True, USE_CEMS=True, USE_CHARMS=True
  - DINOv2 extraction replaced by cached features from src/cache/
    (local env lacks `transformers`; Kaggle env has it and will run the real
     extraction — this driver only validates the *post-extraction* logic).

Exits non-zero on any failure and prints a concise status report.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# --- Working directory resolution ---
HERE      = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]     # nervous-engelbart root
os.chdir(REPO_ROOT)
print(f"cwd: {os.getcwd()}")

# --- Load the notebook ---
nb_path = HERE / "combined_pipeline.ipynb"
with open(nb_path) as f:
    nb = json.load(f)

code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
print(f"Loaded {len(code_cells)} code cells from {nb_path.name}")

# --- Shared exec namespace ---
g: dict = {"__name__": "__main__"}

# --- Cell index markers (0-based within code_cells) ---
# Run in order; we will monkey-patch between cells.
CELL_CFG          = 1   # cfg definition
CELL_DINO_LOAD    = 2   # DINOv2 load (skip in local env)
CELL_CSV_LOAD     = 3   # load train CSV (wide)
CELL_FEATURE_DEF  = 4   # extract_features def
CELL_REAL_EXTRACT = 5   # real feature extraction
CELL_TEST_EXTRACT = 6   # test feature extraction
CELL_SYNTH_LOAD   = 7   # synthetic CSV + features
# remaining cells are pure Python / torch logic


def exec_cell(idx: int, label: str):
    src = code_cells[idx]["source"]
    if isinstance(src, list):
        src = "".join(src)
    print(f"\n--- [cell {idx}] {label} ---")
    try:
        exec(compile(src, f"<cell {idx}>", "exec"), g)
    except Exception:
        print(f"!!! FAILED in cell {idx} ({label})")
        traceback.print_exc()
        sys.exit(2)


def inject(code_str: str, label: str):
    print(f"\n--- [inject] {label} ---")
    exec(compile(code_str, f"<inject {label}>", "exec"), g)


# --- Stub `transformers` for local env (Kaggle has it for real) ---
import types
_transformers_stub = types.ModuleType("transformers")
class _StubAutoModel:
    @staticmethod
    def from_pretrained(*a, **k):
        class _X:
            def eval(self): return self
            def to(self, *a, **k): return self
            def parameters(self): return []
        return _X()
_transformers_stub.AutoModel = _StubAutoModel
sys.modules["transformers"] = _transformers_stub

# --- 1. Imports ---
exec_cell(0, "imports")

# --- 2. cfg — force SMOKE_TEST=True and all flags True ---
#     (patch the cfg source inline)
cfg_src = code_cells[CELL_CFG]["source"]
if isinstance(cfg_src, list):
    cfg_src = "".join(cfg_src)
cfg_src = cfg_src.replace("SMOKE_TEST=False", "SMOKE_TEST=True")
# Also wire local paths (real data lives at data/... not /kaggle/input/...)
patched_cfg = cfg_src + (
    "\n"
    "# --- SMOKE TEST path overrides ---\n"
    "cfg.dataset_dir     = 'data'\n"
    "cfg.dino_weights_dir = 'dinov2-small'\n"
    "cfg.synthetic_dir   = 'data/image'   # relative — CSV paths start 'augmented/...'\n"
    "cfg.synthetic_csv   = 'data/tabular/train/train_augmented.csv'\n"
    "cfg.output_dir      = 'src/methods/combined/_smoke_out'\n"
    "os.makedirs(cfg.output_dir, exist_ok=True)\n"
    "\n"
    "# Path globals were already computed against /kaggle/... above; recompute.\n"
    "TRAIN_CSV     = os.path.join(cfg.dataset_dir, 'tabular', 'train', 'train.csv')\n"
    "TEST_CSV      = os.path.join(cfg.dataset_dir, 'tabular', 'test',  'test.csv')\n"
    "TRAIN_IMG_DIR = os.path.join(cfg.dataset_dir, 'image', 'train')\n"
    "TEST_IMG_DIR  = os.path.join(cfg.dataset_dir, 'image', 'test')\n"
)
print("\n--- [cell 1 patched] cfg (SMOKE_TEST=True, local paths) ---")
exec(compile(patched_cfg, "<cell 1 patched>", "exec"), g)
print(f"  USE_DA_FUSION={g['cfg'].USE_DA_FUSION}  USE_CEMS={g['cfg'].USE_CEMS}  "
      f"USE_CHARMS={g['cfg'].USE_CHARMS}  NUM_EPOCHS={g['cfg'].NUM_EPOCHS}  "
      f"NUM_FOLDS={g['cfg'].NUM_FOLDS}")

# --- 3. Skip DINOv2 load (no transformers in local env). Stub `dino`. ---
inject(
    "class _StubDINO:\n"
    "    def eval(self): return self\n"
    "    def to(self, *a, **k): return self\n"
    "    def parameters(self): return []\n"
    "dino = _StubDINO()\n"
    "print('  [stub] dino replaced with _StubDINO (smoke test; Kaggle will load real weights)')\n",
    "stub DINOv2",
)

# --- 4. Load training CSV ---
exec_cell(CELL_CSV_LOAD, "load train CSV")

# --- 5. Skip feature extract def; provide stub that uses cache ---
#     (still exec the def cell to provide _dino_transform, _ORIENTATIONS in case used later)
exec_cell(CELL_FEATURE_DEF, "extract_features def")

# --- 6. Replace real feature extraction with cache-load ---
inject(
    "# ---- Replace real DINOv2 extraction with cached features ----\n"
    "cache_feats = np.load('src/cache/features_dinov2.npy')\n"
    "cache_ids   = np.load('src/cache/image_ids.npy')\n"
    "print(f'  cache feats: {cache_feats.shape}  ids: {cache_ids.shape}')\n"
    "\n"
    "# Reorder cache to match train_image_ids_all order\n"
    "id_to_row = {str(iid): i for i, iid in enumerate(cache_ids)}\n"
    "row_perm  = np.array([id_to_row[str(iid)] for iid in train_image_ids_all])\n"
    "X_real_single = cache_feats[row_perm]    # (357, 384)\n"
    "N_real        = len(train_image_ids_all)\n"
    "\n"
    "# Apply flip4x by tiling (cache has no flip features locally — same features\n"
    "# for all 4 orientations). This is acceptable for smoke test; Kaggle will\n"
    "# compute real flips via extract_features(augment=True).\n"
    "if cfg.use_flip4x:\n"
    "    X_real         = np.repeat(X_real_single, 4, axis=0)\n"
    "    Y_real         = np.repeat(Y_all_real,    4, axis=0)\n"
    "    real_image_ids = np.repeat(train_image_ids_all, 4)\n"
    "    real_group_ids = np.repeat(np.arange(N_real),   4)\n"
    "else:\n"
    "    X_real         = X_real_single\n"
    "    Y_real         = Y_all_real\n"
    "    real_image_ids = train_image_ids_all\n"
    "    real_group_ids = np.arange(N_real)\n"
    "print(f'  X_real: {X_real.shape}  Y_real: {Y_real.shape}')\n",
    "stub real features",
)

# --- 7. Replace test feature extraction with cache ---
inject(
    "cache_test_feats = np.load('src/cache/test_features_dinov2.npy')\n"
    "cache_test_ids   = np.load('src/cache/test_image_ids.npy')\n"
    "print(f'  cache test feats: {cache_test_feats.shape}  ids: {cache_test_ids.shape}')\n"
    "\n"
    "# For the smoke test we only need test_image_ids present; test CSV load still needed.\n"
    "df_test_raw = pd.read_csv(TEST_CSV)\n"
    "df_test_raw['image_id'] = df_test_raw['sample_id'].str.split('__').str[0]\n"
    "df_test_unique = df_test_raw.drop_duplicates('image_id').copy()\n"
    "\n"
    "# Restrict test set to images we have cache for (smoke test only has 1 image cached)\n"
    "available_test_ids = set(str(x) for x in cache_test_ids)\n"
    "df_test_unique     = df_test_unique[df_test_unique['image_id'].isin(available_test_ids)].copy()\n"
    "test_image_ids     = df_test_unique['image_id'].values\n"
    "\n"
    "id_to_row = {str(iid): i for i, iid in enumerate(cache_test_ids)}\n"
    "X_test    = np.stack([cache_test_feats[id_to_row[str(iid)]] for iid in test_image_ids]).astype(np.float32)\n"
    "print(f'  X_test (smoke subset): {X_test.shape}  ids: {test_image_ids}')\n",
    "stub test features",
)

# --- 8. Synthetic CSV + features (replace extraction with cache) ---
inject(
    "if cfg.USE_DA_FUSION:\n"
    "    from pathlib import Path as _P\n"
    "    df_synth_full = pd.read_csv(cfg.synthetic_csv)\n"
    "    df_synth      = df_synth_full[df_synth_full['is_synthetic'].astype(bool)].copy()\n"
    "    df_synth['source_id'] = df_synth['source_image'].apply(lambda p: _P(str(p)).stem)\n"
    "    # CSV image_path looks like 'augmented/ID1011485656_aug0.jpg' ->\n"
    "    # id matches cached aug_ids ('ID1011485656_aug0').\n"
    "    df_synth['aug_id'] = df_synth['image_path'].apply(lambda p: _P(str(p)).stem)\n"
    "    print(f'  Synthetic rows in CSV: {len(df_synth)}')\n"
    "\n"
    "    cache_aug_feats = np.load('src/cache/augmented_features_dinov2.npy')\n"
    "    cache_aug_ids   = np.load('src/cache/augmented_image_ids.npy')\n"
    "    aug_id_to_row   = {str(iid): i for i, iid in enumerate(cache_aug_ids)}\n"
    "\n"
    "    # Intersect CSV aug_ids with cache (should be identical)\n"
    "    mask = df_synth['aug_id'].isin(set(aug_id_to_row.keys())).values\n"
    "    df_synth = df_synth[mask].reset_index(drop=True)\n"
    "    print(f'  After cache-intersection: {len(df_synth)}')\n"
    "\n"
    "    rows = np.array([aug_id_to_row[a] for a in df_synth['aug_id'].values])\n"
    "    X_synth_all = cache_aug_feats[rows].astype(np.float32)\n"
    "    Y_synth_all = df_synth[TARGETS].values.astype(np.float32)\n"
    "    synth_source_ids = df_synth['source_id'].values\n"
    "    print(f'  X_synth_all: {X_synth_all.shape}  Y_synth_all: {Y_synth_all.shape}')\n"
    "else:\n"
    "    X_synth_all      = np.zeros((0, cfg.input_dim), dtype=np.float32)\n"
    "    Y_synth_all      = np.zeros((0, cfg.output_dim), dtype=np.float32)\n"
    "    synth_source_ids = np.array([], dtype=object)\n",
    "stub synth features",
)

# --- 9. All remaining cells: model, CEMS funcs, loss, train fn, CV loop, OOF, submission ---
for i in range(CELL_SYNTH_LOAD + 1, len(code_cells)):
    exec_cell(i, f"notebook cell {i}")

# --- Report ---
print("\n" + "=" * 60)
print("SMOKE TEST: ALL CELLS EXECUTED SUCCESSFULLY")
print("=" * 60)
cfg = g["cfg"]
print(f"  USE_DA_FUSION={cfg.USE_DA_FUSION}  USE_CEMS={cfg.USE_CEMS}  USE_CHARMS={cfg.USE_CHARMS}")
print(f"  NUM_FOLDS={cfg.NUM_FOLDS}  NUM_EPOCHS={cfg.NUM_EPOCHS}")
summaries = g.get("fold_summaries", [])
for s in summaries:
    wall_times = s["epoch_wall_times"]
    print(f"  fold {s['fold']}: val_R²={s['val_r2']:.4f}  "
          f"epoch_wall_mean={np.mean(wall_times):.2f}s  "
          f"epoch_wall_max={np.max(wall_times):.2f}s")
    for t, r2 in s["per_target_r2"].items():
        print(f"    {t:<16}: R²={r2:+.4f}")
    # Wall-clock gate
    if np.max(wall_times) > 600.0:
        print(f"  *** WARNING: fold {s['fold']} epoch wall-clock > 10 min — consider capping anchors.")

print(f"\n  Outputs written to: {cfg.output_dir}")
for f in sorted(os.listdir(cfg.output_dir)):
    size = os.path.getsize(os.path.join(cfg.output_dir, f))
    print(f"    {f}  ({size} bytes)")
