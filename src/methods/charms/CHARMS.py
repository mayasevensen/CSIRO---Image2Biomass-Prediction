"""
CHARMS (CHannel tAbulaR alignment with optiMal tranSport) on the Image2Biomass
baseline.

Follows Jiang et al., *Tabular Insights, Visual Impacts: Transferring Expertise
from Tables to Images* (ICML 2024). The paper aligns image channels with
tabular attributes via Sinkhorn optimal transport, using the resulting
transport matrix T ∈ R^{D×C} to gate image channels when predicting each
attribute — forcing the image backbone to encode the tabular signal in
attribute-specific channels.

Training loss (paper eq. 7–8):

    L = ℓ(f(x^I), y)                                  # image task
      + ℓ(g(x^T), y)                                  # tabular task
      + L_i2t                                          # image-to-tab aux
    L_i2t = Σ_p MSE(T_p · φ(x^I), x_num_p)
          + Σ_q CE(T_q · φ(x^I), x_cat_q)

At inference the tabular branch is discarded — only the image path is used, so
the model is a drop-in replacement for the baseline BiomassModel.

Adaptations for the Image2Biomass setting
-----------------------------------------
* Baseline architecture is preserved exactly: DINOv2 CLS → Encoder(384→128→32)
  → BiomassHead(32→32→5). CHARMS treats the 32-d latent as C=32 image channels
  — no K-means is needed since the latent is already low-dimensional.
* Tabular encoder ψ is a small FT-Transformer-lite over 2 numeric + 2
  categorical attributes (NDVI, Height; State, Month).
* For scalar image channels, cosine sample-similarity degenerates to sign
  agreement, so we use exp(-|z_a - z_b|²/σ²) instead, which matches the
  paper's "sample-wise similarity" spirit with a non-degenerate signal.
* Transport matrix T is recomputed every cfg.cost_update_every epochs from a
  batch of encoded training samples.
* The primary biomass task on the image side uses the same weighted
  SmoothL1 as the baseline, so CHARMS vs. baseline is a fair comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from shared.model import Encoder, Head
from shared.data_utils import _make_wide_df, _image_id_from_path, TARGETS


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CharmsConfig:
    """Hyperparameters for CHARMS model and training loop."""

    # --- Baseline image trunk (preserved exactly) ---
    input_dim: int = 384
    latent_dim: int = 32
    output_dim: int = 5
    dropout: float = 0.1

    # --- Tabular encoder (FT-Transformer-lite) ---
    attr_embed_dim: int = 64
    tab_transformer_heads: int = 4
    tab_transformer_layers: int = 2
    tab_ff_mult: int = 2

    # --- Metadata schema (populated by load_charms_datasets) ---
    numeric_attrs: Tuple[str, ...] = ("Pre_GSHH_NDVI", "Height_Ave_cm")
    categorical_attrs: Tuple[str, ...] = ("State", "month")
    cat_cardinalities: Tuple[int, ...] = ()

    # --- CHARMS OT alignment ---
    sinkhorn_epsilon: float = 0.05      # OT entropy strength
    sinkhorn_iters: int = 50
    similarity_sigma: float = 1.0       # scale for scalar-channel similarity
    cost_update_every: int = 5          # epochs — matches paper §4.3
    cost_update_samples: int = 256      # # samples used when recomputing T

    # --- Loss weights ---
    lambda_tab: float = 0.5             # ℓ(g(x^T), y) coefficient
    lambda_i2t_num: float = 0.25        # Σ_p MSE aux-image-to-numeric
    lambda_i2t_cat: float = 0.10        # Σ_q CE aux-image-to-categorical


# ---------------------------------------------------------------------------
# Tabular encoder (FT-Transformer-lite)
# ---------------------------------------------------------------------------

class TabularEncoder(nn.Module):
    """
    FT-Transformer-lite. Each attribute becomes a token:
      - Numeric attr p:  x_p → Linear(1 → E)
      - Categorical attr q: Embedding(card_q → E)
    A learnable [CLS] token is prepended, the D+1 tokens are passed through a
    small TransformerEncoder, and:
      * attr_tokens = output tokens 1..D   (used for OT alignment)
      * task_pred   = task_head(CLS)      (tabular branch's 5 biomass preds)
    """

    def __init__(self, cfg: CharmsConfig):
        super().__init__()
        self.cfg = cfg
        E = cfg.attr_embed_dim

        # Per-numeric Linear(1→E)
        self.num_embs = nn.ModuleList([
            nn.Linear(1, E) for _ in cfg.numeric_attrs
        ])
        # Per-categorical Embedding(card→E)
        assert len(cfg.cat_cardinalities) == len(cfg.categorical_attrs), (
            "cat_cardinalities must match categorical_attrs length"
        )
        self.cat_embs = nn.ModuleList([
            nn.Embedding(card, E) for card in cfg.cat_cardinalities
        ])

        self.cls_token = nn.Parameter(torch.zeros(1, 1, E))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=E,
            nhead=cfg.tab_transformer_heads,
            dim_feedforward=E * cfg.tab_ff_mult,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=cfg.tab_transformer_layers)
        self.norm = nn.LayerNorm(E)

        # Tabular-branch task head (5 biomass predictions from CLS)
        self.task_head = nn.Sequential(
            nn.Linear(E, E),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(E, cfg.output_dim),
        )

    def forward(
        self, x_num: torch.Tensor, x_cat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_num: (B, P) float — z-scored numeric attributes
            x_cat: (B, Q) long  — integer-encoded categoricals
        Returns:
            attr_tokens: (B, D, E)  — one contextual embedding per attribute
            task_pred  : (B, output_dim) — tabular-branch biomass predictions
        """
        tokens: List[torch.Tensor] = []
        for i, emb in enumerate(self.num_embs):
            tokens.append(emb(x_num[:, i:i+1]))
        for j, emb in enumerate(self.cat_embs):
            tokens.append(emb(x_cat[:, j]))
        attr_tokens = torch.stack(tokens, dim=1)                 # (B, D, E)

        B = attr_tokens.shape[0]
        cls = self.cls_token.expand(B, -1, -1)                   # (B, 1, E)
        x = torch.cat([cls, attr_tokens], dim=1)                 # (B, D+1, E)
        x = self.transformer(x)
        x = self.norm(x)

        cls_out = x[:, 0]                                         # (B, E)
        attr_out = x[:, 1:]                                       # (B, D, E)
        task_pred = self.task_head(cls_out)                       # (B, output_dim)
        return attr_out, task_pred


# ---------------------------------------------------------------------------
# Sinkhorn OT + channel / attribute similarity
# ---------------------------------------------------------------------------

@torch.no_grad()
def sinkhorn_uniform(
    cost: torch.Tensor, epsilon: float = 0.05, n_iters: int = 50
) -> torch.Tensor:
    """Entropic OT with uniform row / column marginals. cost: (D, C) → T (D, C)."""
    D, C = cost.shape
    device, dtype = cost.device, cost.dtype
    a = torch.full((D,), 1.0 / D, device=device, dtype=dtype)
    b = torch.full((C,), 1.0 / C, device=device, dtype=dtype)

    # Numerical-stability: subtract min so exp doesn't underflow
    cost_shifted = cost - cost.min()
    K = torch.exp(-cost_shifted / max(epsilon, 1e-8)).clamp_min(1e-12)

    u = torch.ones_like(a)
    v = torch.ones_like(b)
    for _ in range(n_iters):
        u = a / (K @ v + 1e-12)
        v = b / (K.T @ u + 1e-12)
    T = torch.diag(u) @ K @ torch.diag(v)
    return T


def _scalar_channel_similarity(z: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Pairwise similarity over samples for each scalar channel.

    Args:
        z: (N, C) — latents for N samples
    Returns:
        (C, N, N) — Gaussian similarity S^I_c[a, b] = exp(-(z_a - z_b)² / σ²)
        per channel. Non-degenerate replacement for cosine on scalars.
    """
    # Pairwise differences per channel:  (C, N, N)
    z_t = z.T                                          # (C, N)
    diff = z_t.unsqueeze(2) - z_t.unsqueeze(1)         # (C, N, N)
    sim = torch.exp(-(diff ** 2) / max(sigma ** 2, 1e-8))
    return sim


def _attribute_similarity(attr_tokens: torch.Tensor) -> torch.Tensor:
    """
    Cosine sample-similarity per attribute.

    Args:
        attr_tokens: (N, D, E)
    Returns:
        (D, N, N) — S^T_d[a, b] = cos(attr_tokens[a, d], attr_tokens[b, d])
    """
    a = F.normalize(attr_tokens, dim=-1)               # (N, D, E)
    a = a.permute(1, 0, 2)                             # (D, N, E)
    sim = torch.einsum("dni,dmi->dnm", a, a)           # (D, N, N)
    return sim


@torch.no_grad()
def compute_transport_matrix(
    latent: torch.Tensor,           # (N, C)
    attr_tokens: torch.Tensor,      # (N, D, E)
    sigma: float = 1.0,
    epsilon: float = 0.05,
    n_iters: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Paper §4.2. Build cost matrix from sample-wise similarity structures, then
    solve Sinkhorn OT.

    Returns:
        T    : (D, C) transport matrix
        cost : (D, C) cost used to solve OT (useful for logging)
    """
    S_I = _scalar_channel_similarity(latent, sigma=sigma)        # (C, N, N)
    S_T = _attribute_similarity(attr_tokens)                     # (D, N, N)
    # C[d, c] = mean_{a,b} (S_T[d, a, b] - S_I[c, a, b])²
    cost = ((S_T.unsqueeze(1) - S_I.unsqueeze(0)) ** 2).mean(dim=(2, 3))  # (D, C)
    T = sinkhorn_uniform(cost, epsilon=epsilon, n_iters=n_iters)
    return T, cost


# ---------------------------------------------------------------------------
# CHARMS model
# ---------------------------------------------------------------------------

class BiomassCharmsModel(nn.Module):
    """
    Image trunk = baseline (Encoder + BiomassHead). Extra machinery is used
    only during training: a tabular encoder, per-attribute auxiliary heads,
    and a buffer holding the current OT transport matrix.

    At inference, `forward(x)` runs exactly the baseline path — the aux
    heads / tabular encoder are never touched.
    """

    def __init__(self, cfg: CharmsConfig):
        super().__init__()
        self.cfg = cfg

        # Image branch — identical to baseline
        self.encoder = Encoder(cfg.input_dim, cfg.latent_dim, cfg.dropout)
        self.biomass_head = Head(cfg.latent_dim, cfg.output_dim, cfg.dropout)

        # Tabular branch
        self.tab_encoder = TabularEncoder(cfg)

        # Auxiliary heads: read a T_d-gated channel vector → predict attribute
        self.num_aux_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cfg.latent_dim, 32),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(32, 1),
            )
            for _ in cfg.numeric_attrs
        ])
        self.cat_aux_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cfg.latent_dim, 32),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(32, card),
            )
            for card in cfg.cat_cardinalities
        ])

        # Transport matrix, recomputed every cfg.cost_update_every epochs.
        # Initialise to uniform so early batches have a sensible gating.
        D = len(cfg.numeric_attrs) + len(cfg.categorical_attrs)
        init = torch.full((D, cfg.latent_dim), 1.0 / (D * cfg.latent_dim))
        self.register_buffer("transport", init)

    # ------------------------------------------------------------------
    # Inference (baseline-compatible)
    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.biomass_head(self.encoder(x))

    # ------------------------------------------------------------------
    # Training path
    # ------------------------------------------------------------------
    def forward_train(
        self,
        x_img: torch.Tensor,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Single training step forward. Returns everything the loss needs.
        """
        z = self.encoder(x_img)                                # (B, C)
        y_img = self.biomass_head(z)                           # (B, output_dim)

        attr_tokens, y_tab = self.tab_encoder(x_num, x_cat)    # (B, D, E), (B, out)

        P = len(self.cfg.numeric_attrs)
        Q = len(self.cfg.categorical_attrs)

        # Aligned aux predictions. Normalise the rows of T so each attribute
        # gets a unit-sum gating (T from Sinkhorn has tiny magnitudes
        # 1 / (D*C); scaling preserves direction but stabilises heads).
        T = self.transport                                     # (D, C)
        T_row_scaled = T * T.shape[1]                          # sum per row ≈ 1

        num_preds: List[torch.Tensor] = []
        for p, head in enumerate(self.num_aux_heads):
            gated = z * T_row_scaled[p].unsqueeze(0)           # (B, C)
            num_preds.append(head(gated).squeeze(-1))          # (B,)

        cat_logits: List[torch.Tensor] = []
        for q, head in enumerate(self.cat_aux_heads):
            idx = P + q
            gated = z * T_row_scaled[idx].unsqueeze(0)
            cat_logits.append(head(gated))                      # (B, card_q)

        return {
            "latent": z,
            "attr_tokens": attr_tokens,
            "y_img": y_img,
            "y_tab": y_tab,
            "num_preds": num_preds,
            "cat_logits": cat_logits,
        }

    # ------------------------------------------------------------------
    # Cost-matrix / transport refresh
    # ------------------------------------------------------------------
    @torch.no_grad()
    def refresh_transport(self, x_img: torch.Tensor, x_num: torch.Tensor,
                          x_cat: torch.Tensor) -> torch.Tensor:
        """
        Recompute the OT transport matrix from a (preferably large) batch of
        training samples. Called every cfg.cost_update_every epochs.
        """
        was_training = self.training
        self.eval()
        z = self.encoder(x_img)                                # (N, C)
        attr_tokens, _ = self.tab_encoder(x_num, x_cat)        # (N, D, E)
        if was_training:
            self.train()

        T, _ = compute_transport_matrix(
            z, attr_tokens,
            sigma=self.cfg.similarity_sigma,
            epsilon=self.cfg.sinkhorn_epsilon,
            n_iters=self.cfg.sinkhorn_iters,
        )
        self.transport.copy_(T)
        return T


# ---------------------------------------------------------------------------
# Metadata / dataset
# ---------------------------------------------------------------------------

@dataclass
class MetadataSpec:
    """Metadata arrays aligned to the cached DINOv2 feature index."""
    image_ids: np.ndarray
    numeric: np.ndarray
    categorical: np.ndarray
    numeric_attrs: List[str]
    categorical_attrs: List[str]
    cat_cardinalities: List[int]
    cat_classes: List[List[str]]
    numeric_mean: np.ndarray
    numeric_std: np.ndarray


def build_metadata_arrays(
    csv_path: str | Path,
    image_ids: np.ndarray,
    numeric_attrs: Tuple[str, ...] = ("Pre_GSHH_NDVI", "Height_Ave_cm"),
    categorical_attrs: Tuple[str, ...] = ("State", "month"),
    fit_ids: Optional[List[str]] = None,
) -> MetadataSpec:
    """Align metadata to the feature-cache ordering; z-score numerics, label-encode cats.

    Args:
        fit_ids: IDs used to fit the numeric scaler. If None, fit on all rows
            (legacy behaviour — introduces mild distributional leakage from
            val/test into the training scaler). Pass the training split IDs
            for a clean fit.
    """
    df = _make_wide_df(csv_path)
    df["Sampling_Date"] = pd.to_datetime(df["Sampling_Date"])
    df["month"] = df["Sampling_Date"].dt.month.astype(int)
    df["image_id"] = df["image_path"].apply(_image_id_from_path)
    df = df.drop_duplicates("image_id").set_index("image_id")

    # Numeric → z-score
    numeric_raw = np.zeros((len(image_ids), len(numeric_attrs)), dtype=np.float32)
    for k, iid in enumerate(image_ids):
        if iid in df.index:
            for p, col in enumerate(numeric_attrs):
                numeric_raw[k, p] = float(df.loc[iid, col])

    # Fit mean/std on fit_ids only (default: training split) so val/test
    # rows don't contribute to the scaler statistics.
    if fit_ids is None:
        fit_mask = np.ones(len(image_ids), dtype=bool)
    else:
        fit_set = set(fit_ids)
        fit_mask = np.array([iid in fit_set for iid in image_ids], dtype=bool)
        assert fit_mask.any(), "fit_ids produced an empty mask — check ID alignment"

    mean = numeric_raw[fit_mask].mean(axis=0)
    std = numeric_raw[fit_mask].std(axis=0) + 1e-8
    numeric = ((numeric_raw - mean) / std).astype(np.float32)

    # Categorical → label-encode (stable alphabetical order → reproducible)
    cat_values = np.zeros((len(image_ids), len(categorical_attrs)), dtype=np.int64)
    cat_cards: List[int] = []
    cat_classes: List[List[str]] = []
    for q, col in enumerate(categorical_attrs):
        classes = sorted(df[col].dropna().astype(str).unique().tolist())
        class_to_idx = {c: i for i, c in enumerate(classes)}
        for k, iid in enumerate(image_ids):
            if iid in df.index:
                cat_values[k, q] = class_to_idx.get(str(df.loc[iid, col]), 0)
        cat_cards.append(len(classes))
        cat_classes.append(classes)

    return MetadataSpec(
        image_ids=image_ids,
        numeric=numeric,
        categorical=cat_values,
        numeric_attrs=list(numeric_attrs),
        categorical_attrs=list(categorical_attrs),
        cat_cardinalities=cat_cards,
        cat_classes=cat_classes,
        numeric_mean=mean.astype(np.float32),
        numeric_std=std.astype(np.float32),
    )


class CharmsDataset(Dataset):
    """(X, y, num, cat) dataset built on the cached DINOv2 features."""

    def __init__(
        self,
        ids: List[str],
        features: np.ndarray,
        labels: np.ndarray,
        meta: MetadataSpec,
        id_to_idx: Dict[str, int],
    ):
        indices = [id_to_idx[i] for i in ids if i in id_to_idx]
        self.X = torch.tensor(features[indices], dtype=torch.float32)
        self.y = torch.tensor(labels[indices], dtype=torch.float32)
        self.num = torch.tensor(meta.numeric[indices], dtype=torch.float32)
        self.cat = torch.tensor(meta.categorical[indices], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "X": self.X[idx],
            "y": self.y[idx],
            "num": self.num[idx],
            "cat": self.cat[idx],
        }


def load_charms_datasets(
    csv_path: str | Path,
    cache_dir: str | Path,
    numeric_attrs: Tuple[str, ...] = ("Pre_GSHH_NDVI", "Height_Ave_cm"),
    categorical_attrs: Tuple[str, ...] = ("State", "month"),
    val_fold: int = 0,
    n_splits: int = 5,
    group_by: str = "visit",
) -> Tuple[CharmsDataset, CharmsDataset, MetadataSpec]:
    """Mirror of shared.data_utils.load_datasets but yielding CharmsDataset + metadata.

    Uses the shared `make_splits` helper so the baseline and CHARMS see the
    exact same train/val partition. Metadata z-score stats are fit on the
    training split only.
    """
    from shared.data_utils import make_splits

    cache_dir = Path(cache_dir)
    features = np.load(cache_dir / "features_dinov2.npy")
    image_ids = np.load(cache_dir / "image_ids.npy", allow_pickle=True)

    df = _make_wide_df(csv_path)
    id_to_label = {
        _image_id_from_path(row["image_path"]): row[TARGETS].values.astype(np.float32)
        for _, row in df.iterrows()
    }
    id_to_idx = {iid: i for i, iid in enumerate(image_ids)}

    # Split FIRST so metadata scaling can be fit on the training split only.
    train_ids, val_ids = make_splits(
        csv_path, n_splits=n_splits, val_fold=val_fold, group_by=group_by
    )

    missing = [iid for iid in list(train_ids) + list(val_ids)
               if iid not in id_to_label]
    assert not missing, (
        f"{len(missing)} split IDs have no label in the CSV "
        f"(first few: {missing[:5]})"
    )

    labels = np.stack([
        id_to_label.get(iid, np.zeros(5, dtype=np.float32))
        for iid in image_ids
    ])

    meta = build_metadata_arrays(
        csv_path=csv_path,
        image_ids=image_ids,
        numeric_attrs=numeric_attrs,
        categorical_attrs=categorical_attrs,
        fit_ids=train_ids,                  # <- scaler sees train only
    )

    train_ds = CharmsDataset(train_ids, features, labels, meta, id_to_idx)
    val_ds = CharmsDataset(val_ids, features, labels, meta, id_to_idx)
    return train_ds, val_ds, meta
