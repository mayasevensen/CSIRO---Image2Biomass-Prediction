import pandas as pd
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

TARGETS = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]

def make_wide_train(train_csv: str) -> pd.DataFrame:
    df = pd.read_csv(train_csv)

    meta_cols = [
        "sample_id", "image_path", "Sampling_Date", "State", "Species",
        "Pre_GSHH_NDVI", "Height_Ave_cm"
    ]

    meta = df[meta_cols].drop_duplicates("image_path").copy()

    y = (
        df.pivot_table(
            index="image_path",
            columns="target_name",
            values="target",
            aggfunc="first"
        )
        .reset_index()
    )

    out = meta.merge(y, on="image_path", how="inner")
    out["Sampling_Date"] = pd.to_datetime(out["Sampling_Date"])
    out["month"] = out["Sampling_Date"].dt.month.astype(int)
    out["dayofyear"] = out["Sampling_Date"].dt.dayofyear.astype(int)
    return out



@dataclass
class CharmsConfig:
    backbone_name: str = "tf_efficientnetv2_s.in21k_ft_in1k"
    pretrained: bool = True
    num_targets: int = 5

    # Tabular attributes used only during training
    num_numeric_attrs: int = 2  # NDVI, height
    cat_cardinalities: Tuple[int, ...] = (4, 12)  # state_count, month_count example
    attr_embed_dim: int = 128

    # CHARMS alignment
    reduced_channels: int = 64
    ot_epsilon: float = 0.05
    sinkhorn_iters: int = 30

    # Loss weights
    lambda_numeric: float = 0.25
    lambda_categorical: float = 0.25
    lambda_alignment: float = 0.05


class TabularAttributeEncoder(nn.Module):
    """
    Produces one embedding per tabular attribute.
    Numerical attrs: learned projection from scalar -> embedding
    Categorical attrs: embedding table per attribute
    """
    def __init__(self, num_numeric_attrs: int, cat_cardinalities: Tuple[int, ...], embed_dim: int):
        super().__init__()
        self.num_numeric_attrs = num_numeric_attrs
        self.cat_cardinalities = cat_cardinalities
        self.embed_dim = embed_dim

        self.num_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )
            for _ in range(num_numeric_attrs)
        ])

        self.cat_embs = nn.ModuleList([
            nn.Embedding(cardinality, embed_dim)
            for cardinality in cat_cardinalities
        ])

        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """
        x_num: [B, P]
        x_cat: [B, Q]
        returns: attr_tokens [B, D, E]
        """
        tokens = []

        for i in range(self.num_numeric_attrs):
            tok = self.num_mlps[i](x_num[:, i:i+1])  # [B, E]
            tokens.append(tok)

        for j, emb in enumerate(self.cat_embs):
            tok = emb(x_cat[:, j])  # [B, E]
            tokens.append(tok)

        attr_tokens = torch.stack(tokens, dim=1)  # [B, D, E]
        return self.out_norm(attr_tokens)


def pairwise_cosine_sim(x: torch.Tensor) -> torch.Tensor:
    """
    x: [B, F]
    returns [B, B]
    """
    x = F.normalize(x, dim=-1)
    return x @ x.T


@torch.no_grad()
def sinkhorn_uniform(cost: torch.Tensor, epsilon: float = 0.05, n_iters: int = 30) -> torch.Tensor:
    """
    Entropic OT with uniform marginals.
    cost: [D, C]
    returns transport matrix T: [D, C]
    """
    D, C = cost.shape
    a = torch.full((D,), 1.0 / D, device=cost.device, dtype=cost.dtype)
    b = torch.full((C,), 1.0 / C, device=cost.device, dtype=cost.dtype)

    K = torch.exp(-cost / epsilon).clamp_min(1e-12)
    u = torch.ones_like(a)
    v = torch.ones_like(b)

    for _ in range(n_iters):
        u = a / (K @ v + 1e-12)
        v = b / (K.T @ u + 1e-12)

    T = torch.diag(u) @ K @ torch.diag(v)
    return T


class CharmsModel(nn.Module):
    def __init__(self, cfg: CharmsConfig):
        super().__init__()
        self.cfg = cfg

        self.backbone = timm.create_model(
            cfg.backbone_name,
            pretrained=cfg.pretrained,
            num_classes=0,
            global_pool=""
        )
        self.feature_info = self.backbone.feature_info
        num_chs = self.feature_info.channels()[-1]

        # Reduce image channels C -> C'
        self.channel_reducer = nn.Conv2d(num_chs, cfg.reduced_channels, kernel_size=1, bias=False)
        self.channel_norm = nn.BatchNorm2d(cfg.reduced_channels)

        self.tab_encoder = TabularAttributeEncoder(
            num_numeric_attrs=cfg.num_numeric_attrs,
            cat_cardinalities=cfg.cat_cardinalities,
            embed_dim=cfg.attr_embed_dim
        )

        # Main biomass head
        self.biomass_head = nn.Sequential(
            nn.Linear(cfg.reduced_channels, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, cfg.num_targets),
        )

        # Auxiliary heads for tabular prediction from aligned channels
        D = cfg.num_numeric_attrs + len(cfg.cat_cardinalities)
        self.num_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cfg.reduced_channels, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )
            for _ in range(cfg.num_numeric_attrs)
        ])
        self.cat_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cfg.reduced_channels, 64),
                nn.GELU(),
                nn.Linear(64, card),
            )
            for card in cfg.cat_cardinalities
        ])

        # Optional learnable projection for alignment space
        self.attr_proj = nn.Linear(cfg.attr_embed_dim, cfg.reduced_channels)

    def extract_channels(self, x: torch.Tensor) -> torch.Tensor:
        """
        returns image channels/features [B, C']
        """
        feat = self.backbone.forward_features(x)        # [B, C, H, W]
        feat = self.channel_norm(self.channel_reducer(feat))
        ch = feat.mean(dim=(2, 3))                      # GAP -> [B, C']
        return ch

    def compute_ot_alignment(
        self,
        image_channels: torch.Tensor,   # [B, C']
        attr_tokens: torch.Tensor       # [B, D, E]
    ) -> torch.Tensor:
        """
        Build cost matrix by comparing sample-similarity structures of
        each reduced image channel and each tabular attribute.

        Returns T: [D, C']
        """
        B, C = image_channels.shape
        _, D, _ = attr_tokens.shape

        # Per-channel image "feature": scalar value across batch -> similarity structure
        img_sims = []
        for c in range(C):
            x_c = image_channels[:, c:c+1]                # [B, 1]
            sim = pairwise_cosine_sim(x_c)                # [B, B]
            img_sims.append(sim)
        img_sims = torch.stack(img_sims, dim=0)           # [C, B, B]

        # Per-attribute similarity
        attr_sims = []
        for d in range(D):
            a_d = attr_tokens[:, d, :]                    # [B, E]
            sim = pairwise_cosine_sim(a_d)                # [B, B]
            attr_sims.append(sim)
        attr_sims = torch.stack(attr_sims, dim=0)         # [D, B, B]

        # Cost[d, c] = ||S_I^c - S_T^d||^2
        cost = ((attr_sims[:, None, :, :] - img_sims[None, :, :, :]) ** 2).mean(dim=(2, 3))  # [D, C]
        T = sinkhorn_uniform(cost, epsilon=self.cfg.ot_epsilon, n_iters=self.cfg.sinkhorn_iters)
        return T

    def forward(
        self,
        images: torch.Tensor,
        x_num: Optional[torch.Tensor] = None,
        x_cat: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        ch = self.extract_channels(images)                # [B, C']
        y_pred = self.biomass_head(ch)                    # [B, 5]

        out = {
            "y_pred": y_pred,
            "channels": ch,
        }

        if (x_num is not None) and (x_cat is not None):
            attr_tokens = self.tab_encoder(x_num, x_cat)  # [B, D, E]
            T = self.compute_ot_alignment(ch, attr_tokens)  # [D, C']

            # aligned_features[d] = T[d] weighted sum over image channels
            # produce one aligned image-derived feature vector per attribute
            # here we gate channels and reuse full channel vector
            D = T.shape[0]
            aligned = []
            for d in range(D):
                gated = ch * T[d].unsqueeze(0)            # [B, C']
                aligned.append(gated)
            out["aligned"] = aligned
            out["transport"] = T

            # Aux predictions
            num_preds, cat_logits = [], []
            for i in range(self.cfg.num_numeric_attrs):
                num_preds.append(self.num_heads[i](aligned[i]).squeeze(-1))
            for j in range(len(self.cfg.cat_cardinalities)):
                idx = self.cfg.num_numeric_attrs + j
                cat_logits.append(self.cat_heads[j](aligned[idx]))

            out["num_preds"] = num_preds
            out["cat_logits"] = cat_logits

        return out