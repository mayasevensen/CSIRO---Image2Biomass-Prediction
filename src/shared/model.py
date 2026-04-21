"""
Model architecture for the CEMS pipeline.

image → DINOv2 (frozen, 384-d) → Encoder MLP → 32-d latent → Head MLP → 5 targets

Encoder and Head are separate modules so CEMS can be inserted between them.
BiomassModel.encode(x) returns the 32-d latent without running the head.
BiomassModel.forward_cems(args, x, y_scaled) runs the CEMS augmentation path:
  encoder(x).detach() → get_batch_cems → head → (pred_aug, y_aug_scaled)
The real (non-augmented) path uses forward(x) as before; both paths can run in
the same training step so the encoder receives gradients from both.
"""

import torch
import torch.nn as nn

from methods.cems.cems import get_batch_cems


class Encoder(nn.Module):
    """384 → 128 → 32  (two hidden layers, GELU, dropout)."""

    def __init__(self, input_dim: int = 384, latent_dim: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Head(nn.Module):
    """32 → 32 → 5  (one hidden layer, GELU)."""

    def __init__(self, latent_dim: int = 32, output_dim: int = 5,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class BiomassModel(nn.Module):
    """
    Wraps Encoder + Head.
    The CEMS augmentation hook lives between encode() and the Head forward;
    in Part 2 it is a no-op (identity). Part 3 will replace the hook body.
    """

    def __init__(self, input_dim: int = 384, latent_dim: int = 32,
                 output_dim: int = 5, dropout: float = 0.1):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim, dropout)
        self.head = Head(latent_dim, output_dim, dropout)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 32-d latent representation (no head)."""
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))

    def forward_cems(
        self,
        args,
        x: torch.Tensor,
        y_scaled: torch.Tensor,
    ):
        """CEMS augmentation path (mirrors reference forward_mixup).

        Encodes x, detaches the latent so SVD/ridge-solve do not propagate
        gradients back through the encoder, runs get_batch_cems in latent
        space, then passes the augmented latent through the head.

        The real (non-augmented) path is forward(x); call both in the same
        training step so the encoder receives gradients from both paths.

        Args:
            args:      CEMS hyperparameter namespace (sigma, cems_method, id,
                       use_hessian).
            x:         DINOv2 features (b, 384).
            y_scaled:  MinMaxScaled targets (b, n_targets).

        Returns:
            (pred_aug, y_aug_scaled): augmented predictions and scaled labels,
            both shape (b, n_targets).
        """
        latent = self.encode(x)                   # (b, 32) — grad flows for real path
        latent_aug, y_aug_scaled = get_batch_cems(
            args,
            latent.detach(),    # SVD/ridge not meant to backprop through encoder
            y_scaled,
            latent=True,
        )
        pred_aug = self.head(latent_aug)
        return pred_aug, y_aug_scaled
