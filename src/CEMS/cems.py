"""
CEMS — Curvature Enhanced Manifold Sampling (ICML 2025).
Port of the reference implementation (CEMS-main/src/cems.py).

Each public/private function references the corresponding step in Algorithm 1
and the source function in the reference codebase.

Key addition over the reference: `args.use_hessian` flag.
  True  → full CEMS (second-order curvature correction).
  False → CEMS-L ablation (first-order / linear-only, no Hessian term).

All tensor operations are in PyTorch; inputs and outputs live on the same device
as the incoming tensors.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Per-batch intrinsic dimension — lightweight TwoNN for latent mode
# ---------------------------------------------------------------------------

def _twonn_batch(zi: torch.Tensor) -> int:
    """TwoNN ID estimate on a small batch (b, m). Returns rounded int.

    Used for per-batch d re-estimation when latent=True (Algorithm 1, step 2).
    Matches the reference intrinsic_dimension() call inside get_batch_cems.
    """
    with torch.no_grad():
        dists = torch.cdist(zi.float(), zi.float())   # (b, b)
        dists.fill_diagonal_(float('inf'))
        top2 = dists.topk(2, dim=1, largest=False).values  # (b, 2)
        r1, r2 = top2[:, 0], top2[:, 1]
        mask = r1 > 0
        if mask.sum() < 3:
            return 2   # fallback for degenerate batches
        mu = (r2 / r1)[mask]
        mu_sorted, _ = mu.sort()
        n = len(mu_sorted)
        ecdf = torch.arange(1, n + 1, dtype=torch.float32, device=zi.device) / n
        log_mu = torch.log(mu_sorted)
        log_surv = -torch.log1p(-ecdf + 1e-10)
        denom = (log_mu * log_mu).sum()
        if denom.abs() < 1e-12:
            return 2   # fallback for degenerate distributions
        d = float((log_mu * log_surv).sum() / denom)
    if not math.isfinite(d) or d < 1:
        return 2
    return max(1, int(round(d)))


# ---------------------------------------------------------------------------
# Step 1 — _adjust_dims  (port of reference _adjust_dims)
# ---------------------------------------------------------------------------

def _adjust_dims(
        x: torch.Tensor,
        y: torch.Tensor,
        xk: Optional[torch.Tensor] = None,
        yk: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], int]:
    """Flatten x, ensure y is 2-D, concatenate into joint space zi = [x | y].

    Step 1 of Algorithm 1 — port of _adjust_dims from reference cems.py.
    Handles multi-target y (b, n_targets) as well as single-target (b,) / (b, 1).
    Returns (x_flat, zi, zk, m) where m = number of feature dims.
    """
    if x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    if y.ndim == 1:
        y = y.reshape(y.shape[0], 1)

    m = x.shape[-1]
    zi = torch.cat((x, y), dim=-1)   # (b, m + n_targets)

    zk: Optional[torch.Tensor] = None
    if xk is not None and yk is not None:
        if xk.ndim > 3:
            xk = xk.reshape(xk.shape[0], xk.shape[1], -1)
        zk = torch.cat((xk, yk), dim=-1)

    return x, zi, zk, m


# ---------------------------------------------------------------------------
# Step 3 — _get_projection  (port of reference _get_projection)
# ---------------------------------------------------------------------------

def _get_projection(
        args,
        x: torch.Tensor,                   # zi, shape (b, D)
        xk: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """SVD-based local orthonormal basis and projected coordinates.

    Step 3 of Algorithm 1 — port of _get_projection from reference cems.py.

    Method 1 (cems_method=1, batch-centred SVD):
      - Subtract batch mean from the transposed matrix x_c.
      - SVD on the double-centred matrix (matches reference exactly, including
        the double-centring quirk in the SVD call).
      - Project x_c (singly-centred) onto the basis.
      - Build pairwise displacement vectors u[i] - u[j] for all i != j.

    Returns (basis, u, u_prev, x_c_mean).
      basis  : (D, K) left singular vectors
      u      : (b, K, b-1) pairwise displacements in projected space
      u_prev : (b, K)      each point's own projected coordinates
      x_c_mean : (D,) or None
    """
    x_c = x.transpose(-2, -1)          # (D, b)
    x_c_mean: Optional[torch.Tensor] = None

    if args.cems_method == 1:
        x_c_mean = torch.mean(x_c, -1)              # (D,)
        x_c = x_c - x_c_mean.unsqueeze(-1)          # singly-centred (D, b)
    else:
        assert xk is not None, "`xk` required when cems_method != 1"
        xk_t = xk.transpose(-1, -2)
        x = x.unsqueeze(-1)
        x_c = xk_t - x

    # SVD — double-centring for method 1 matches reference exactly.
    # driver="gesvd" is CUDA-only; omit on CPU/MPS.
    svd_input = (x_c - x_c_mean.unsqueeze(-1)
                 if x_c_mean is not None else x_c)
    svd_kwargs = {}
    if svd_input.is_cuda:
        svd_kwargs["driver"] = "gesvd"
    basis, _, _ = torch.linalg.svd(
        svd_input, full_matrices=False, **svd_kwargs
    )   # basis: (D, K), K = min(D, b)

    u = basis.transpose(-2, -1) @ x_c   # (K, b)
    u_prev = u.transpose(-2, -1)         # (b, K)

    if args.cems_method == 1:
        # Pairwise differences: u_t[i] - u_t[j] for all i, j
        u_t = u.transpose(-1, -2)                        # (b, K)
        u = (u_t.unsqueeze(1) - u_t).transpose(-1, -2)  # (b, K, b)
        n = x.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool, device=x.device)
        u = -u.transpose(-1, -2)[mask].reshape(
            (u.shape[0], u.shape[2] - 1, u.shape[1])
        ).transpose(-1, -2)   # (b, K, b-1)
    elif args.cems_method == 2:
        u = u.unsqueeze(0)

    return basis, u, u_prev, x_c_mean


# ---------------------------------------------------------------------------
# Steps 3–4 — _estimate_grad_hessian  (port of reference _estimate_grad_hessian)
# ---------------------------------------------------------------------------

def _estimate_grad_hessian(
        args,
        x: torch.Tensor,
        xk: Optional[torch.Tensor],
        d: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           Optional[torch.Tensor]]:
    """Local gradient and Hessian via ridge regression (Eq. 6 of the paper).

    Steps 3–4 of Algorithm 1 — port of _estimate_grad_hessian from reference cems.py.

    When args.use_hessian=False (CEMS-L ablation):
      - psi uses only linear terms (d columns instead of d + d*(d+1)/2).
      - Hessian is returned as all-zeros.
    """
    tidx = torch.triu_indices(d, d, device=x.device)
    ones_mult = torch.ones((d, d), device=x.device)
    ones_mult.fill_diagonal_(0.5)

    basis, u, u_prev, x_mean = _get_projection(args, x, xk)

    u_d = u[:, :d]                        # (b, d, b-1)
    f = u[:, d:].transpose(-2, -1)        # (b, b-1, n_normal), n_normal = K - d

    if args.use_hessian:
        # Quadratic design matrix: linear terms + upper-triangle products
        uu = torch.einsum(
            'bki,bkj->bkij',
            u_d.transpose(-2, -1),   # (b, b-1, d)
            u_d.transpose(-2, -1),   # (b, b-1, d)
        )                                          # (b, b-1, d, d)
        uu = uu * ones_mult                        # scale diagonal by 0.5
        uu = uu[:, :, tidx[0], tidx[1]].transpose(-2, -1)   # (b, n_quad, b-1)
        psi = torch.cat((u_d, uu), dim=1).transpose(-2, -1)  # (b, b-1, d+n_quad)
    else:
        # CEMS-L: linear-only design matrix
        psi = u_d.transpose(-2, -1)   # (b, b-1, d)

    lam = torch.linalg.norm(psi, dim=(-1, -2)).mean()
    b_coef = _solve_ridge(psi, f, lam=lam).transpose(-2, -1)  # (b, n_normal, d or d+n_quad)

    gradient = b_coef[..., :d]   # (b, n_normal, d)

    hessian = torch.zeros(
        (u.shape[0], b_coef.shape[1], d, d),
        dtype=b_coef.dtype, device=b_coef.device,
    )
    if args.use_hessian:
        hessian[..., tidx[0], tidx[1]] = b_coef[..., d:]
        hessian[..., tidx[1], tidx[0]] = b_coef[..., d:]   # symmetrise

    return basis, gradient, hessian, u_d, u_prev, x_mean


# ---------------------------------------------------------------------------
# Step 5 — _sample_tangent  (port of reference _sample_tangent)
# ---------------------------------------------------------------------------

def _sample_tangent(
        args,
        x: torch.Tensor,              # zi, (b, D)
        u_k_d: torch.Tensor,
        u_prev: torch.Tensor,         # (b, K)
        x_mean: Optional[torch.Tensor],
        basis: torch.Tensor,          # (D, K)
        grad: torch.Tensor,           # (b, n_normal, d)
        hess: torch.Tensor,           # (b, n_normal, d, d)
) -> torch.Tensor:
    """Sample in the tangent bundle and project back to ambient space.

    Step 5 of Algorithm 1 — port of _sample_tangent from reference cems.py.
    When args.use_hessian=False, the 0.5 * ν^T H ν term is skipped (CEMS-L).
    """
    d = grad.shape[-1]
    nu = torch.distributions.Normal(0, args.sigma).sample(
        (x.shape[0], d, 1)
    ).to(x.device)   # (b, d, 1)

    # First-order term: gradient ∘ ν
    f_nu = (grad @ nu).squeeze(-1)   # (b, n_normal)

    # Second-order term: 0.5 * ν^T H ν  (curvature correction — CEMS vs CEMS-L)
    if args.use_hessian:
        nu_ex = nu.unsqueeze(1)   # (b, 1, d, 1)
        f_nu = f_nu + 0.5 * (
            nu_ex.transpose(-1, -2) @ hess @ nu_ex
        ).squeeze((-1, -2))   # (b, n_normal)

    x_zero = nu.squeeze(-1)                              # (b, d)
    x_new_local = torch.cat((x_zero, f_nu), dim=-1)     # (b, K)

    if args.cems_method == 1:
        x_new_local = x_new_local + u_prev              # add back anchor's local coords

    x_cems = (basis @ x_new_local.unsqueeze(-1)).squeeze(-1)   # (b, D)
    x_cems = x_cems + (x_mean if args.cems_method == 1 else x)

    return x_cems   # (b, D)


# ---------------------------------------------------------------------------
# Ridge regression solver
# ---------------------------------------------------------------------------

def _solve_ridge(
        a: torch.Tensor,
        b: torch.Tensor,
        lam: float = 1.0,
) -> torch.Tensor:
    """Solve (A^T A + λI) X = A^T B — port of _solve_ridge_regression from reference."""
    n = a.shape[-1]
    eye = torch.eye(n, device=a.device, dtype=a.dtype)
    a_t = a.transpose(-2, -1)
    a_reg = a_t @ a + lam * eye
    return torch.linalg.inv(a_reg) @ a_t @ b


# ---------------------------------------------------------------------------
# Public entry point — get_batch_cems  (port of reference get_batch_cems)
# ---------------------------------------------------------------------------

def get_batch_cems(
        args,
        x: torch.Tensor,
        y: torch.Tensor,
        xk: Optional[torch.Tensor] = None,
        yk: Optional[torch.Tensor] = None,
        *,
        latent: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CEMS augmentation for one minibatch.

    Port of get_batch_cems from reference cems.py. Returns (x_new, y_new)
    with the same shapes as the inputs.

    Args:
        args: namespace with at minimum: sigma, cems_method, id, use_hessian.
        x:    features or latent (b, m).
        y:    targets (b, n_targets) or (b, 1) or (b,).
        xk, yk: neighbour tensors for method 0 (None for method 1).
        latent: if True, re-estimate d per batch via TwoNN (default True).
                if False, use args.id directly (useful for testing).
    """
    x_shape, y_shape = x.shape, y.shape

    x_flat, zi, zk, m = _adjust_dims(x, y, xk, yk)

    d = args.id
    if latent:
        with torch.no_grad():
            d = _twonn_batch(zi)

    # Clamp for numerical stability (matches reference)
    d = min(d, zi.shape[-1] - 1, zi.shape[-2] - 1)
    d = max(d, 1)

    basis, grad, hess, u_k_d, u_prev, x_mean = _estimate_grad_hessian(
        args, zi, zk, d
    )
    z_sampled = _sample_tangent(args, zi, u_k_d, u_prev, x_mean, basis, grad, hess)

    x_new = z_sampled[..., :m].reshape(x_shape)
    y_new = z_sampled[..., m:].reshape(y_shape)

    return x_new, y_new
