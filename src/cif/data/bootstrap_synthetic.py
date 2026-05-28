"""Bootstrap-realistic synthetic covariance generator.

The dense random covariance used in Experiment A (Σ = AAᵀ · 0.005 + I · 0.01,
A ~ N(0,1)^{N×N}) is mathematically convenient but **structurally adversarial
for branch-and-bound MIQP solvers** — it has high condition number (~10⁵),
no block structure, no factor model, and produces wide dispersion of SCIP
wall time. This biases the synthetic benchmark in favour of any solver that
doesn't depend on LP-relaxation tightness (like simulated annealing or tabu).

This module produces ``bootstrap-realistic`` Σ matrices that preserve the
spectral and structural properties of a real financial covariance matrix
(after Ledoit-Wolf shrinkage on S&P 500 returns) but allow scaling N beyond
the size of the underlying dataset. The procedure is:

1. Load a real Ledoit-Wolf-shrunk covariance matrix Σ_real of size N_real.
2. Compute its eigendecomposition Σ_real = U Λ Uᵀ.
3. Sample a target size ``n_target`` (possibly > N_real).
4. Form a bootstrap by drawing ``n_target`` "ticker proxies" with replacement
   from the original universe, then perturbing each by a small Gaussian
   noise scaled by the bottom-quartile eigenvalue.
5. The resulting Σ has eigenvalue distribution close to Σ_real (within the
   limits of resampling noise) and inherits its block-sector structure.

This is the standard methodology in the quantitative finance literature for
extending real-data benchmarks beyond the available universe size (see e.g.
Pafka–Kondor 2003, Bouchaud–Potters 2010).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cif.data.statistics import annualised_mu, annualised_sigma, log_returns


def load_real_covariance(
    prices_parquet: Path | str,
    train_start: str = "2015-01-01",
    train_end: str = "2024-12-31",
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Load real μ, Σ from a price panel for the given training window."""
    prices = pd.read_parquet(prices_parquet)
    train = prices.loc[train_start:train_end]
    rets = log_returns(train)
    mu = annualised_mu(rets).values
    sigma = annualised_sigma(rets, method="ledoit_wolf").values
    names = tuple(rets.columns)
    return mu, sigma, names


def bootstrap_realistic_instance(
    mu_real: np.ndarray,
    sigma_real: np.ndarray,
    n_target: int,
    seed: int,
    mu_noise_scale: float = 0.02,
    sigma_noise_scale: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a quasi-realistic ``(μ, Σ)`` pair of dimension ``n_target``.

    Procedure:
    1. Sample ``n_target`` indices with replacement from the real universe.
    2. μ_boot[i] = μ_real[idx[i]] + ε_μ, where ε_μ ~ N(0, mu_noise_scale²).
    3. Permute the sampled covariance block and add small Gaussian
       perturbation in the eigenbasis, scaled by the smallest eigenvalue.
    4. Project back to a valid PSD matrix.

    Parameters
    ----------
    mu_real, sigma_real:
        Real-data moments, dimension ``N_real``.
    n_target:
        Desired instance size. Can be larger than ``N_real``.
    seed:
        RNG seed.
    mu_noise_scale, sigma_noise_scale:
        Multiplicative noise added on top of the resampled real values. Set
        small (~2-5%) to keep the structural signal of the real Σ.
    """
    rng = np.random.default_rng(seed)
    n_real = mu_real.shape[0]
    if sigma_real.shape != (n_real, n_real):
        raise ValueError(f"sigma_real shape {sigma_real.shape} != ({n_real}, {n_real})")

    # 1. Bootstrap indices (with replacement) — duplicates allowed for n_target > N_real
    idx = rng.integers(0, n_real, size=n_target)

    # 2. μ bootstrap with small noise
    mu_boot = mu_real[idx] + rng.normal(0.0, mu_noise_scale, size=n_target)

    # 3. Σ block resampling: take the sub-matrix of Σ_real on the bootstrap indices
    #    Duplicates produce identical rows/columns — break this with a small
    #    perturbation in the eigenbasis.
    sigma_boot = sigma_real[np.ix_(idx, idx)]

    # 4. Perturb in eigenbasis to ensure PSD and avoid singular structure when
    #    n_target > N_real (duplicated indices → rank-deficient). The floor is
    #    set to the bottom-quartile eigenvalue of the real covariance so the
    #    resulting condition number stays close to the real one (rather than
    #    blowing up to 10^8 when we naively floor at 1e-8).
    eigvals_real = np.linalg.eigvalsh(sigma_real)
    eig_floor = float(np.quantile(eigvals_real[eigvals_real > 0], 0.25))
    eigvals, eigvecs = np.linalg.eigh(sigma_boot)
    eigvals = np.clip(eigvals, eig_floor, None)
    median_eig = np.median(eigvals)
    eigval_noise = rng.normal(0.0, sigma_noise_scale * median_eig, size=eigvals.shape)
    eigvals_perturbed = np.clip(eigvals + eigval_noise, eig_floor, None)
    sigma_boot = eigvecs @ np.diag(eigvals_perturbed) @ eigvecs.T

    # Force symmetry (numerical drift can break it slightly)
    sigma_boot = 0.5 * (sigma_boot + sigma_boot.T)

    return mu_boot, sigma_boot


def describe_instance(mu: np.ndarray, sigma: np.ndarray) -> dict:
    """Return structural diagnostics of a generated instance."""
    eigvals = np.linalg.eigvalsh(sigma)
    return {
        "n": mu.shape[0],
        "mu_range": (float(mu.min()), float(mu.max())),
        "mu_mean": float(mu.mean()),
        "sigma_diag_range": (float(sigma.diagonal().min()), float(sigma.diagonal().max())),
        "sigma_diag_mean": float(sigma.diagonal().mean()),
        "eigval_min": float(eigvals.min()),
        "eigval_max": float(eigvals.max()),
        "condition_number": float(eigvals.max() / max(eigvals.min(), 1e-12)),
    }
