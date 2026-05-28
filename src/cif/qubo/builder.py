"""Build a :class:`dimod.BinaryQuadraticModel` from a :class:`PortfolioProblem`.

Only the **one-hot per asset** encoding is supported in the builder because
it gives a clean, explicit QUBO:

* Variables ``x_{i,k} ∈ {0, 1}`` for every asset ``i`` and discrete level
  ``k ∈ {0, …, L}``. In the feasible subspace, exactly one ``x_{i,k}`` is
  active per asset, and the continuous weight is ``w_i = (k/L)·budget``.
* One-hot feasibility is enforced by a quadratic penalty
  ``λ_onehot · (Σ_k x_{i,k} − 1)²``.
* Budget by ``λ_budget · (Σ_i w_i − budget)²``.
* Cardinality by ``λ_card · (|support(w)| − K)²`` where the support is
  ``|{i : x_{i,0}=0}| = N − Σ_i x_{i,0}``. This is linear in the ``x_{i,0}``
  variables, so the induced penalty is a clean quadratic in binary form.

The builder does not try to be cleverly sparse — construction runs in
``O((NL)²)`` Python calls, which is fine for ``N ≤ 100`` and ``L ≤ 10``.
"""

from __future__ import annotations

from dataclasses import dataclass

import dimod
import numpy as np

from cif.problem import PortfolioProblem
from cif.qubo.encoding import WeightEncoding


@dataclass(frozen=True)
class Penalties:
    """Penalty coefficients for the QUBO soft constraints."""

    onehot: float
    budget: float
    cardinality: float = 0.0


def suggest_penalties(
    problem: PortfolioProblem,
    encoding: WeightEncoding,
    multiplier: float = 5.0,
) -> Penalties:
    """Heuristic initial penalties (Lucas 2014): multiple of ``|H_obj|`` scale.

    ``H_obj`` magnitude is estimated from ``max(|μ|)·budget + λ·max(|Σ|)·budget²``
    — an upper bound on any feasible weight assignment's objective. The
    default ``multiplier`` of 5 satisfies the "penalty ≫ objective gap"
    criterion for typical mean-variance problems.
    """
    mu_scale = float(np.max(np.abs(problem.mu))) * encoding.budget
    sigma_scale = float(np.max(np.abs(problem.sigma))) * encoding.budget ** 2
    base = max(mu_scale + problem.risk_aversion * sigma_scale, 1e-6)
    lam = multiplier * base
    return Penalties(onehot=lam, budget=lam, cardinality=lam)


def _var(i: int, k: int) -> tuple[int, int]:
    return (i, k)


def build_bqm(
    problem: PortfolioProblem,
    encoding: WeightEncoding,
    penalties: Penalties,
) -> dimod.BinaryQuadraticModel:
    """Return a fully-assembled BQM ready for ``neal`` or any QUBO sampler."""
    if encoding.kind != "one_hot":
        raise NotImplementedError(
            f"Builder only supports one_hot encoding, got {encoding.kind!r}"
        )
    if encoding.n_assets != problem.n:
        raise ValueError(
            f"encoding N={encoding.n_assets} != problem N={problem.n}"
        )

    n = problem.n
    L = encoding.n_levels
    B = encoding.budget
    lam = problem.risk_aversion
    mu = problem.mu
    sigma = problem.sigma

    bqm = dimod.BinaryQuadraticModel("BINARY")

    # ---------- Objective: -μᵀw + (λ/2) wᵀΣw ----------
    # Linear part from -μᵀw and diagonal of quadratic (since x² = x).
    scale = B / L
    for i in range(n):
        for k in range(L + 1):
            linear = -mu[i] * k * scale
            linear += 0.5 * lam * sigma[i, i] * (k * scale) ** 2
            bqm.add_linear(_var(i, k), linear)

    # Off-diagonal quadratic: i != j OR a != b.
    # w'Σw expansion: (λ/2) sum_{i,j,a,b} Σ_ij (a*scale)(b*scale) x_{i,a} x_{j,b}
    # For i==j, a==b already handled in the linear loop above.
    # BQM quadratic is symmetric: add_quadratic(u, v, c) => energy += c * u * v.
    # Since w'Σw sums both (i,j) and (j,i), the contribution to a single
    # unordered pair (i,a)-(j,b) (i != j) is 2 * (λ/2) * Σ_ij * a * b * scale²
    # = λ Σ_ij a b scale².
    for i in range(n):
        for j in range(n):
            sij = sigma[i, j]
            if sij == 0.0:
                continue
            for a in range(L + 1):
                for b in range(L + 1):
                    if i == j and a == b:
                        continue  # already in linear
                    if (i, a) >= (j, b):
                        continue  # only process each unordered pair once
                    coef = lam * sij * a * b * scale * scale
                    if coef != 0.0:
                        bqm.add_quadratic(_var(i, a), _var(j, b), coef)

    # ---------- Penalty: one-hot per asset ----------
    # λ_oh * sum_i (sum_k x_{i,k} - 1)²
    # = λ_oh * sum_i [(sum_k x_{i,k})² - 2 sum_k x_{i,k} + 1]
    # (sum_k x_{i,k})² = sum_k x_{i,k} + 2 sum_{k<l} x_{i,k}x_{i,l}   (binary)
    lam_oh = penalties.onehot
    for i in range(n):
        for k in range(L + 1):
            bqm.add_linear(_var(i, k), lam_oh * (1 - 2))  # (+1 - 2) = -1
        for k in range(L + 1):
            for l in range(k + 1, L + 1):
                bqm.add_quadratic(_var(i, k), _var(i, l), 2 * lam_oh)
        bqm.offset += lam_oh  # constant term +1 per asset

    # ---------- Penalty: budget ----------
    # λ_b * (sum_i w_i - B)²
    # = λ_b * ((B/L) sum_{i,k} k x_{i,k} - B)²
    # Let S = sum_{i,k} k x_{i,k} . Then penalty = λ_b * (B/L)² * (S - L)²
    # = λ_b (B/L)² [S² - 2 L S + L²]
    # S² = sum_{(i,k),(j,l)} k l x_{i,k} x_{j,l}
    #    = sum_{i,k} k² x_{i,k} + 2 sum_{(i,k)<(j,l)} k l x_{i,k} x_{j,l}
    lam_b = penalties.budget
    scale_b = (B / L) ** 2
    # Linear from S²'s diagonal (k² x_{i,k}) and -2LS (k x_{i,k})
    for i in range(n):
        for k in range(L + 1):
            bqm.add_linear(_var(i, k), lam_b * scale_b * (k * k - 2 * L * k))
    # Quadratic from S²'s cross terms
    keys = [(i, k) for i in range(n) for k in range(L + 1)]
    for idx1 in range(len(keys)):
        i, k = keys[idx1]
        for idx2 in range(idx1 + 1, len(keys)):
            j, l = keys[idx2]
            coef = 2 * lam_b * scale_b * k * l
            if coef != 0.0:
                bqm.add_quadratic(_var(i, k), _var(j, l), coef)
    # Constant L²
    bqm.offset += lam_b * scale_b * L * L

    # ---------- Penalty: cardinality ----------
    # λ_c * (|support| - K)² where |support| = N - sum_i x_{i,0}
    # = λ_c * ((N - K) - sum_i x_{i,0})²
    # Let Z = sum_i x_{i,0}. Penalty = λ_c ((N-K) - Z)²
    # = λ_c [Z² - 2(N-K)Z + (N-K)²]
    if problem.cardinality is not None and penalties.cardinality > 0.0:
        K = int(problem.cardinality)
        NK = n - K
        lam_c = penalties.cardinality
        for i in range(n):
            bqm.add_linear(_var(i, 0), lam_c * (1 - 2 * NK))
        for i in range(n):
            for j in range(i + 1, n):
                bqm.add_quadratic(_var(i, 0), _var(j, 0), 2 * lam_c)
        bqm.offset += lam_c * NK * NK

    return bqm


def build_selection_bqm(
    problem: PortfolioProblem,
    cardinality: int,
    cardinality_penalty: float | None = None,
) -> dimod.BinaryQuadraticModel:
    """Build the **binary-inclusion** QUBO for cardinality Markowitz.

    This is the canonical formulation in the quantum-portfolio literature:
    ``x_i ∈ {0, 1}`` indicates whether asset ``i`` is held, with the implicit
    equal weighting ``w_i = x_i / K``. The BQM energy is::

        H(x) = − (1/K) μᵀx + (λ / (2 K²)) xᵀΣx  +  λ_c (Σ_i x_i − K)²

    The feasible manifold is the set of all ``C(N, K)`` size-``K`` subsets.
    Every sample decodes to a *meaningful* (if suboptimal) portfolio, which
    makes SA convergence robust even at ``N ≥ 50``.

    The final portfolio weights returned by the pipeline are **not** the raw
    ``1/K`` equal weights. Downstream we take the subset chosen by the
    sampler and run :func:`cif.classical.continuous.solve_continuous_mvo` on
    it to recover the continuous mean-variance optimum inside that subset.
    """
    n = problem.n
    K = int(cardinality)
    if not (1 <= K <= n):
        raise ValueError(f"cardinality must be in [1, {n}], got {K}")
    mu = problem.mu
    sigma = problem.sigma
    lam = problem.risk_aversion

    # Heuristic penalty: dominate the largest possible energy contribution.
    if cardinality_penalty is None:
        obj_scale = float(np.abs(mu).max()) / max(K, 1) + lam * float(np.abs(sigma).max()) / max(K * K, 1)
        cardinality_penalty = 20.0 * max(obj_scale, 1e-6)

    bqm = dimod.BinaryQuadraticModel("BINARY")

    # Linear from -μ·(x/K) and diagonal of (λ/(2K²)) x'Σx (since x² = x)
    for i in range(n):
        linear = -mu[i] / K + (lam / (2 * K * K)) * sigma[i, i]
        bqm.add_linear(i, linear)

    # Off-diagonal quadratic: (λ / K²) Σ_ij x_i x_j for i < j  (factor of 2 absorbed)
    for i in range(n):
        for j in range(i + 1, n):
            coef = (lam / (K * K)) * sigma[i, j]
            if coef != 0.0:
                bqm.add_quadratic(i, j, coef)

    # Cardinality penalty: λ_c (Σ x - K)² = λ_c [Σ x + 2 Σ_{i<j} x_i x_j - 2K Σ x + K²]
    for i in range(n):
        bqm.add_linear(i, cardinality_penalty * (1 - 2 * K))
    for i in range(n):
        for j in range(i + 1, n):
            bqm.add_quadratic(i, j, 2 * cardinality_penalty)
    bqm.offset += cardinality_penalty * K * K

    return bqm


def selection_bits_to_subset(sample: dict | np.ndarray, n: int) -> list[int]:
    """Decode a binary-inclusion BQM sample into the chosen subset indices."""
    if isinstance(sample, dict):
        return sorted([i for i in range(n) if int(sample.get(i, 0)) == 1])
    arr = np.asarray(sample, dtype=int).reshape(-1)
    return sorted([i for i in range(n) if int(arr[i]) == 1])


def build_weighted_bqm(
    problem: PortfolioProblem,
    bits_per_asset: int = 3,
    budget_penalty: float | None = None,
    cardinality_penalty: float | None = None,
) -> dimod.BinaryQuadraticModel:
    """Build the **weighted binary encoding** QUBO for Markowitz.

    Each asset ``i`` is represented by ``B = bits_per_asset`` binary variables
    ``b_{i,0}, …, b_{i,B-1}``. The weight is encoded directly as::

        w_i = (1 / M) · Σ_k 2^k · b_{i,k}     where M = 2^B − 1

    So ``w_i ∈ {0, 1/M, 2/M, …, 1}`` — i.e. each asset can take ``M+1`` discrete
    weight levels. The budget constraint ``Σ_i w_i = 1`` becomes a soft
    quadratic penalty on the natural QUBO variables.

    Compared to the binary-inclusion formulation in :func:`build_selection_bqm`,
    this encoding **directly optimises the actual Markowitz objective**
    (not a proxy where SA assumes equal weighting). The expected effect is
    closing the residual optimisation gap from ~11–13% down to a few percent.

    Variables are keyed by ``(i, k)`` tuples — same convention as the one-hot
    builder above. Total variable count: ``N × B``.
    """
    n = problem.n
    B = bits_per_asset
    if B < 1:
        raise ValueError(f"bits_per_asset must be ≥ 1, got {B}")
    M = (1 << B) - 1  # 2^B − 1
    mu = problem.mu
    sigma = problem.sigma
    lam = problem.risk_aversion

    if budget_penalty is None:
        # Heuristic: dominate the objective scale by a large factor.
        # |H_obj| has order |μ|_∞ + λ |Σ|_∞ on w ∈ [0, 1]^N. Choose 100× to
        # ensure budget violation is always energetically worse than any gain.
        budget_penalty = 100.0 * (
            float(np.abs(mu).max()) + lam * float(np.abs(sigma).max())
        )

    bqm = dimod.BinaryQuadraticModel("BINARY")

    # Pre-compute bit coefficients: bit (i, k) contributes 2^k / M to w_i.
    bit_coef = [2 ** k / M for k in range(B)]

    # ---------- Objective: −μᵀw + (λ/2) wᵀΣw ----------
    # Linear from −μᵀw (every bit contributes −μ_i · coef_{i,k}).
    # Plus diagonal of quadratic: for binary x, x² = x, so each b_{i,k}²
    # contributes 0.5·λ·Σ_ii·coef² to the linear coefficient.
    for i in range(n):
        for k in range(B):
            ck = bit_coef[k]
            linear = -mu[i] * ck + 0.5 * lam * sigma[i, i] * ck * ck
            bqm.add_linear((i, k), linear)

    # Quadratic from (λ/2) wᵀΣw, off-diagonal in the bit space.
    # Within a single asset (i == j, k != l): coefficient λ · Σ_ii · coef_k · coef_l.
    for i in range(n):
        for k in range(B):
            for l in range(k + 1, B):
                coef = lam * sigma[i, i] * bit_coef[k] * bit_coef[l]
                if coef != 0.0:
                    bqm.add_quadratic((i, k), (i, l), coef)

    # Cross-asset (i < j), all bit pairs: coefficient λ · Σ_ij · coef_a · coef_b.
    # Factor of 2 from symmetric (i,j)/(j,i) sum already absorbed into the
    # single i<j loop below.
    for i in range(n):
        for j in range(i + 1, n):
            sij = sigma[i, j]
            if sij == 0.0:
                continue
            for a in range(B):
                for b in range(B):
                    coef = lam * sij * bit_coef[a] * bit_coef[b]
                    if coef != 0.0:
                        bqm.add_quadratic((i, a), (j, b), coef)

    # ---------- Penalty: budget ----------
    # λ_b · (Σ_i w_i − 1)² = λ_b · ((1/M) Σ_{i,k} 2^k b_{i,k} − 1)²
    # Let Z = Σ_{i,k} 2^k b_{i,k}. Penalty = (λ_b / M²) (Z − M)².
    # Expanding: (λ_b/M²) [Z² − 2MZ + M²]
    #   Z² = Σ 4^k b + 2 Σ_{(i,k)<(j,l)} 2^{k+l} b_{i,k} b_{j,l}   (since b² = b)
    #   −2MZ = −2M Σ 2^k b
    # Linear coefficient on b_{i,k}: (λ_b/M²) · (4^k − 2M · 2^k) = (λ_b/M²) · 2^k · (2^k − 2M)
    # Quadratic on b_{i,k} · b_{j,l} (unordered, distinct): (2 λ_b / M²) · 2^{k+l}
    # Constant offset: λ_b
    lam_b = budget_penalty
    inv_M2 = 1.0 / (M * M)

    for i in range(n):
        for k in range(B):
            pow_k = 1 << k  # 2^k
            bqm.add_linear((i, k), lam_b * inv_M2 * pow_k * (pow_k - 2 * M))

    bit_keys = [(i, k) for i in range(n) for k in range(B)]
    for idx1 in range(len(bit_keys)):
        i1, k1 = bit_keys[idx1]
        pow_k1 = 1 << k1
        for idx2 in range(idx1 + 1, len(bit_keys)):
            i2, k2 = bit_keys[idx2]
            pow_k2 = 1 << k2
            coef = 2.0 * lam_b * inv_M2 * pow_k1 * pow_k2
            if coef != 0.0:
                bqm.add_quadratic((i1, k1), (i2, k2), coef)

    bqm.offset += lam_b

    # ---------- Penalty: cardinality (optional, on b_{i,0} indicator) ----------
    # An asset is "held" iff at least one of its bits is 1. We approximate the
    # cardinality count by ``Σ_i max_k b_{i,k}``, which is hard to encode.
    # As a practical proxy, we use ``Σ_i (1 − (1 − b_{i,0})(1 − b_{i,1}) …)``
    # approximation via the linear sum ``Σ_i b_{i,0}`` (cheapest bit-0 indicator).
    # This is loose: an asset with weight = 2/M but b_{i,0}=0, b_{i,1}=1 isn't
    # counted. The cleaner — and what we do here — is the "any-bit indicator":
    # ``z_i = 1 − prod_k (1 − b_{i,k})``, but this isn't quadratic.
    # We use a different proxy: penalise the *total* bit count exceeding K·B.
    # Σ_{i,k} b_{i,k} ≤ K · B (in expectation, if average weight per held asset
    # is M/2 and total budget is M, then K ≈ 2). This is a soft signal at best.
    #
    # In practice for cardinality-constrained Markowitz with weighted encoding,
    # the cleanest path is post-hoc filtering of candidates by the support size,
    # done in the neal wrapper (``solve_with_neal_weighted``). We expose the
    # cardinality_penalty here as a *guidance* term, not a strict enforcer.
    if (
        problem.cardinality is not None
        and cardinality_penalty is not None
        and cardinality_penalty > 0.0
    ):
        K = int(problem.cardinality)
        lam_c = cardinality_penalty
        # Penalty: λ_c · (Σ_{i,k} b_{i,k} − K · B / 2)²
        # This nudges the average number of active bits toward K assets at
        # 50% weight saturation each, which is a reasonable prior.
        target = K * B / 2.0
        # (Σ b)² = Σ b + 2 Σ_{(i,k)<(j,l)} b b (binary)
        # −2·target·Σ b
        # +target²
        for i in range(n):
            for k in range(B):
                bqm.add_linear((i, k), lam_c * (1.0 - 2.0 * target))
        for idx1 in range(len(bit_keys)):
            for idx2 in range(idx1 + 1, len(bit_keys)):
                bqm.add_quadratic(bit_keys[idx1], bit_keys[idx2], 2.0 * lam_c)
        bqm.offset += lam_c * target * target

    return bqm


def sample_to_weights_weighted(
    sample: dict | np.ndarray,
    n_assets: int,
    bits_per_asset: int,
    renormalize: bool = True,
) -> np.ndarray:
    """Decode a weighted-binary-encoding sample into a weight vector.

    Parameters
    ----------
    sample:
        ``dimod`` sample dict keyed by ``(i, k)`` tuples, or a flat numpy
        array of length ``N × B`` ordered by ``(i, k)`` with bit index
        varying fastest.
    n_assets:
        Number of assets ``N``.
    bits_per_asset:
        Number of bits per asset ``B``.
    renormalize:
        If True, rescale the decoded weights so ``Σ w_i = 1`` exactly.
        Because the QUBO penalty enforces the budget only approximately,
        post-hoc normalisation gives a strictly feasible portfolio.
        If the raw sum is zero (degenerate sample), returns zeros.
    """
    B = bits_per_asset
    M = (1 << B) - 1
    weights = np.zeros(n_assets, dtype=float)
    if isinstance(sample, dict):
        getter = lambda i, k: int(sample.get((i, k), 0))  # noqa: E731
    else:
        arr = np.asarray(sample, dtype=int).reshape(n_assets, B)
        getter = lambda i, k: int(arr[i, k])  # noqa: E731
    for i in range(n_assets):
        code = sum(getter(i, k) * (1 << k) for k in range(B))
        weights[i] = code / M
    if renormalize:
        total = weights.sum()
        if total > 0:
            weights = weights / total
    return weights


def sample_to_weights(
    sample: dict | np.ndarray,
    encoding: WeightEncoding,
) -> np.ndarray:
    """Convert a ``dimod`` sample (dict keyed by ``(i, k)``) into weights.

    Graceful fallback: if no bit is active for an asset, weight is 0; if more
    than one is active, pick the highest-index active level (consistent with
    :func:`cif.qubo.encoding.bits_to_weights`).
    """
    if encoding.kind != "one_hot":
        raise NotImplementedError
    n = encoding.n_assets
    L = encoding.n_levels
    weights = np.zeros(n, dtype=float)
    if isinstance(sample, dict):
        getter = lambda i, k: int(sample.get((i, k), 0))  # noqa: E731
    else:
        arr = np.asarray(sample, dtype=int).reshape(n, L + 1)
        getter = lambda i, k: int(arr[i, k])  # noqa: E731
    for i in range(n):
        picks = [k for k in range(L + 1) if getter(i, k) == 1]
        if not picks:
            weights[i] = 0.0
        else:
            weights[i] = encoding.level_value(max(picks))
    return weights
