"""Binary encoding of continuous portfolio weights.

Two encoding families are supported. Both map a weight vector
``w ∈ [w_min, w_max]^N`` to a fixed-length binary string ``b ∈ {0,1}^M`` and
back. They differ in the trade-off between bit count and constraint
structure:

* **Unit-ticks encoding (one-hot / integer counts)**: each asset ``i``
  is represented by ``L+1`` levels ``{0, 1, …, L}`` and consumes
  ``⌈log₂(L+1)⌉`` bits. The integer ``k_i`` is interpreted as
  ``k_i / L · budget``. Naturally grid-aligned with the brute-force ground
  truth in :mod:`cif.classical.brute_force`, which makes approximation
  ratios directly comparable.
* **Fixed-point binary encoding**: each asset has ``B`` bits and each weight
  is ``w_i = w_min + (w_max − w_min) · (Σ_k 2^k b_{i,k}) / (2^B − 1)``.
  More compact than one-hot for fine granularity; ``B = 3`` gives 8
  possible levels per asset in only 3 bits.

The default throughout the project is the **unit-ticks** encoding because it
aligns with the brute-force enumerator and because the resulting QUBO has a
simple quadratic budget-constraint penalty. Switch to fixed-point only when
bit-count matters (i.e. for QAOA on real hardware).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


EncodingKind = Literal["unit_ticks", "fixed_point", "one_hot"]


@dataclass(frozen=True)
class WeightEncoding:
    """Description of how N weights map to M binary variables."""

    n_assets: int
    kind: EncodingKind
    bits_per_asset: int
    n_levels: int
    budget: float
    w_min: float
    w_max: float

    @property
    def n_bits(self) -> int:
        return self.n_assets * self.bits_per_asset

    def level_value(self, k: int) -> float:
        """Return the weight value corresponding to the ``k``-th level."""
        if self.kind == "unit_ticks":
            return k * self.budget / self.n_levels
        if self.kind == "one_hot":
            return k * self.budget / self.n_levels
        max_code = (1 << self.bits_per_asset) - 1
        return self.w_min + (self.w_max - self.w_min) * k / max_code

    def asset_bit_slice(self, i: int) -> slice:
        return slice(i * self.bits_per_asset, (i + 1) * self.bits_per_asset)


def make_encoding(
    n_assets: int,
    n_levels: int | None = None,
    bits_per_asset: int | None = None,
    kind: EncodingKind = "unit_ticks",
    budget: float = 1.0,
    w_min: float = 0.0,
    w_max: float = 1.0,
) -> WeightEncoding:
    """Build a :class:`WeightEncoding` from user-level parameters.

    Either ``n_levels`` (for unit-ticks) or ``bits_per_asset`` (for
    fixed-point) must be provided.
    """
    if kind == "unit_ticks":
        if n_levels is None:
            if bits_per_asset is None:
                raise ValueError("Provide n_levels or bits_per_asset for unit_ticks")
            n_levels = (1 << bits_per_asset) - 1
        bpa = int(np.ceil(np.log2(n_levels + 1))) if n_levels > 0 else 1
        return WeightEncoding(
            n_assets=n_assets,
            kind=kind,
            bits_per_asset=bpa,
            n_levels=int(n_levels),
            budget=float(budget),
            w_min=float(w_min),
            w_max=float(w_max),
        )
    if kind == "fixed_point":
        if bits_per_asset is None:
            raise ValueError("Provide bits_per_asset for fixed_point encoding")
        return WeightEncoding(
            n_assets=n_assets,
            kind=kind,
            bits_per_asset=int(bits_per_asset),
            n_levels=(1 << bits_per_asset) - 1,
            budget=float(budget),
            w_min=float(w_min),
            w_max=float(w_max),
        )
    if kind == "one_hot":
        if n_levels is None:
            raise ValueError("Provide n_levels for one_hot encoding")
        # one-hot: one bit per level, exactly one bit is 1 per asset
        return WeightEncoding(
            n_assets=n_assets,
            kind=kind,
            bits_per_asset=int(n_levels) + 1,
            n_levels=int(n_levels),
            budget=float(budget),
            w_min=float(w_min),
            w_max=float(w_max),
        )
    raise ValueError(f"Unknown encoding kind: {kind!r}")


def bits_to_weights(bits: np.ndarray, encoding: WeightEncoding) -> np.ndarray:
    """Decode a bit vector into weights. Supports all three encodings.

    For one-hot, the convention when multiple bits are active is "choose the
    highest-index active bit" so that noisy samples degrade gracefully toward
    the larger weight choice. Zero active bits → weight 0.
    """
    bits = np.asarray(bits, dtype=int).reshape(-1)
    if bits.shape[0] != encoding.n_bits:
        raise ValueError(
            f"Expected {encoding.n_bits} bits, got {bits.shape[0]}"
        )
    weights = np.empty(encoding.n_assets, dtype=float)
    for i in range(encoding.n_assets):
        chunk = bits[encoding.asset_bit_slice(i)]
        if encoding.kind == "one_hot":
            active = np.flatnonzero(chunk)
            code = int(active.max()) if active.size else 0
        else:
            code = int(sum(int(bit) * (1 << k) for k, bit in enumerate(chunk)))
            if encoding.kind == "unit_ticks":
                code = min(code, encoding.n_levels)
        weights[i] = encoding.level_value(code)
    return weights


def weights_to_bits(weights: np.ndarray, encoding: WeightEncoding) -> np.ndarray:
    """Encode a weight vector into a bit string.

    Quantisation is by nearest level. This is a lossy operation for
    arbitrary continuous weights and primarily exists for roundtrip tests and
    seeding of local-search solvers with good initial configurations.
    """
    weights = np.asarray(weights, dtype=float)
    if weights.shape[0] != encoding.n_assets:
        raise ValueError(f"Expected {encoding.n_assets} weights, got {weights.shape[0]}")
    bits = np.zeros(encoding.n_bits, dtype=int)
    if encoding.kind == "one_hot":
        for i, w in enumerate(weights):
            code = int(round(w / encoding.budget * encoding.n_levels))
            code = max(0, min(encoding.n_levels, code))
            chunk = bits[encoding.asset_bit_slice(i)]
            chunk[code] = 1
        return bits
    max_code = encoding.n_levels if encoding.kind == "unit_ticks" else (1 << encoding.bits_per_asset) - 1
    for i, w in enumerate(weights):
        if encoding.kind == "unit_ticks":
            code = int(round(w / encoding.budget * encoding.n_levels))
        else:
            span = encoding.w_max - encoding.w_min
            frac = 0.0 if span == 0 else (w - encoding.w_min) / span
            code = int(round(frac * max_code))
        code = max(0, min(max_code, code))
        chunk = bits[encoding.asset_bit_slice(i)]
        for k in range(encoding.bits_per_asset):
            chunk[k] = (code >> k) & 1
    return bits
