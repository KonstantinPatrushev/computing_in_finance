"""Round-trip and constraint tests for :mod:`cif.qubo.encoding`."""

from __future__ import annotations

import numpy as np
import pytest

from cif.qubo.encoding import bits_to_weights, make_encoding, weights_to_bits


@pytest.mark.parametrize("kind,n_levels", [("unit_ticks", 7), ("unit_ticks", 15), ("one_hot", 5)])
def test_level_round_trip(kind, n_levels):
    """All level values round-trip exactly through encode → decode."""
    enc = make_encoding(n_assets=4, n_levels=n_levels, kind=kind)
    for k in range(n_levels + 1):
        for asset in range(4):
            w = np.zeros(4)
            w[asset] = enc.level_value(k)
            bits = weights_to_bits(w, enc)
            decoded = bits_to_weights(bits, enc)
            assert np.allclose(w, decoded), (
                f"Level {k} asset {asset} kind={kind}: encoded {bits}, decoded {decoded}"
            )


def test_one_hot_exactly_one_active_bit_after_encode():
    enc = make_encoding(n_assets=3, n_levels=5, kind="one_hot")
    weights = np.array([0.4, 0.4, 0.2])
    bits = weights_to_bits(weights, enc)
    for i in range(3):
        chunk = bits[enc.asset_bit_slice(i)]
        assert int(chunk.sum()) == 1, f"asset {i} chunk {chunk} not one-hot"


def test_unit_ticks_compositions_round_trip():
    """For integer compositions of L into N, unit_ticks is lossless."""
    rng = np.random.default_rng(0)
    enc = make_encoding(n_assets=5, n_levels=10, kind="unit_ticks")
    for _ in range(20):
        k = rng.integers(0, 11, size=5)
        w = k / 10.0
        bits = weights_to_bits(w, enc)
        decoded = bits_to_weights(bits, enc)
        assert np.allclose(w, decoded), f"compositions {k}: round-trip mismatch"


def test_invalid_kind_raises():
    with pytest.raises(ValueError, match="Unknown encoding"):
        make_encoding(n_assets=3, n_levels=4, kind="invalid")  # type: ignore[arg-type]


def test_one_hot_decode_picks_highest_active():
    """When multiple bits are accidentally active, the highest index wins."""
    enc = make_encoding(n_assets=2, n_levels=3, kind="one_hot")
    bits = np.array([0, 1, 0, 1,  # asset 0: levels 1 and 3 active → pick 3
                     1, 0, 0, 0])  # asset 1: level 0
    decoded = bits_to_weights(bits, enc)
    assert decoded[0] == enc.level_value(3)
    assert decoded[1] == 0.0
