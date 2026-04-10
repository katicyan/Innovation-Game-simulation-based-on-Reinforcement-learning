import numpy as np

from market_core import (
    apply_bankruptcy_reset,
    apply_tech_progress,
    codetocost,
    run_market_session,
    update_capital_innovation,
)


def demand(q: float) -> float:
    return 100.0 - q


def test_codetocost_maps_indices():
    c = np.array([30.0, 20.0, 10.0])
    s = np.array([0, 2, 1])
    out = codetocost(s, c)
    np.testing.assert_allclose(out, np.array([30.0, 10.0, 20.0]))


def test_apply_tech_progress_is_seed_deterministic():
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)

    s = np.array([0, 1])
    i = np.array([10.0, 20.0])

    out1 = apply_tech_progress(s, i, num_cost_levels=3, rng=rng1)
    out2 = apply_tech_progress(s, i, num_cost_levels=3, rng=rng2)

    for a, b in zip(out1, out2):
        np.testing.assert_allclose(a, b)


def test_market_session_returns_finite_cash():
    k = np.array([100.0, 100.0])
    sc = np.array([20.0, 10.0])
    expansion = np.array([30.0, 40.0])

    cash, q, e_eff, constrained, price, total_q = run_market_session(k, sc, demand, expansion)

    assert cash.shape == (2,)
    assert q.shape == (2,)
    assert e_eff.shape == (2,)
    assert constrained.shape == (2,)
    assert np.isfinite(price)
    assert np.isfinite(total_q)


def test_update_capital_innovation_shapes():
    k = np.array([100.0, 100.0])
    i = np.array([0.0, 0.0])
    s = np.array([0, 1])
    c = np.array([20.0, 10.0, 5.0])
    e_eff = np.array([20.0, 10.0])
    cash = np.array([30.0, 40.0])

    k_new, i_new = update_capital_innovation(k, i, s, c, e_eff, cash)
    assert k_new.shape == (2,)
    assert i_new.shape == (2,)


def test_apply_bankruptcy_reset_resets_negative_capital():
    k = np.array([-1.0, 10.0])
    i = np.array([5.0, 2.0])
    s = np.array([2, 1])

    k_new, i_new, s_new = apply_bankruptcy_reset(k, i, s)

    np.testing.assert_allclose(k_new, np.array([0.0, 10.0]))
    np.testing.assert_allclose(i_new, np.array([0.0, 2.0]))
    np.testing.assert_allclose(s_new, np.array([0, 1]))
