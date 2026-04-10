import numpy as np

import utili


def codetocost(s: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Transform discrete technology state codes into effective marginal costs."""
    return np.asarray([c[idx] for idx in s], dtype=np.float64)


def innovation_progress_probability(innovation_stock: np.ndarray) -> np.ndarray:
    """Map innovation stock to progress probability with the existing nonlinear rule."""
    x = np.asarray(innovation_stock, dtype=np.float64)
    return 0.1 * x / (0.1 * x + 1.0)


def apply_tech_progress(
    s: np.ndarray,
    i: np.ndarray,
    num_cost_levels: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply stochastic technology progress, resetting innovation stock on progress."""
    s_new = np.asarray(s, dtype=np.int64).copy()
    i_new = np.asarray(i, dtype=np.float64).copy()

    probs = innovation_progress_probability(i_new)
    draws = rng.uniform(0.0, 1.0, size=s_new.shape[0])
    progressed = (draws < probs) & (s_new < (num_cost_levels - 1))

    s_new[progressed] += 1
    i_new[progressed] = 0.0

    return s_new, i_new, probs, progressed


def optimal_quantities(sc: np.ndarray, demand_function) -> np.ndarray:
    """Compute Cournot-optimal quantities capped by market choke quantity."""
    choke_q = utili.zp(demand_function)
    if choke_q is None:
        raise ValueError("Demand function has no zero crossing in configured interval")
    if isinstance(choke_q, tuple):
        choke_q_value = float(choke_q[0])
    else:
        choke_q_value = float(choke_q)

    cournot_q = np.asarray(utili.cournot(sc, demand_function)).flatten()
    return np.minimum(np.ones_like(cournot_q, dtype=np.float64) * choke_q_value, cournot_q)


def input_limit(sc: np.ndarray, demand_function) -> tuple[np.ndarray, np.ndarray]:
    """Return quantity and expansion upper bounds induced by current costs."""
    level = optimal_quantities(sc, demand_function)
    return level, sc * level


def run_market_session(
    k: np.ndarray,
    sc: np.ndarray,
    demand_function,
    expansion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Run one market period with expansion spending and return period outcomes."""
    k_arr = np.asarray(k, dtype=np.float64)
    sc_arr = np.asarray(sc, dtype=np.float64)
    e = np.asarray(expansion, dtype=np.float64).copy()

    if np.any(e > k_arr):
        raise ValueError("Insufficient capital")

    q_limit, e_limit = input_limit(sc_arr, demand_function)
    constrained = e > e_limit
    e_effective = np.where(constrained, e_limit, e)

    q = np.where(sc_arr > 0.0, e_effective / sc_arr, 0.0)
    q = np.where(constrained, q_limit, q)

    total_q = float(np.sum(q))
    price = float(demand_function(total_q))

    cash = price * q - e_effective
    return cash, q, e_effective, constrained, price, total_q


def update_capital_innovation(
    k: np.ndarray,
    i: np.ndarray,
    s: np.ndarray,
    c: np.ndarray,
    expansion_effective: np.ndarray,
    cash: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply post-session capital and innovation-stock updates using legacy rules."""
    k_prev = np.asarray(k, dtype=np.float64)
    i_new = np.asarray(i, dtype=np.float64).copy()
    s_arr = np.asarray(s, dtype=np.int64)
    e_arr = np.asarray(expansion_effective, dtype=np.float64)
    cash_arr = np.asarray(cash, dtype=np.float64)

    k_new = np.zeros_like(k_prev)
    frontier = len(c) - 1

    for idx in range(k_prev.shape[0]):
        if s_arr[idx] < frontier:
            i_new[idx] += -e_arr[idx] + k_prev[idx]
            k_new[idx] = cash_arr[idx]
        else:
            k_new[idx] = cash_arr[idx] + k_prev[idx] - e_arr[idx]

    return k_new, i_new


def apply_bankruptcy_reset(k: np.ndarray, i: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reset bankrupt firms to baseline state following legacy behavior."""
    k_new = np.asarray(k, dtype=np.float64).copy()
    i_new = np.asarray(i, dtype=np.float64).copy()
    s_new = np.asarray(s, dtype=np.int64).copy()

    bankrupt = k_new < 0.0
    k_new[bankrupt] = 0.0
    i_new[bankrupt] = 0.0
    s_new[bankrupt] = 0

    return k_new, i_new, s_new
