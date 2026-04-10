from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

import gymnasium
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from pettingzoo.utils import ParallelEnv

from market_core import (
    apply_bankruptcy_reset,
    apply_tech_progress,
    codetocost,
    run_market_session,
    update_capital_innovation,
)


def default_demand(price_input: float) -> float:
    return 100.0 - float(price_input)


class MarketParallelEnv(ParallelEnv):
    metadata = {"name": "innovation_market_v0", "render_modes": ["human"], "is_parallelizable": True}

    def __init__(
        self,
        gamma: float = 0.9,
        n: int = 2,
        demand_function=default_demand,
        c: Sequence[float] | np.ndarray = (20.0, 10.0),
        k0: Sequence[float] | np.ndarray = (100.0, 100.0),
        i0: Sequence[float] | np.ndarray = (0.0, 0.0),
        s0: Sequence[int] | np.ndarray = (0, 0),
        max_steps: int = 200,
        render_mode: str | None = None,
    ) -> None:
        if n <= 0:
            raise ValueError("n must be positive")

        self.gamma = float(gamma)
        self.n = int(n)
        self.demand_function = demand_function
        self.c = np.asarray(c, dtype=np.float64)

        if self.c.shape[0] == 0:
            raise ValueError("Cost levels cannot be empty")

        self._k0 = np.asarray(k0, dtype=np.float64)
        self._i0 = np.asarray(i0, dtype=np.float64)
        self._s0 = np.asarray(s0, dtype=np.int64)

        if any(arr.shape[0] != self.n for arr in (self._k0, self._i0, self._s0)):
            raise ValueError("k0, i0, s0 must each have length n")

        self.max_steps = int(max_steps)
        self.render_mode = render_mode

        self.possible_agents = [f"firm_{idx}" for idx in range(self.n)]
        self.agent_name_mapping = {name: idx for idx, name in enumerate(self.possible_agents)}
        self.agents = copy.copy(self.possible_agents)

        obs_low = np.array([-np.inf] * (4 * self.n + 1), dtype=np.float32)
        obs_high = np.array([np.inf] * (4 * self.n + 1), dtype=np.float32)

        self.observation_spaces = {
            agent: spaces.Box(low=obs_low, high=obs_high, dtype=np.float32) for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: spaces.Box(low=np.array([0.0], dtype=np.float32), high=np.array([1.0], dtype=np.float32), dtype=np.float32)
            for agent in self.possible_agents
        }

        self._np_random: np.random.Generator | None = None
        self._step_count = 0

        self.k = self._k0.copy()
        self.i = self._i0.copy()
        self.s = self._s0.copy()
        self.sc = codetocost(self.s, self.c)

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]

    def _get_obs(self) -> dict[str, np.ndarray]:
        base = np.concatenate(
            [
                self.k.astype(np.float32),
                self.i.astype(np.float32),
                self.s.astype(np.float32),
                self.sc.astype(np.float32),
                np.array([float(self._step_count)], dtype=np.float32),
            ]
        )
        return {agent: base.copy() for agent in self.agents}

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        self._np_random, _ = seeding.np_random(seed)

        self.agents = copy.copy(self.possible_agents)
        self._step_count = 0

        self.k = self._k0.copy()
        self.i = self._i0.copy()
        self.s = self._s0.copy()
        self.sc = codetocost(self.s, self.c)

        observations = self._get_obs()
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions: Mapping[str, np.ndarray | float]):
        if not self.agents:
            return {}, {}, {}, {}, {}

        if self._np_random is None:
            self._np_random, _ = seeding.np_random(None)

        assert self._np_random is not None

        # Keep the same event ordering as the original loop: tech progress then expansion decisions.
        self.s, self.i, progress_prob, progressed = apply_tech_progress(
            self.s,
            self.i,
            num_cost_levels=len(self.c),
            rng=self._np_random,
        )
        self.sc = codetocost(self.s, self.c)

        ratio_vec = np.zeros(self.n, dtype=np.float64)
        for agent in self.agents:
            idx = self.agent_name_mapping[agent]
            raw = actions.get(agent, 0.0)
            if isinstance(raw, np.ndarray):
                val = float(np.asarray(raw).reshape(-1)[0])
            else:
                val = float(raw)
            ratio_vec[idx] = np.clip(val, 0.0, 1.0)

        expansion_requested = ratio_vec * self.k

        cash, q, expansion_effective, constrained, price, total_quantity = run_market_session(
            self.k,
            self.sc,
            self.demand_function,
            expansion_requested,
        )

        self.k, self.i = update_capital_innovation(
            self.k,
            self.i,
            self.s,
            self.c,
            expansion_effective,
            cash,
        )
        self.k, self.i, self.s = apply_bankruptcy_reset(self.k, self.i, self.s)
        self.sc = codetocost(self.s, self.c)

        self._step_count += 1

        bankrupt_all = bool(np.all(self.k <= 0.0))
        horizon_end = bool(self._step_count >= self.max_steps)

        terminations = {agent: bankrupt_all for agent in self.agents}
        truncations = {agent: horizon_end for agent in self.agents}
        rewards = {
            agent: float(cash[self.agent_name_mapping[agent]])
            for agent in self.agents
        }

        infos = {}
        for agent in self.agents:
            idx = self.agent_name_mapping[agent]
            infos[agent] = {
                "price": price,
                "total_quantity": total_quantity,
                "quantity": float(q[idx]),
                "expansion_requested": float(expansion_requested[idx]),
                "expansion_effective": float(expansion_effective[idx]),
                "constrained": bool(constrained[idx]),
                "progress_probability": float(progress_prob[idx]),
                "progressed": bool(progressed[idx]),
                "capital": float(self.k[idx]),
                "innovation_stock": float(self.i[idx]),
                "tech_state": int(self.s[idx]),
                "cost": float(self.sc[idx]),
            }

        observations = self._get_obs()

        if bankrupt_all or horizon_end:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(self):
        if self.render_mode != "human":
            return None

        print(
            {
                "step": self._step_count,
                "capital": self.k.tolist(),
                "innovation": self.i.tolist(),
                "tech": self.s.tolist(),
                "cost": self.sc.tolist(),
            }
        )
        return None

    def close(self):
        return None
