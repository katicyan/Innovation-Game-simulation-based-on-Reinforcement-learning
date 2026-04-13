from __future__ import annotations

import copy
from typing import Dict, List, Optional

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

import env as market_env


class MarketParallelEnv(ParallelEnv):
    """PettingZoo ParallelEnv wrapper for v1.0innovation.env.market.

    Each agent is one firm that selects a discrete budget index every step.
    All actions are applied simultaneously to the shared market state.
    """

    metadata = {"name": "innovation_market_parallel_v0", "render_modes": ["human"]}

    @staticmethod
    def _num_agents(base_env: market_env.market) -> int:
        return int(getattr(base_env, "num_of_agents"))

    @staticmethod
    def _get_doors(base_env: market_env.market) -> np.ndarray:
        return np.asarray(getattr(base_env, "doors"), dtype=int)

    @staticmethod
    def _set_doors(base_env: market_env.market, doors: np.ndarray) -> None:
        setattr(base_env, "doors", doors)

    def __init__(
        self,
        base_env: market_env.market,
        max_steps: int,
        bankrupt_penalty: float = 100000.0,
    ) -> None:
        if not isinstance(base_env, market_env.market):
            raise TypeError("base_env must be an instance of env.market")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")

        # A1
        self._template_env = copy.deepcopy(base_env)
        self._env = copy.deepcopy(base_env)
        self.max_steps = max_steps
        self.bankrupt_penalty = float(bankrupt_penalty)

        # B1
        self.possible_agents = [f"firm_{i}" for i in range(self._num_agents(self._template_env))]
        self.agent_name_mapping = {agent: idx for idx, agent in enumerate(self.possible_agents)}
        self.agents: List[str] = []

        # C1
        self._n = self._num_agents(self._template_env)
        self._obs_size = 4 * self._n + 2
        self._action_spaces = {
            agent: spaces.Discrete(self._template_env.num_actions) for agent in self.possible_agents
        }
        self._observation_spaces = {
            agent: spaces.Box(low=-1e12, high=1e12, shape=(self._obs_size,), dtype=np.float32)
            for agent in self.possible_agents
        }

        self._step_count = 0
        self._last_outputs = np.zeros(self._n, dtype=np.float32)

    def observation_space(self, agent: str):
        # C2
        return self._observation_spaces[agent]
    def action_space(self, agent: str):
        # B2
        return self._action_spaces[agent]


    def _all_agent_observations(self) -> Dict[str, np.ndarray]:
        # A1C3
        capital = np.asarray(self._env.now_capital, dtype=np.float32)
        r_and_d = np.asarray(self._env.innovation_input, dtype=np.float32)
        tech_state = np.asarray(self._env.technology_state, dtype=np.float32)
        tech_cost = np.asarray(self._env.codetotech(), dtype=np.float32)

        step_ratio = np.array([self._step_count / self.max_steps], dtype=np.float32)
        total_output = np.array([float(np.sum(self._last_outputs))], dtype=np.float32)
        base = np.concatenate([capital, r_and_d, tech_state, tech_cost, step_ratio, total_output], axis=0)

        obs: Dict[str, np.ndarray] = {}
        for agent in self.agents:
            idx = self.agent_name_mapping[agent]
            self_signal = np.array([idx / max(self._n - 1, 1)], dtype=np.float32)
            obs[agent] = np.concatenate([base[:-1], self_signal, base[-1:]], axis=0)
        return obs

    def _build_action_cost_grid(self) -> np.ndarray:
        # A2
        budget = np.maximum(np.zeros(self._n), np.asarray(self._env.now_capital, dtype=float))
        try:
            limit_costs = np.asarray(self._env.input_limit()[1], dtype=float)
            max_production_costs = np.minimum(budget, limit_costs)
        except Exception:
            max_production_costs = budget

        return np.linspace(1e-4, max_production_costs, num=self._env.num_actions)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            np.random.seed(seed)
        # A1
        self._env = copy.deepcopy(self._template_env)
        self._set_doors(self._env, np.ones(self._n, dtype=int))

        self.agents = self.possible_agents[:]
        self._step_count = 0
        self._last_outputs = np.zeros(self._n, dtype=np.float32)

        observations = self._all_agent_observations()
        infos = {
            agent: {
                "agent_index": self.agent_name_mapping[agent],
                "num_agents": self._n,
            }
            for agent in self.agents
        }
        return observations, infos

    def step(self, actions: Dict[str, int]):
        if not self.agents:
            return {}, {}, {}, {}, {}

        self._step_count += 1
        # print("entering step", self._step_count)
        self._env.update_tech()
        # print("end technology update")
        # B3
        action_indices = np.zeros(self._n, dtype=int)
        for agent in self.agents:
            idx = self.agent_name_mapping[agent]
            a = int(actions.get(agent, 0))
            if self._env.isbankrupt(idx):
                action_indices[idx] = 0
            else:
                action_indices[idx] = int(np.clip(a, 0, self._env.num_actions - 1))
        # A3
        action_costs = self._build_action_cost_grid()
        outputs, cash = self._env.session(action_indices, action_costs)
        outputs = np.asarray(outputs, dtype=np.float32)
        cash = np.asarray(cash, dtype=np.float32).reshape(-1)
        self._last_outputs = outputs

        rewards: Dict[str, float] = {}
        terminations: Dict[str, bool] = {}
        truncations: Dict[str, bool] = {}
        infos: Dict[str, dict] = {}

        for agent in self.agents:
            idx = self.agent_name_mapping[agent]
            reward = float(cash[idx])
            terminated = False

            if self._env.isbankrupt(idx):
                doors = self._get_doors(self._env)
                if doors[idx] == 1:
                    doors[idx] = 0
                    self._set_doors(self._env, doors)
                    reward -= self.bankrupt_penalty
                terminated = True

            truncated = self._step_count >= self.max_steps

            rewards[agent] = reward
            terminations[agent] = terminated
            truncations[agent] = truncated
            infos[agent] = {
                "capital": float(self._env.now_capital[idx]),
                "innovation_input": float(self._env.innovation_input[idx]),
                "technology_state": int(self._env.technology_state[idx]),
                "output": float(outputs[idx]),
                "raw_cash": float(cash[idx]),
            }

        self._env.update(cash, action_indices, action_costs)

        observations = self._all_agent_observations()

        alive_agents = [a for a in self.agents if not terminations[a] and not truncations[a]]
        if not alive_agents:
            self.agents = []
        else:
            self.agents = alive_agents

        return observations, rewards, terminations, truncations, infos

    def render(self):
        capital = np.asarray(self._env.now_capital, dtype=float)
        tech = np.asarray(self._env.technology_state, dtype=int)
        innovation_input = np.asarray(self._env.innovation_input, dtype=float)
        print(
            "step=", self._step_count,
            "tech=", tech,
            "capital=", capital.round(2),
            "innovation_input=", innovation_input.round(2),
            "outputs=", self._last_outputs.round(4),
        )

    def get_state(self):
        capital = np.asarray(self._env.now_capital, dtype=float)
        tech = np.asarray(self._env.technology_state, dtype=int)
        innovation_input = np.asarray(self._env.innovation_input, dtype=float)
        return capital, tech, innovation_input
    
    def close(self):
        return


def parallel_env(base_env: market_env.market, max_steps: int, bankrupt_penalty: float = 100000.0):
    return MarketParallelEnv(base_env=base_env, max_steps=max_steps, bankrupt_penalty=bankrupt_penalty)
