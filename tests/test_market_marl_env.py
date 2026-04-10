import numpy as np

from market_marl_env import MarketParallelEnv


def test_reset_and_spaces():
    env = MarketParallelEnv(max_steps=5)
    obs, infos = env.reset(seed=123)

    assert set(obs.keys()) == set(env.possible_agents)
    assert set(infos.keys()) == set(env.possible_agents)

    for agent, agent_obs in obs.items():
        assert env.observation_space(agent).contains(agent_obs)


def test_step_signature_and_progress():
    env = MarketParallelEnv(max_steps=3)
    obs, _ = env.reset(seed=123)

    actions = {agent: np.array([0.5], dtype=np.float32) for agent in obs.keys()}
    next_obs, rewards, terminations, truncations, infos = env.step(actions)

    assert isinstance(next_obs, dict)
    assert isinstance(rewards, dict)
    assert isinstance(terminations, dict)
    assert isinstance(truncations, dict)
    assert isinstance(infos, dict)

    for agent in env.possible_agents:
        assert agent in rewards
        assert np.isfinite(rewards[agent])


def test_truncation_by_max_steps():
    env = MarketParallelEnv(max_steps=1)
    obs, _ = env.reset(seed=1)

    actions = {agent: np.array([0.1], dtype=np.float32) for agent in obs.keys()}
    _, _, _, truncations, _ = env.step(actions)

    assert all(truncations.values())
