# Innovation Game simulation based on Reinforcement learning

Thanks for your reading!

This is my coding projects for my personal undergraduate paper of economics degree in Xiamen University.

I am a greehand of everything in coding, and hope one day I can figure all things out.

This project is under constructing.

## New MARL Benchmark Path (PettingZoo + RLlib)

This repository now includes a true multi-agent environment and baseline benchmark scripts.

### Added modules

- `market_core.py`: pure transition helpers extracted from the legacy environment logic
- `market_marl_env.py`: PettingZoo `ParallelEnv` implementation (`MarketParallelEnv`)
- `reinforcement/access_from_paper/v0.1_producer/executables/train_rllib_ppo.py`: PPO training entrypoint
- `reinforcement/access_from_paper/v0.1_producer/executables/eval_policy.py`: checkpoint evaluation entrypoint
- `tests/test_market_core.py` and `tests/test_market_marl_env.py`: smoke tests

### Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install gymnasium pettingzoo ray[rllib] numpy scipy pytest
```

### Run tests

```bash
pytest -q
```

### Train PPO baseline

```bash
python reinforcement/access_from_paper/v0.1_producer/executables/train_rllib_ppo.py \
	--stop-timesteps 5000 \
	--seed 42 \
	--num-workers 1
```

### Evaluate a checkpoint

```bash
python reinforcement/access_from_paper/v0.1_producer/executables/eval_policy.py \
	--checkpoint <path_to_checkpoint> \
	--episodes 20 \
	--seed 42
```
