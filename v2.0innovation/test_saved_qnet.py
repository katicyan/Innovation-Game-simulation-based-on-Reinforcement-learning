import argparse
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

import env as market_env
from market_pettingzoo_env import parallel_env
from train_pettingzoo_independent_q import dict_save_as_csv, IndependentDQNAgent, list_save_as_csv

class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_linear_demand(intercept: float, slope: float):
    if slope <= 0:
        raise ValueError("slope must be positive")

    def demand(total_q: float) -> float:
        return max(0.0, intercept - slope * total_q)

    return demand


def build_env(args):
    demand_fn = make_linear_demand(args.demand_intercept, args.demand_slope)
    base_market = market_env.market(
        gamma=args.gamma,
        n=args.n_agents,
        demand_function=demand_fn,
        c=[float(x) for x in args.tech_levels],
        num_actions=args.num_actions,
        k0=[float(x) for x in args.initial_capital],
        i=[0.0] * args.n_agents,
        s=[0] * args.n_agents,
    )
    return parallel_env(base_market, max_steps=args.max_steps, bankrupt_penalty=args.bankrupt_penalty)


def select_greedy_action(model: QNetwork, obs: np.ndarray, device: torch.device) -> tuple[int, np.ndarray]:
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        q_values = model(obs_t)
    action = int(torch.argmax(q_values, dim=1).item())
    return action, q_values.detach().cpu().numpy()


def run_test(args):
    if args.seed == 0:
        args.seed = random.randint(1, 1000000)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model_dir = Path(args.model_path).absolute()
    model_paths = [model_dir / f"firm_{i}_qnet.pth" for i in range(args.n_agents)]
    for model_path in model_paths:
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    env = build_env(args)

    init_obs, _ = env.reset(seed=args.seed)
    if args.agent_name not in env.possible_agents:
        raise ValueError(f"agent_name '{args.agent_name}' is not in env agents: {env.possible_agents}")

    obs_dim = int(init_obs[args.agent_name].shape[0])
    agents: Dict[str, QNetwork] = {}
    load_reports: Dict[str, Dict[str, list[str]]] = {}
    for i in range(args.n_agents):
        agent_id = f"firm_{i}"
        model = QNetwork(obs_dim=obs_dim, num_actions=args.num_actions, hidden_dim=args.hidden_dim).to(device)
        load_result = model.load_state_dict(torch.load(model_paths[i], map_location=device))
        model.eval()
        agents[agent_id] = model
        load_reports[agent_id] = {
            "missing_keys": list(load_result.missing_keys),
            "unexpected_keys": list(load_result.unexpected_keys),
        }

    print(f"Loaded models from: {model_dir}")
    print(f"Test agent: {args.agent_name}")
    print(f"Device: {device}")
    print(f"Load report for {args.agent_name}: {load_reports.get(args.agent_name, {})}")

    episode_rewards = []
    first_step_logged = False
    Q = []
    revenue_history = {agent: [] for agent in env.possible_agents}
    action_history = {agent: [] for agent in env.possible_agents}
    # technology_history = {agent: [] for agent in env.possible_agents}
    balance_log_capital = []
    balance_log_tech = []
    balance_log_innovation_input = []
    for ep in range(args.episodes):
        observations, _ = env.reset(seed=args.seed + ep)
        ep_reward = 0.0
        total_reward = {agent: 0.0 for agent in env.possible_agents}

        while env.agents:
            actions = {}
            for agent in env.agents:
                if agent in agents:
                    action, q_values = select_greedy_action(agents[agent], observations[agent], device)
                    actions[agent] = action
                    if (agent == args.agent_name) and (not first_step_logged):
                        print(f"First-step Q-values for {agent}: {q_values.flatten()}")
                        print(f"First-step greedy action for {agent}: {action}")
                        first_step_logged = True
                else:
                    actions[agent] = int(np.random.randint(0, args.num_actions))
                action_history[agent].append(actions[agent])

            next_obs, rewards, terminations, truncations, _ = env.step(actions)
            capital, tech, innovation_input = env.get_state()
            balance_log_capital.append(capital)
            balance_log_tech.append(tech)
            balance_log_innovation_input.append(innovation_input)

            for agent, reward in rewards.items():
                action = int(actions[agent])
                done = bool(terminations[agent] or truncations[agent])
                
                revenue_history[agent].append(rewards[agent])

                if done:
                    fallback_next_obs = np.zeros_like(observations[agent], dtype=np.float32)
                    agent_next_obs = next_obs.get(agent, fallback_next_obs)
                else:
                    agent_next_obs = next_obs[agent]


                total_reward[agent] += float(reward)
            


            observations = next_obs

            done = terminations.get(args.agent_name, False) or truncations.get(args.agent_name, False)
            if done and args.agent_name not in env.agents:
                pass

        episode_rewards.append(ep_reward)
        print(f"Episode {ep + 1}: reward({args.agent_name}) = {ep_reward:.4f}")

    env.close()

    mean_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    print(f"Average reward over {args.episodes} episodes: {mean_reward:.4f}")
    cols = ",".join([f"action_{i}" for i in range(args.num_actions)])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # output_path = output_dir / args.output_file

    # np.savetxt(
    #     output_path,
    #     np.vstack(Q),
    #     delimiter=",",
    #     header=cols,
    #     comments="",
    #     fmt="%.6f"
    # )
    # print(f"Saved Q-values to: {output_path}")
    # np.savez(output_dir / "revenue_history.npz", **revenue_history)
    # np.savez(output_dir / "action_history.npz", **action_history)
    dict_save_as_csv(revenue_history, output_dir / "revenue_history.csv")
    dict_save_as_csv(action_history, output_dir / "action_history.csv")
    list_save_as_csv(balance_log_capital, output_dir / "balance_log_capital.csv")
    list_save_as_csv(balance_log_tech, output_dir / "balance_log_tech.csv")
    list_save_as_csv(balance_log_innovation_input, output_dir / "balance_log_innovation_input.csv")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a saved QNetwork model from firm_0_qnet.pth")
    parser.add_argument("--model-path", type=str, default="./v2.0innovation/experiment_data/trained_models/")
    parser.add_argument("--agent-name", type=str, default="firm_0")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--n-agents", type=int, default=5)
    parser.add_argument("--num-actions", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--tech-levels", type=float, nargs="+", default=[20.0, 10.0, 5.0, 1.0])
    parser.add_argument("--initial-capital", type=float, nargs="+", default=[50.0, 50.0, 50.0, 50.0, 50.0,50.0, 50.0, 50.0, 50.0, 50.0])
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--bankrupt-penalty", type=float, default=100000.0)
    parser.add_argument("--demand-intercept", type=float, default=200.0)
    parser.add_argument("--demand-slope", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="./v2.0innovation/experiment_data/test_results/")

    run_test(parser.parse_args())
