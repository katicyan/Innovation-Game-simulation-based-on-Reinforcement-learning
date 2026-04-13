import argparse
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List
import pandas as pd

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import env as market_env
from market_pettingzoo_env import parallel_env

def dict_save_as_csv(data: Dict[str, List], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, mode="w")
    print(f"Saved data to: {output_path}")

def list_save_as_csv(data: List, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    l = len(data[0])
    df = pd.DataFrame(data, columns=[f"firm_{i}" for i in range(l)])
    df.to_csv(output_path, index=False, mode="w")
    print(f"Saved data to: {output_path}")

def make_linear_demand(intercept: float, slope: float):
    if slope <= 0:
        raise ValueError("slope must be positive")

    def demand(total_q: float) -> float:
        return max(0.0, intercept - slope * total_q)

    return demand


def epsilon_decay(step: int, total_steps: int, eps_start: float, eps_end: float) -> float:
    ratio = step / max(total_steps, 1)
    return max(eps_end, eps_start * (1.0 - ratio))


def print_training_progress(
    current_episode: int,
    total_episodes: int,
    mean_reward: float,
    mean_loss: float,
    epsilon: float,
    width: int = 30,
) -> None:
    ratio = current_episode / max(total_episodes, 1)
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    print(
        f"\rTraining [{bar}] {current_episode}/{total_episodes} "
        f"reward={mean_reward:.3f} loss={mean_loss:.5f} eps={epsilon:.4f}",
        end="",
        flush=True,
    )


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


@dataclass
class Transition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class IndependentDQNAgent:
    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        hidden_dim: int,
        lr: float,
        gamma: float,
        buffer_size: int,
        batch_size: int,
        device: torch.device,
    ) -> None:
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device

        self.q_net = QNetwork(obs_dim, num_actions, hidden_dim).to(device)
        self.target_net = QNetwork(obs_dim, num_actions, hidden_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        self.loss_fn = nn.MSELoss()

    def act(self, obs: np.ndarray, epsilon: float):
        if np.random.rand() < epsilon:
            action = int(np.random.randint(0, self.num_actions))
            return action, np.zeros((1, self.num_actions), dtype=np.float32)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(obs_t)
        return int(torch.argmax(q_values, dim=1).item()), q_values.detach().cpu().numpy()

    def store(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool) -> None:
        self.buffer.add(
            Transition(
                obs=np.asarray(obs, dtype=np.float32),
                action=int(action),
                reward=float(reward),
                next_obs=np.asarray(next_obs, dtype=np.float32),
                done=bool(done),
            )
        )

    def update(self) -> float:
        if len(self.buffer) < self.batch_size:
            return 0.0

        batch = self.buffer.sample(self.batch_size)

        obs = torch.tensor(np.stack([t.obs for t in batch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(np.stack([t.next_obs for t in batch]), dtype=torch.float32, device=self.device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=self.device)

        q_values = self.q_net(obs).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_obs).max(dim=1).values
            target_q = rewards + self.gamma * (1.0 - dones) * next_q

        loss = self.loss_fn(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        return float(loss.item())

    def sync_target(self) -> None:
        self.target_net.load_state_dict(self.q_net.state_dict())


def export_qnetwork_parameters(agents: Dict[str, IndependentDQNAgent], output_dir: Path) -> None:
    model_dir = output_dir / "trained_models"
    report_dir = output_dir / "model_parameters"
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    for agent_name, dqn_agent in agents.items():
        model_path = model_dir / f"{agent_name}_qnet.pth"
        torch.save(dqn_agent.q_net.state_dict(), model_path)
        print(f"Saved model state_dict to: {model_path}")

        report_path = report_dir / f"{agent_name}_qnet_parameters.txt"
        with report_path.open("w") as f:
            for name, param in dqn_agent.q_net.named_parameters():
                arr = param.detach().cpu().numpy()
                flat = arr.reshape(-1)
                preview = ", ".join(f"{x:.6f}" for x in flat[:8])
                f.write(
                    f"{name} | shape={arr.shape} | mean={arr.mean():.6f} | "
                    f"std={arr.std():.6f} | preview=[{preview}]\\n"
                )
        print(f"Saved parameter report to: {report_path}")


def train(args):
    if args.seed == 0:
        args.seed = random.randint(1, 1000000)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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

    env = parallel_env(base_market, max_steps=args.max_steps, bankrupt_penalty=args.bankrupt_penalty)

    init_obs, _ = env.reset(seed=args.seed)
    obs_dim = int(init_obs[env.possible_agents[0]].shape[0])
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    agents: Dict[str, IndependentDQNAgent] = {
        agent_name: IndependentDQNAgent(
            obs_dim=obs_dim,
            num_actions=args.num_actions,
            hidden_dim=args.hidden_dim,
            lr=args.learning_rate,
            gamma=args.gamma,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            device=device,
        )
        for agent_name in env.possible_agents
    }

    episode_rewards = []
    global_step = 0
    Q = []
    revenue_history = {agent: [] for agent in env.possible_agents}
    action_history = {agent: [] for agent in env.possible_agents}
    # technology_history = {agent: [] for agent in env.possible_agents}
    balance_log_capital = []
    balance_log_tech = []
    balance_log_innovation_input = []
    for episode in range(args.episodes):
        observations, infos = env.reset(seed=args.seed + episode)
        total_reward = {agent: 0.0 for agent in env.possible_agents}
        total_loss = {agent: 0.0 for agent in env.possible_agents}
        loss_updates = {agent: 0 for agent in env.possible_agents}
        q_table = {agent: np.ones((1, args.num_actions), dtype=np.float32) * 1e-4 for agent in env.possible_agents}
        # print("P1 Env info")
        # env._env.info()
        while env.agents:
            actions = {}
            for agent in env.agents:
                eps = epsilon_decay(global_step, args.eps_decay_steps, args.eps_start, args.eps_end)
                actions[agent], q_table[agent] = agents[agent].act(observations[agent], eps)
                action_history[agent].append(actions[agent])
            # print(f"P{episode}{env._step_count} Env info before step")
            # env._env.info()
            next_obs, rewards, terminations, truncations, infos = env.step(actions)
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

                agents[agent].store(
                    obs=observations[agent],
                    action=action,
                    reward=float(reward),
                    next_obs=agent_next_obs,
                    done=done,
                )
                loss = agents[agent].update()
                if loss > 0:
                    total_loss[agent] += loss
                    loss_updates[agent] += 1

                total_reward[agent] += float(reward)

            observations = next_obs
            global_step += 1

            if global_step % args.target_update_interval == 0:
                for dqn_agent in agents.values():
                    dqn_agent.sync_target()

        mean_reward = float(np.mean(list(total_reward.values())))
        episode_rewards.append(mean_reward)
        mean_loss_vals = [
            (total_loss[a] / loss_updates[a]) if loss_updates[a] > 0 else 0.0
            for a in env.possible_agents
        ]
        mean_loss = float(np.mean(mean_loss_vals))

        if (episode + 1) % args.log_every == 0:
            print(
                f"Episode {episode + 1}: mean reward={mean_reward:.3f}, "
                f"mean loss={mean_loss:.5f}, epsilon={epsilon_decay(global_step, args.eps_decay_steps, args.eps_start, args.eps_end):.4f}"
            )

        current_eps = epsilon_decay(global_step, args.eps_decay_steps, args.eps_start, args.eps_end)
        print_training_progress(episode + 1, args.episodes, mean_reward, mean_loss, current_eps)
        
        action_q_matrix = np.vstack([q_table[a].reshape(1, -1) for a in env.possible_agents])
        Q.append(action_q_matrix)
    # C8
    cols = ",".join([f"action_{i}" for i in range(args.num_actions)])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_file

    np.savetxt(
        output_path,
        np.vstack(Q),
        delimiter=",",
        header=cols,
        comments="",
        fmt="%.6f"
    )
    print(f"Saved Q-values to: {output_path}")
    # np.savez(output_dir / "revenue_history.npz", **revenue_history)
    # np.savez(output_dir / "action_history.npz", **action_history)
    dict_save_as_csv(revenue_history, output_dir / "revenue_history.csv")
    dict_save_as_csv(action_history, output_dir / "action_history.csv")
    list_save_as_csv(balance_log_capital, output_dir / "balance_log_capital.csv")
    list_save_as_csv(balance_log_tech, output_dir / "balance_log_tech.csv")
    list_save_as_csv(balance_log_innovation_input, output_dir / "balance_log_innovation_input.csv")

    export_qnetwork_parameters(agents, output_dir)

    print()


    env.close()
    print("Training completed.")
    print(f"Final average reward (last 20 episodes): {np.mean(episode_rewards[-20:]):.3f}")


def test(args):
    # A5 B5 C4 -> A8 B8
    if args.seed == 0:
        args.seed = random.randint(1, 1000000)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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

    env = parallel_env(base_market, max_steps=args.max_steps, bankrupt_penalty=args.bankrupt_penalty)

    init_obs, _ = env.reset(seed=args.seed)
    obs_dim = int(init_obs[env.possible_agents[0]].shape[0])
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    agents: Dict[str, IndependentDQNAgent] = {
        agent_name: IndependentDQNAgent(
            obs_dim=obs_dim,
            num_actions=args.num_actions,
            hidden_dim=args.hidden_dim,
            lr=args.learning_rate,
            gamma=args.gamma,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            device=device,
        )
        for agent_name in env.possible_agents
    }

    episode_rewards = []
    global_step = 0
    Q = []
    revenue_history = {agent: [] for agent in env.possible_agents}
    action_history = {agent: [] for agent in env.possible_agents}
    # technology_history = {agent: [] for agent in env.possible_agents}
    balance_log_capital = []
    balance_log_tech = []
    balance_log_innovation_input = []
    for episode in range(args.episodes):
        observations, infos = env.reset(seed=args.seed + episode)
        total_reward = {agent: 0.0 for agent in env.possible_agents}
        total_loss = {agent: 0.0 for agent in env.possible_agents}
        loss_updates = {agent: 0 for agent in env.possible_agents}
        q_table = {agent: np.ones((1, args.num_actions), dtype=np.float32) * 1e-4 for agent in env.possible_agents}
        print("P1 Env info")
        env._env.info()
        while env.agents:
            actions = {}
            for agent in env.agents:
                eps = epsilon_decay(global_step, args.eps_decay_steps, args.eps_start, args.eps_end)
                actions[agent], q_table[agent] = agents[agent].act(observations[agent], eps)
                action_history[agent].append(actions[agent])
            # print(f"P{episode}{env._step_count} Env info before step")
            # env._env.info()
            next_obs, rewards, terminations, truncations, infos = env.step(actions)
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

                agents[agent].store(
                    obs=observations[agent],
                    action=action,
                    reward=float(reward),
                    next_obs=agent_next_obs,
                    done=done,
                )
                loss = agents[agent].update()
                if loss > 0:
                    total_loss[agent] += loss
                    loss_updates[agent] += 1

                total_reward[agent] += float(reward)

            observations = next_obs
            global_step += 1

            if global_step % args.target_update_interval == 0:
                for dqn_agent in agents.values():
                    dqn_agent.sync_target()

        mean_reward = float(np.mean(list(total_reward.values())))
        episode_rewards.append(mean_reward)
        mean_loss_vals = [
            (total_loss[a] / loss_updates[a]) if loss_updates[a] > 0 else 0.0
            for a in env.possible_agents
        ]
        mean_loss = float(np.mean(mean_loss_vals))

        if (episode + 1) % args.log_every == 0:
            print(
                f"Episode {episode + 1}: mean reward={mean_reward:.3f}, "
                f"mean loss={mean_loss:.5f}, epsilon={epsilon_decay(global_step, args.eps_decay_steps, args.eps_start, args.eps_end):.4f}"
            )
        
        action_q_matrix = np.vstack([q_table[a].reshape(1, -1) for a in env.possible_agents])
        Q.append(action_q_matrix)
    # C8
    cols = ",".join([f"action_{i}" for i in range(args.num_actions)])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_file

    np.savetxt(
        output_path,
        np.vstack(Q),
        delimiter=",",
        header=cols,
        comments="",
        fmt="%.6f"
    )
    print(f"Saved Q-values to: {output_path}")
    # np.savez(output_dir / "revenue_history.npz", **revenue_history)
    # np.savez(output_dir / "action_history.npz", **action_history)
    dict_save_as_csv(revenue_history, output_dir / "revenue_history.csv")
    dict_save_as_csv(action_history, output_dir / "action_history.csv")
    list_save_as_csv(balance_log_capital, output_dir / "balance_log_capital.csv")
    list_save_as_csv(balance_log_tech, output_dir / "balance_log_tech.csv")
    list_save_as_csv(balance_log_innovation_input, output_dir / "balance_log_innovation_input.csv")

    export_qnetwork_parameters(agents, output_dir)


    env.close()
    print("Training completed.")
    print(f"Final Q-values saved to: {output_path}")

if __name__ == "__main__":
    # parser0 = argparse.ArgumentParser(description="Train or test Independent DQN on PettingZoo market env")
    # parser0.add_argument("--mode", type=str, choices=["train", "test"], default="train")
    # if parser0.parse_known_args()[0].mode == "test":
    #     parser = argparse.ArgumentParser(description="Test Independent DQN on PettingZoo market env")
    #     parser.add_argument("--model-dir", type=str, default="./v2.0innovation/experiment_data/trained_models")
    #     parser.add_argument("--output-dir", type=str, default="./v2.0innovation/experiment_data/test_results")
    #     parser.add_argument("--episodes", type=int, default=20)
    #     parser.add_argument("--max-steps", type=int, default=200)
    #     parser.add_argument("--n-agents", type=int, default=5)
    #     parser.add_argument("--num-actions", type=int, default=10)
    #     parser.add_argument("--tech-levels", type=float, nargs="+", default=[20.0, 10.0, 5.0, 1.0])
    #     parser.add_argument("--initial-capital", type=float, nargs="+", default=[50.0, 50.0, 50.0, 50.0, 50.0])
    #     parser.add_argument("--gamma", type=float, default=0.95)
    #     parser.add_argument("--demand-intercept", type=float, default=200.0)
    #     parser.add_argument("--demand-slope", type=float, default=1.0)
    #     parser.add_argument("--seed", type=int, default=42)

    #     test(parser.parse_args())
    # else:
    parser = argparse.ArgumentParser(description="Independent DQN over PettingZoo market env")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--n-agents", type=int, default=5)
    parser.add_argument("--num-actions", type=int, default=10)
    parser.add_argument("--tech-levels", type=float, nargs="+", default=[20.0, 10.0, 5.0, 1.0])
    parser.add_argument("--initial-capital", type=float, nargs="+", default=[50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0])
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--target-update-interval", type=int, default=50)
    parser.add_argument("--eps-start", type=float, default=0.2)
    parser.add_argument("--eps-end", type=float, default=0.02)
    parser.add_argument("--eps-decay-steps", type=int, default=20000)
    parser.add_argument("--bankrupt-penalty", type=float, default=100000.0)
    parser.add_argument("--demand-intercept", type=float, default=200.0)
    parser.add_argument("--demand-slope", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="./v2.0innovation/experiment_data")
    parser.add_argument("--output-file", type=str, default="e0001_Q_values.csv")

    train(parser.parse_args())
