import gymnasium as gym

from stable_baselines3 import DQN
class InnovationEnv(gym.Env):
    def __init__(self, num_of_agents, num_of_states, num_of_actions, technology_level, demand_function, gamma):
        super(InnovationEnv, self).__init__()

        self.num_of_agents = num_of_agents
        self.num_of_states = num_of_states
        self.num_of_actions = num_of_actions
        self.technology_level = technology_level
        self.demand_function = demand_function
        self.gamma = gamma

        self.now_capital = np.zeros(self.num_of_agents) # capital list
        self.innovation_input = np.zeros(self.num_of_agents) # R&D input list

    def reset(self,seed=None, options=None):
        super().reset()
        self.now_capital = np.zeros(self.num_of_agents) # capital list
        self.innovation_input = np.zeros(self.num_of_agents) # R&D input list
        return self.now_capital, self.innovation_input
    


    def session(self, choices_actions, production_costs):
        '''
        choices_actions: list of actions chosen by agents, which is the index of production cost list\
        production_costs: list of production costs corresponding to actions, which is determined by the budget and input limit
        return: output and profit of each agent in this session
        '''


env = gym.make("CartPole-v1", render_mode="human")

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
model.save("dqn_cartpole")

del model # remove to demonstrate saving and loading

model = DQN.load("dqn_cartpole")

obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()


