import numpy as np
import env
import copy

class simulate:
    def __init__(self, alpha, eps, epi, max_steps, env) -> None:
        '''
        alpha: stepwise\
        eps: greedy para receive a function that can decay with time steps\
        epi: mont carlo para\
        max_steps: in an epsisode\
        env: environment object should have all info we need\
        '''
        self.alpha = alpha
        self.gamma = env.gamma
        self.eps = eps
        self.epi = epi
        self.max_steps = max_steps
        self.env = env

    def begin(self):
        Q = np.zeros((self.env.num_of_agents, self.env.num_states, self.env.num_actions))
        actions = np.zeros((self.max_steps, self.env.num_of_agents))
        profit = np.zeros((self.max_steps, self.env.num_of_agents))
        output = np.zeros((self.max_steps, self.env.num_of_agents))
        techs = np.zeros((self.max_steps, self.env.num_of_agents))

        for epis in range(self.epi):
            # reset
            ienv = copy.deepcopy(self.env)
            done = False
            step = 0
            incre_Q = np.zeros(Q.shape)
            incre_actions = np.zeros((self.max_steps, self.env.num_of_agents), dtype=int)
            incre_profit = np.zeros((self.max_steps, self.env.num_of_agents))
            incre_output = np.zeros((self.max_steps, self.env.num_of_agents))
            incre_techs = np.zeros((self.max_steps, self.env.num_of_agents),dtype=int)

            # step
            while not done and step < self.max_steps:
                
                #预算决定
                budget = np.maximum.reduce([np.zeros(ienv.num_of_agents), np.asarray(ienv.now_capital)])
                max_production_costs = np.minimum.reduce([budget, np.asarray(ienv.input_limit()[1])])
                actions_costs = np.linspace(1e-4, max_production_costs, num=self.env.num_actions)
                incre_techs[step] = ienv.update_tech()

                # 实施行动
                for _ in range(self.env.num_of_agents):
                    if ienv.isbankrupt(_):
                        action = 0
                    else:
                        if env.dice(self.eps(step,self.max_steps)):
                            action = np.random.randint(0, len(actions_costs))
                        else:
                            action = np.argmax(incre_Q[_, ienv.technology_state[_]])
                    incre_actions[step][_] = action
                incre_output[step], incre_profit[step] = ienv.session(incre_actions[step],actions_costs)
                # 结算奖励并更新 Q 值
                for _ in range(self.env.num_of_agents):
                    if ienv.isbankrupt(_):
                        incre_profit[step][_] -= 100000
                    state_idx = int(ienv.technology_state[_])
                    action_idx = int(incre_actions[step][_])
                    
                    incre_Q[_, state_idx, action_idx] = (1 - self.alpha) * incre_Q[_, state_idx, action_idx] \
                    + self.alpha * (incre_profit[step][_] + self.gamma * np.max(incre_Q[_, state_idx, :]))
                ienv.update(incre_profit[step], incre_actions[step],actions_costs)
                step += 1
            
            # 蒙特卡洛
            profit = (epis/(1+epis))*profit + incre_profit/(epis+1)
            actions = (epis/(1+epis))*actions + incre_actions/(epis+1)
            Q = (epis/(1+epis))*Q + incre_Q/(epis+1)
            output = (epis/(1+epis))*output + incre_output/(epis+1)
            techs = (epis/(1+epis))*techs + incre_techs/(epis+1)
        
        return profit, actions, Q, output, techs
            


             