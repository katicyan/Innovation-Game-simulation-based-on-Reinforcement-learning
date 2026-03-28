import numpy as np
import env

class simulate:
    def __init__(self, alpha, eps, epi, max_steps, env) -> None:
        '''
        alpha: stepwise\
        eps: greedy para\
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
        Q = np.zeros((self.env.n, self.env.num_states, self.env.num_actions))
        actions = np.zeros((self.max_steps, self.env.n))
        profit = np.zeros((self.max_steps, self.env.n))
        for epis in range(self.epi):
            ienv = self.env
            done = False
            step = 0
            
            incre_Q = np.zeros(Q.shape)
            incre_actions = np.zeros((self.max_steps, self.env.n), dtype=int)
            incre_profit = np.zeros((self.max_steps, self.env.n))
            e = np.zeros((self.max_steps,self.env.n))

            while not done and step < self.max_steps:
                # print(f'ienv.k: {ienv.k}, ienv.inputlimit: {ienv.input_limit()}')
                max_e = np.minimum(np.asarray(ienv.k), np.asarray(ienv.input_limit()[1]))
                e_levels = np.linspace(1e-10, max_e, num=self.env.num_actions)
                ienv.new_tech()
                # agent decisions
                for _ in range(self.env.n):
                    
                    if env.dice(self.eps):
                        # actions[step][_] = np.random.randint(self.env.num_actions)
                        action = np.random.randint(self.env.num_actions)
                    else:
                        action = np.argmax(incre_Q[_, ienv.s[_]])

                    incre_actions[step][_] = action
                    e[step][_] = e_levels[action][_]

                # 数据收集
                reward = ienv.session(e[step])
                
                incre_profit[step] = reward

                for _ in range(self.env.n):    
                    
                    # incre_action = np.append(incre_action, actions[step])
                    
                    # 更新环境
                    # 更新对策略的评估
                    # print(f'Reward like: {reward.shape}')
                    state_idx = int(ienv.s[_])
                    action_idx = int(incre_actions[step][_])
                    incre_Q[_, state_idx, action_idx] = (1 - self.alpha) * incre_Q[_, state_idx, action_idx] \
                    + self.alpha * (reward[_] + self.gamma * np.max(incre_Q[_, state_idx, :]))


                ienv.update(e[step])
                ienv.update_agent()
                
                step += 1

            
            profit = (epis/(1+epis))*profit + incre_profit/(epis+1)
            actions = (epis/(1+epis))*actions + incre_actions/(epis+1)
            Q = (epis/(1+epis))*Q + incre_Q/(epis+1)
    
        return actions, profit, Q
        






def update(capitals,actions,techs,demand_function,cost,output,invest,transform):
    price = demand_function(sum(actions))
    profit = [price*output(actions[i])-cost(techs[i],actions[i])-invest(actions[i]) for i in range(len(actions))]
    techs = transform(techs, actions,capitals)
    capitals += profit
    return capitals, techs