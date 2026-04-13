import numpy as np
import utili
def dice(p) -> int:
    if p < 0 or p > 1:
        raise ValueError("Not prob")
    m = np.random.uniform(0,1)
    while p > m:
        return 1
    return 0

class market:
    def __init__(self,gamma:float,n:int,
                 demand_function,c:list,num_actions:int,k0:list,i:list,s:list):
        '''
        k0 initial capital list\
        demand_function return price\
        c cost list\
        n number of firms\
        i innovation input\
        s technology states code\
        beta discount rate
        '''
        
        self.initial_capital = k0
        self.now_capital = k0
        self.demand_function = demand_function
        self.technology_level = c
        self.technology_state = s
        self.gamma = gamma
        self.innovation_input = i
        self.num_of_agents = n
        self.num_states = len(c)
        self.num_actions = num_actions
        self.doors = np.ones(n, dtype=int)

        if len(self.initial_capital) != self.num_of_agents or len(self.innovation_input) != self.num_of_agents or len(self.technology_state) != self.num_of_agents:
            raise ValueError("Length of initial_capital, innovation_input, and technology_state must match the number of firms.")
        if self.gamma < 0 or self.gamma > 1:
            raise ValueError("Discount factor gamma must be between 0 and 1.")
        if self.num_actions <= 0:
            raise ValueError("Number of actions must be a positive integer.")

    def info(self):
        print(f'number of firms: {self.num_of_agents}')
        print(f'current capital: {self.now_capital}')
        # print(f'initial technology level: {self.technology_level}')
        print(f'current technology state: {self.technology_state}')
        print(f'current innovation input: {self.innovation_input}')
        # print(f'discount factor: {self.gamma}')

    def innovation_to_probability(self, innovation_input):
        '''
        Map non-negative innovation input to progress probability.

        For x in [0, +inf), this normalized logistic mapping gives:
        x=0 -> p=0, x->+inf -> p->1.

        Raw logistic: s(x) = 1 / (1 + exp(-k * (x - x0)))
        Normalized: p(x) = (s(x) - s(0)) / (1 - s(0))
        '''
        x = np.asarray(innovation_input, dtype=float)
        x = np.maximum(x, 0.0)
        k = 0.03
        x0 = 10000.0
        logits = np.clip(k * (x - x0), -60, 60)
        s = 1.0 / (1.0 + np.exp(-logits))

        s0_logit = np.clip(-k * x0, -60, 60)
        s0 = 1.0 / (1.0 + np.exp(-s0_logit))

        p = (s - s0) / (1.0 - s0)
        return np.clip(p, 0.0, 1.0)
    
    def update_tech(self):
        '''
        model simplification innovation experienc does not accumulate between\
        different tech state\
        
        s is a list of technology states from different firms

        transform state code into prob of progress, you can change different funcs here
        
     
        '''
        # print("updating technology states...")
        p = self.innovation_to_probability(self.innovation_input)
        # print(f'innovation input: {self.innovation_input}, progress prob: {p}, tech state: {self.technology_state}')

        
        # for _ in range(self.n):
        #     print(f'company {_} has innovation input {self.innovation_input[_]} with progress prob {p[_]}')
        # print(f'innovation input: {self.innovation_input}, progress prob: {p}')
        for _ in range(len(p)):
            
            # p = 0.1*self.innovation_input[_] / (0.1*self.innovation_input[_]+1)
            if dice(p[_]) and self.technology_state[_] < len(self.technology_level) - 1:
                # print(f'company {_} has a tech progress from state {self.technology_state[_]} to state {self.technology_state[_]+1} with innovation input {self.innovation_input[_]}')
                self.technology_state[_] += 1
                self.innovation_input[_] = 0
            elif self.technology_state[_] == len(self.technology_level) - 1:
                # print(f'company {_} has reached the maximum technology state {self.technology_state[_]} with innovation input {self.innovation_input[_]}')
                self.now_capital[_] = self.now_capital[_] + self.innovation_input[_]
                self.innovation_input[_] = 0
            
        return self.technology_state
    
        
     
    def revenue(self,q):
        '''
        q is a list of quantity from different firms

        give mono's output return revenue
        '''
        return [self.demand_function(sum(q))*q[i] - self.codetotech()[i]*q[i] for i in range(self.num_of_agents)]
    
    def optimal(self):
        '''
        Find the optimal quantity for a given expansion levelb
        '''

        constriants = np.minimum(np.ones(self.num_of_agents)*np.array(utili.zp(self.demand_function)) \
                                 ,utili.cournot(self.codetotech(), self.demand_function))[0] # type: ignore
        # return [utili.maximize(self.revenue, 0, constriants[i]) for i in range(self.num_of_agents)]
        
        return np.asarray(constriants).flatten()
    

    #  科技状态上限 产量上限
    
    
    
    def input_limit(self):
        '''
        your money limit spent on expansion
        '''
        
        levels = self.optimal()
        return levels, levels * self.codetotech() # type: ignore
     
    def session(self,production_costs,choices_actions):
        '''
        e:expansion list\
        s:tech state code\
        given expansion level return what you will get
        '''    
        '''
        翻译 当任何一个公司的扩张投资超过了输入限制时，利润将被计算为最优产量水平下的利润
        。否则，利润将根据实际的扩张投资计算。
        '''
        # if np.array([exp > cap for exp, cap in zip(e, self.now_capital)]).any():
        #     raise ValueError("Insufficient capital")        # 当每一个状态成为成本函数的时候每一个self.c后面要进行call
        # production_costs = np.minimum(np.array(production_costs), np.array(self.now_capital))
        # i:innovation e:expansion s:state k:captial at the beginning

        # [0.0] * self.n 
        # in limits[0] [1] self.optimal()[1] e 
            # if production_costs[i] > limits[1][i]:
            #     # print('surpass!')
            #     outputs[i] = limits[0][i]
            #     production_costs[i] = limits[1][i]
            # else:
        # limits = self.input_limit()          
        outputs = np.zeros(self.num_of_agents) # quantity list
        cash = np.array([]) # profit list
        for i in range(self.num_of_agents):
            # print(choices_actions)
            # print(production_costs[i])
            # print(choices_actions[production_costs[i],i])
            outputs[i] = choices_actions[production_costs[i],i] / self.codetotech()[i]

        total_quantity = sum(outputs)
        demand_fluctuation = np.random.normal(0, self.demand_function(0) * 0.01)        
        for i in range(self.num_of_agents):        
            if self.now_capital[i] > 0:
                cash = np.append(cash, self.demand_function(total_quantity + demand_fluctuation)*outputs[i] - [choices_actions[production_costs[i],i]])
            else:
                cash = np.append(cash, 0)
        
        return outputs, cash



    def update(self,cash,incre_action,actions_costs):
        for j in range(self.num_of_agents):
            
            if incre_action[j] == -1:
                continue
            if self.technology_state[j] < len(self.technology_level)-1:
                # if self.now_capital[j] < actions_costs[incre_action[j], j]:
                    # print("the bug is here!")
                    # print(f'company {j} has capital {self.now_capital[j]} but production cost {actions_costs[incre_action[j], j]}')
                self.innovation_input[j] += self.now_capital[j]-actions_costs[incre_action[j], j]
                self.now_capital[j] = cash[j]
            else:
                self.now_capital[j] = cash[j] + self.innovation_input[j]
                self.innovation_input[j] = 0
                # 还可以继续进步
                # print(f'company {j} has innovation input {self.innovation_input[j]} with expansion {production_costs[j]} and capital {self.now_capital[j]}')
        # print(f'capital: {self.now_capital}, R&D: {self.innovation_input}')


    def isbankrupt(self,i):
        if self.now_capital[i] < 0:
        # 破产了 也可以允许一些负债
            self.now_capital[i] = 0
            self.innovation_input[i] = 0
            # print("company {} declares bankrupt".format(i))
            return True
        return False    
        # print(self.now_capital)
        #print('summary: ')
        #print(f'Tech:{self.s},profit: {profit}, capital: {self.now_capital},R&D:{self.i},Expansion:{e}')
        
        # 每一期应当选择将当期初始资本全部投入
        
        # i = -e+self.now_capital
  
    def codetotech(self):
        '''
        transform state code into tech level, you can change different funcs here
        '''
        return np.array([self.technology_level[i] for i in self.technology_state])



