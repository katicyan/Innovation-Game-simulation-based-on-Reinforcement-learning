import numpy as np
import utili
def dice(p) -> int:
    if p < 0 or p > 1:
        raise ValueError("Not prob")
    m = np.random.uniform(0,1)
    while p > m:
        return 1
    return 0

def codetocost(s:list,c:list)->list:

    '''
    transform state code into cost, you can change different funcs here
    '''
    return np.array([c[_] for _ in s])








class market:
    def __init__(self,gamma:float,n:int,demand_function,c:list,num_actions:int,k0:list,i:list,s:list):
        '''
        k0 initial capital list\
        demand_function return price\
        c cost list\
        n number of firms\
        i innovation input\
        s technology states code\
        beta discount rate
        '''
        self.k0 = k0
        self.demand_function = demand_function
        self.c = c
        self.n = n
        self.k = k0
        self.i = i
        self.s = s
        self.gamma = gamma
        self.num_states = len(c)
        self.num_actions = num_actions
        self.sc = codetocost(s,c)
    
    
        
     
    def revenue(self,q):
        '''
        q is a list of quantity from different firms

        give mono's output return revenue
        '''
        return [self.demand_function(sum(q))*q[i] - self.sc[i]*q[i] for i in range(self.n)]
    
    def optimal(self):
        '''
        Find the optimal quantity for a given expansion levelb
        '''
        constriants = np.minimum(np.ones(self.n) * utili.zp(self.demand_function), utili.cournot(self.sc, self.demand_function))[0] # type: ignore
        # return [utili.maximize(self.revenue, 0, constriants[i]) for i in range(self.n)]
        
        return np.asarray(constriants).flatten()
    

# #  科技状态上限 产量上限
    
    def new_tech(self):
        '''
        model simplification innovation experienc does not accumulate between\
        different tech state\
        
        s is a list of technology states from different firms

        transform state code into prob of progress, you can change different funcs here
        
     
        '''
        for _ in range(len(self.s)):
            
            p = 0.1*self.i[_] / (0.1*self.i[_]+1)
            print(f'company {_} has innovation input {self.i[_]} with progress prob {p}')
            if dice(p) and self.s[_]<len(self.c)-1:
                self.s[_] += 1
                self.i[_] = 0
        return self.s
    
    
    def input_limit(self):
        '''
        your money limit spent on expansion
        '''
        
        level = self.optimal()[0]
        return level, self.sc*level
     
    def session(self,e:list):
        '''
        e:expansion list\
        s:tech state code\
        given expansion level return what you will get
        '''
        
        if np.array([exp > cap for exp, cap in zip(e, self.k)]).any():
            raise ValueError("Insufficient capital")        # 当每一个状态成为成本函数的时候每一个self.c后面要进行call
        


        # i:innovation e:expansion s:state k:captial at the beginning
        limits = self.input_limit()          
        q = np.zeros(self.n) # quantity list
        # [0.0] * self.n 
        cash = np.array([]) # profit list
        # in limits[0] [1] self.optimal()[1] e 
        for i in range(self.n):
            if e[i] > limits[1][i]:
                # print('surpass!')
                q[i] = limits[0][i]
                e[i] = limits[1][i]
                
            else:
                
                q[i] = e[i] / self.sc[i]
                print(q[i])
        total_quantity = sum(q)        
        for i in range(self.n):        
                cash = np.append(cash, self.demand_function(total_quantity)*q[i] - e[i])
        '''
        翻译 当任何一个公司的扩张投资超过了输入限制时，利润将被计算为最优产量水平下的利润
        。否则，利润将根据实际的扩张投资计算。
        '''
        
        # print(f'Expansion levels: {e}'
        #       f'\nQuantities: {q}'
        #       f'\nProfits: {cash}')
        
        
        # out profit
        return cash



    def update(self,e):
        cash = self.session(e)
        for j in range(self.n):
            if self.s[j] < len(self.c)-1:
                # 还可以继续进步
                self.i[j] += -e[j]+self.k[j]
                self.k[j] = cash[j]
            else:
                self.k[j] = cash[j] + self.k[j] - e[j]
        # print(f'capital: {self.k}, R&D: {self.i}')


    def update_tech(self,e):
        pass

    def update_agent(self):
        for i in range(self.n):
            if self.k[i] < 0:
                # 破产了 也可以允许一些负债
                self.k[i] = 0
                self.s[i] = 0
                self.i[i] = 0

        # print(self.k)
        #print('summary: ')
        #print(f'Tech:{self.s},profit: {profit}, capital: {self.k},R&D:{self.i},Expansion:{e}')
        
        # 每一期应当选择将当期初始资本全部投入
        
        # i = -e+self.k
  
            
