# Here is how companies make their decisions after each commerical
# run, then they would reveive different revenue and achieve profit which
# can be considered as capital in the next period.
import numpy as np
import func

class firm:
    
    def __init__(self,beta:float,c:list,k0:float,i=0,s=0):
        self.beta = beta
        self.k0 = k0
        self.c = c # 可能的成本列表 长度代表了科技状态的数量
        #discount rate and initial capital
        self.k = k0
        self.i = i # innovation input
        self.s = s # technology state
    

    
    def tech(self,s):
        self.innovation = 0.1*s / (0.1*s+1)
        return self.innovation
     
    def revenue(self,demand_function,q):
        return demand_function(q)*q - self.c[self.s]*q
    
    def optimal(self,roof):
        # Find the optimal quantity for a given expansion level
        return func.maximize(self.revenue, 0, roof)
    
    

# #  科技状态上限 产量上限
    
    def new_tech(self):
        p = self.tech(self.i)
        m = np.random.uniform(0,1)
        if m < p:
            self.s += 1
            self.i = 0
        if self.s >= len(self.c):
            self.s = len(self.c)-1
        return self.s
    
    def input_limit(self,roof):
        level = self.optimal(roof)[0]
        max = float(self.c[self.s]*level)
        return max
     
    def session(self,e,roof,demand_function):
        '''
        e:expansion
        s:state 三种科技状态
        roof:总产量的上限
        demand_function: 市场需求
        '''
        
        if e>self.k:
            raise ValueError("Insufficient capital")        # 当每一个状态成为成本函数的时候每一个self.c后面要进行call
        

        # i:innovation e:expansion s:state k:captial at the beginning
        level = self.optimal(roof)[0]
        max = float(self.c[self.s]*level)           
        if e > max:
            #print('surpass!')
            q = level
            e = max
            profit = self.optimal(roof)[1]
        else:
            
            q = e / self.c[self.s]
            #print(q)
            profit = demand_function(q)*q - e
        return profit    

    def update(self,e):
        profit = self.session(e)
        if self.s < len(self.c)-1:
            #print('invest in tech')
            self.i += -e+self.k
            self.k = profit
        else:
            self.k = profit + self.k - e
        
        #print('summary: ')
        #print(f'Tech:{self.s},profit: {profit}, capital: {self.k},R&D:{self.i},Expansion:{e}')
        
        # 每一期应当选择将当期初始资本全部投入
        
        # i = -e+self.k
  
            
