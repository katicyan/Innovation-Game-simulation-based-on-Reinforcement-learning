# 需要写一个在给定不同公司生产量确定市场价格的函数

class market:
    def __init__(self,a:float,b:float,n:int):
        '''
        n firms in this market

        '''
        self.a = a
        self.b = b
        self.n = n

    def demand_function(self,q):
        return self.b - self.a*q


    def price(self,Q):
        '''
        sum of Q has to be under zero point 
        '''
        
        if sum(Q) > (self.b/self.a):
            
            raise ValueError("Oversupply!")
        else:
            return self.demand_function(sum(Q))
    