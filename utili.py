from scipy.optimize import minimize_scalar
from scipy.optimize import brentq
import numpy as np
def maximize(g, a, b, *args):
    """
    在区
    间[a, b]上最大化函数g。

    我们利用了这样一个事实：在任意区间上，g 的最大值点，同时也是 -g 的最小值点。
    参数元组 args 收集传递给 g 的额外参数。

    返回最大值和最大值点。
    """

    objective = lambda x: -g(x, *args)
    result = minimize_scalar(objective, bounds=(a, b), method='bounded')
    maximizer, maximum = result.x, -result.fun
    return maximizer, maximum

def zp(demand_function, upper=100):
    '''
    zero point of demand function, the upper bound of optimal quantity
    '''

    try:
        # Try to find a root in [0, 100], adjust interval as needed
        zero = brentq(demand_function, 0, upper)
    except ValueError:
        zero = None  # No root found in interval
# if zero is not None else None
    return zero 

def cournot(sc,demand_function):
    '''
    make sure demand_function is linear and decreasing.\
    Cournot competition between two firms, each firm has its own tech state and capital, but
    they share the same demand function and cost function
    
    return a list of optimal quantity for each firm
    '''
    
    cournot_matrix = np.matrix(np.eye(len(sc)) + 1)
    b = demand_function(0)
    a = b - demand_function(1)
    y = (b-np.array(sc)) / a
    
    return cournot_matrix.I @ y
