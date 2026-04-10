from scipy.optimize import minimize_scalar
from scipy.optimize import brentq
import numpy as np
import matplotlib.pyplot as plt
# company_code state actionimport numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider
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
        print("interval is too small or too big")
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


def draw_along_time(data):


    pass

def draw_q_surface(Q):
    '''    Q is a 3D array with shape (company, state, action)
    '''
    # If Q shape is (company, state, action) = (10, 7, 10)
    assert Q.ndim == 3 and Q.shape[0] == 10 and Q.shape[1] == 7 and Q.shape[2] == 10
    draw_states = np.arange(Q.shape[1])    # 0..6
    draw_actions = np.arange(Q.shape[2])   # 0..9
    A, S = np.meshgrid(draw_actions, draw_states)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    company_idx = 0  # Default to the first company, can be changed with interact
    Z = Q[company_idx]  # (7, 10)
    surf = ax.plot_surface(A, S, Z, cmap="viridis", edgecolor="k", linewidth=0.3)

    ax.set_xlabel("action")
    ax.set_ylabel("state")
    ax.set_zlabel("Q value")
    ax.set_title(f"Q surface - company {company_idx}")
    fig.colorbar(surf, ax=ax, shrink=0.65, label="Q value")
    plt.tight_layout()
    plt.show()
    interact(
        plot_q_surface,
        company_idx=IntSlider(min=0, max=Q.shape[0]-1, step=1, value=0, description="company")
    )


