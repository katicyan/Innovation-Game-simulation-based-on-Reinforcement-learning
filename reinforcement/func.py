from scipy.optimize import minimize_scalar
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