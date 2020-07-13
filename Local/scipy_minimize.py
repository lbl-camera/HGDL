from scipy.optimize import minimize
def scipy_minimize(x, func, *args, **kwargs):
    return minimize(fun=func, x0=x, *args, **kwargs)

