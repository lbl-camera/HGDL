from scipy.optimize import minimize
def scipy_minimize(x, func, jac):
    return minimize(fun=func, x0=x, jac=jac)

