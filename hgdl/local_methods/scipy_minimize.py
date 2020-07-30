from scipy.optimize import minimize
def scipy_minimize(x, func, *args, **kwargs):
    try:
        return minimize(fun=func, x0=x, *args, **kwargs)
    except NotImplementedError:
        return {"success":False}
