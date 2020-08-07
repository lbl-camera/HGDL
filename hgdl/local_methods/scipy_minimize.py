from scipy.optimize import minimize
def scipy_minimize(x, func, *args, **kwargs):
    if 'options' not in kwargs:
        kwargs['options'] = {}
    kwargs['options']['ftol'] = 0
    try:
        return minimize(fun=func, x0=x, *args, **kwargs)
    except NotImplementedError:
        return {"success":False}
