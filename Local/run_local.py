from .newton import newton
import numpy as np
from .bump import deflation, deflation_der
from functools import partial
from scipy.optimize import minimize

def already_found(x, other_x, r):
    dist2 = np.sum(np.power(x-other_x,2),1)
    if (dist2<r).any():
        return True
    return False

def modified_jac(x, hgdl):
    j = hgdl.grad(x)
    defl = deflation(x, hgdl.results.minima_x, hgdl.r**2, hgdl.alpha)
    return j*defl

def modified_hess(x, hgdl):
    h = hgdl.hess(x)
    j = hgdl.grad(x)
    defl = deflation(x, hgdl.results.minima_x, hgdl.r**2, hgdl.alpha)
    defl_der = deflation_der(x, hgdl.results.minima_x, hgdl.r**2, hgdl.alpha)
    return h*defl + np.outer(defl_der, j)

def run_local(hgdl):

    for i in range(hgdl.max_local):
        new_minima = np.empty((0, hgdl.k))
        num_none = 0
        jac = partial(modified_jac, hgdl=hgdl)
        hess = partial(modified_hess, hgdl=hgdl)

        if hgdl.local_method == 'my_newton':
            func = newton
        elif hgdl.local_method == "scipy":
            func = lambda x: minimize(fun=hgdl.func, x0=x, jac=jac)
        else:
            raise NotImplementedError("local method not understood")

        for j in range(len(hgdl.x0)):
            try:
                res = func(hgdl.x0[i])
            except NotImplementedError:
                num_none += 1
                continue

            if not res["success"]:
                num_none += 1
            elif not hgdl.in_bounds(res["x"], hgdl.bounds):
                num_none += 1
            elif already_found(res["x"], new_minima, hgdl.r**2):
                num_none += 1
            else:
                new_minima = np.append(new_minima, res["x"].reshape(1,-1), 0)

            if num_none / hgdl.num_individuals > 0.4:
               break
        hgdl.results.update_minima(new_minima)
