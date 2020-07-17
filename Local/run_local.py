# to do: args for local/global method 
import numpy as np
from functools import partial
from .bump import deflation, deflation_der
from .newton import newton
from .scipy_minimize import scipy_minimize
import os
import dask import dask.distributed

def in_bounds(x, bounds):
    if (bounds[:,1]-x > 0).all() and (bounds[:,0] - x < 0).all():
        return True
    return False

def already_found(x, other_x, r):
    dist2 = np.sum(np.power(x-other_x,2),1)
    if (dist2<r).any():
        return True
    return False

def modified_jac(x, grad, minima, r2, alpha):
    j = grad(x)
    defl = deflation(x, minima, r2, alpha)
    return j*defl

def modified_hess(x, grad, hess, minima, r2, alpha):
    h = hess(x)
    j = grad(x)
    defl = deflation(x, minima, r2, alpha)
    defl_der = deflation_der(x, minima, r2, alpha)
    return h*defl + np.outer(defl_der, j)

def run_local(hgdl):
    bound_check = partial(in_bounds, bounds=hgdl.bounds)
    for i in range(hgdl.max_local):
        new_minima = np.empty((0, hgdl.k))
        num_none = 0
        jac = partial(modified_jac, grad=hgdl.grad,
                minima=hgdl.results.minima_x, r2=hgdl.r**2, alpha=hgdl.alpha)
        hess = partial(modified_hess, grad=hgdl.grad,
                hess=hgdl.hess, minima=hgdl.results.minima_x, r2=hgdl.r**2, alpha=hgdl.alpha)

        if hgdl.local_method == 'my_newton':
            func = partial(newton,
                    func=hgdl.func, jac=jac,
                    hess=hess, in_bounds=bound_check)
        elif hgdl.local_method == "scipy":
            func = partial(scipy_minimize,
                    func=hgdl.func, jac=jac,
                    hess=hess, *hgdl.local_args,
                    **hgdl.local_kwargs)
        else:
            raise NotImplementedError("local method not understood")
        futures = hgdl.client.map(func, hgdl.x0)
        for f in dask.distributed.as_completed(futures):
            if f.exception() is not None:
                num_none += 1
                continue
            else:
                res = f.result()

            if num_none / hgdl.num_individuals > .4:
                break
            if not res["success"]:
                num_none += 1
            elif not hgdl.in_bounds(res["x"]):
                num_none += 1
            elif already_found(res["x"], new_minima, hgdl.r**2):
                num_none += 1
            else:
                new_minima = np.append(new_minima, res["x"].reshape(1,-1), 0)
        hgdl.results.update_minima(new_minima)
