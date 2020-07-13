# to do: args for local/global method 
import numpy as np
from functools import partial
from .bump import deflation, deflation_der
from .newton import newton
from .scipy_minimize import scipy_minimize
import os
import dask
import dask.distributed

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
            func = partial(newton,
                    func=hgdl.func, jac=jac,
                    hess=hess, in_bounds=hgdl.in_bounds)
        elif hgdl.local_method == "scipy":
            func = partial(scipy_minimize,
                    func=hgdl.func, jac=jac,
                    hess=hess, *hgdl.local_args,
                    **hgdl.local_kwargs)
        else:
            raise NotImplementedError("local method not understood")
        #client = dask.distributed.Client(dashboard_address=0)
        #def inc(x):
        #    return x + 1
        #results = client.map(inc, range(4))
        #for f in dask.distributed.as_completed(results): print(f.result())
        #""" 
        for z in hgdl.x0:
            try:
                res = func(z)
            except:
                num_none += 1
                continue
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
        #client.shutdown()
        hgdl.results.update_minima(new_minima)
