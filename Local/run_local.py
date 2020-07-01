# to do: args for local/global method 
import numpy as np
from functools import partial
import dask.config
import dask.distributed
from .bump import deflation, deflation_der
from .newton import newton
from .scipy_minimize import scipy_minimize

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
    #dask.config.set(scheduler="processes")
    #cluster = dask.distributed.LocalCluster(dashboard_address=0)
    workers = dask.distributed.Client(dashboard_address=0)#cluster)

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
                    func=hgdl.func, jac=jac)
        else:
            raise NotImplementedError("local method not understood")
        futures = workers.map(func, hgdl.x0)
        for f in dask.distributed.as_completed(futures):
            try:
                res = f.result()
            except NotImplementedError:
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
        #workers.cancel(futures)
        hgdl.results.update_minima(new_minima)
    #workers.shutdown()
