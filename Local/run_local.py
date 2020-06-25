import numpy as np
from functools import partial
from multiprocessing import Pool

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

    for i in range(hgdl.max_local):
        new_minima = np.empty((0, hgdl.k))
        num_none = 0
        jac = partial(modified_jac, hgdl=hgdl)
        hess = partial(modified_hess, hgdl=hgdl)

        if hgdl.local_method == 'my_newton':
            func = partial(newton,
                    func=hgdl.func, jac=jac, hess=hess, in_bounds=hgdl.in_bounds)
        elif hgdl.local_method == "scipy":
            func = partial(scipy_minimize,
                    func=hgdl.func, jac=jac)
        else:
            raise NotImplementedError("local method not understood")

        if hgdl.num_workers != 1:
            workers = Pool(processes=hgdl.num_workers)
            iterable = workers.imap_unordered(func, hgdl.x0)
        else:
            iterable = (func(z) for z in hgdl.x0)
        while num_none / hgdl.num_individuals < .4:
            try:
                res = next(iterable)
            except NotImplementedError:
                num_none += 1
                continue
            except StopIteration:
                break
            except:
                raise
            if not res["success"]:
                num_none += 1
            elif not hgdl.in_bounds(res["x"]):
                num_none += 1
            elif already_found(res["x"], new_minima, hgdl.r**2):
                num_none += 1
            else:
                new_minima = np.append(new_minima, res["x"].reshape(1,-1), 0)
        if hgdl.num_workers != 1: workers.close()
        hgdl.results.update_minima(new_minima)
