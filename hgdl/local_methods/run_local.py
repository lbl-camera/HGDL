# to do: args for local/global method 
import numpy as np
from functools import partial
from .newton import newton
from .scipy_minimize import scipy_minimize
import dask
import dask.distributed

def in_bounds(x, bounds):
    if (bounds[:,1]-x > 0).all() and (bounds[:,0] - x < 0).all():
        return True
    return False

def already_found(x, other_x, r):
    dist2 = np.sum(np.power(x-other_x,2),1)
    if (dist2<r).any():
        return True
    return False


def run_local(info):
    if info.use_dask_map:
        client = dask.distributed.get_client()
    for i in range(info.max_local):
        new_minima = np.empty((0, info.k))
        num_none = 0
        if info._hess is not None:
            hess = info.hess
        else:
            hess = info._hess

        # my newton doesn't support using func for func+grad
        if info.local_method == 'my_newton':
            minimizer = partial(newton,
                    func=info.func, jac=info.grad,
                    hess=hess, in_bounds=info.in_bounds)
        elif info.local_method == "scipy":
            minimizer = partial(scipy_minimize,
                func=info.func, jac=info.grad,
                hess=hess, bounds=info.bounds,
                *info.local_args, **info.local_kwargs)
        else:
            raise NotImplementedError("local method not understood")
        if not info.use_dask_map:
            iterable = (minimizer(z) for z in info.x0)
        else:
            futures = client.map(minimizer, info.x0)
            iterable = (a.result() for a in dask.distributed.as_completed(futures))
        for i, res in enumerate(iterable):
            if num_none / info.num_individuals > .4:
                break
            if not res["success"]:
                num_none += 1
            elif not info.in_bounds(res["x"]):
                num_none += 1
            elif already_found(res["x"], new_minima, info.r**2):
                num_none += 1
            else:
                new_minima = np.append(new_minima, res["x"].reshape(1,-1), 0)
        if info.use_dask_map:
            client.cancel(futures)
    return new_minima
