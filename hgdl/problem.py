from functools import partial

import numpy as np

from .local_methods.kkt import normalize_constraints


class Problem:
    """
    Everything a walker needs to run a local optimization.

    One instance is scattered to every worker once per ``HGDL.optimize()`` call and
    passed to each walker task by reference, so the objective (and any data it
    carries) is not re-sent with every task. It is serialized with dask's
    cloudpickle, so closures work, but it must not reference a dask client,
    future, lock or thread (e.g. a bound method of HGDL).
    """

    def __init__(self, func, grad, hess, bounds, args, local_optimizer, local_max_iter,
                 tolerance, constraints, mode="minimization", local_time_limit=None, run_id=None):
        self.mode = mode
        # seconds after which a walker gives up, and the id under which the run is marked
        # as stopped on the workers (both checked between iterations)
        self.local_time_limit = local_time_limit
        self.run_id = run_id
        self.func = func
        self.grad = grad
        self.hess = hess
        self.bounds = bounds
        self.dim = len(bounds)
        self.args = args
        self.local_optimizer = local_optimizer
        self.local_max_iter = local_max_iter
        self.tolerance = tolerance
        self.constraints = constraints
        # the same constraints as blocks lb <= c(x) <= ub, for the KKT test of the results
        self.constraint_blocks = normalize_constraints(constraints)


def approximate_hessian(x, *args, grad_func=None):
    """First-order forward-difference Hessian built from the gradient."""
    len_x = len(x)
    hess = np.zeros((len_x, len_x))
    epsilon = 1e-6
    grad_x = grad_func(x, *args)
    for i in range(len_x):
        x_temp = np.array(x)
        x_temp[i] = x_temp[i] + epsilon
        hess[i, i:] = ((grad_func(x_temp, *args) - grad_x) / epsilon)[i:]
    return hess + hess.T - np.diag(np.diag(hess))


def hessian_or_approximation(hess, grad):
    """The user's Hessian, or a picklable forward-difference approximation of it."""
    if hess:
        return hess
    return partial(approximate_hessian, grad_func=grad)
