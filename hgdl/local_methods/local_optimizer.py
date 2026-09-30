import warnings
from functools import partial

import numpy as np
from loguru import logger
from scipy.optimize import minimize

from . import bump_function as defl
from .dNewton import DNewton as DNewton


###########################################################################
def collect_results(results, dim, x_defl=(), radii=()):
    """
    Stack the walker results of one epoch and reject walkers that converged
    too close to each other or into an already deflated region.

    input:
        results: list of local_method return tuples, one per walker
        dim: dimensionality of the domain
        x_defl, radii: deflation points and radii the walkers ran with
    return:
        optima_locations, func values, gradients, eigenvalues, radii, local_success(bool)
    """
    number_of_walkers = len(results)
    x = np.empty((number_of_walkers, dim))
    f = np.empty((number_of_walkers))
    g = np.empty((number_of_walkers, dim))
    eig = np.empty((number_of_walkers, dim))
    r = np.empty((number_of_walkers))
    local_success = np.empty((number_of_walkers), dtype=bool)

    for i in range(number_of_walkers):
        x[i], f[i], g[i], eig[i], r[i], local_success[i] = results[i]
        for j in range(i):
            if np.linalg.norm(np.subtract(x[i], x[j])) < r[i] and local_success[j] == True:
                logger.warning("points converged too close to each other in HGDL; point removed")
                local_success[i] = False
        for j in range(len(x_defl)):
            if np.linalg.norm(np.subtract(x[i], x_defl[j])) < radii[j] and all(np.abs(g[i]) < 1e-5):
                logger.warning("local method converged within 2 x radius of a deflated position in HGDL")
                local_success[i] = False
    return x, f, g, eig, r, local_success


# A deflation radius is at most this fraction of the domain diagonal, so a single
# badly conditioned optimum (tiny Hessian eigenvalue) cannot deflate the whole domain.
MAX_RADIUS_FRACTION = 0.1

# iteration limit of dNewton when the user leaves local_max_iter=None; scipy methods
# then keep their own default maxiter
DNEWTON_DEFAULT_MAX_ITER = 1000


def local_method(x0, problem, x_defl=(), radius=()):
    """
    One walker: a deflated local optimization from x0. Runs as a dask task.
    `problem` is the hgdl.problem.Problem scattered to the workers.

    The walker optimizes the deflated problem, but whether it found an acceptable
    point, and its deflation radius, are judged on the true (undeflated,
    symmetrized) Hessian at the result: in mode "minimization" only strict minima
    are accepted, in mode "stationary_points" any non-degenerate stationary point.
    """
    d = problem
    x0 = np.array(x0)
    tol = d.tolerance
    bounds = d.bounds
    # stacked once per walker: the deflation set is fixed during a local optimization
    x_defl = np.asarray(x_defl, dtype=float).reshape(-1, len(bounds))
    r_defl = np.asarray(radius, dtype=float)
    max_iter = d.local_max_iter
    args = d.args
    method = d.local_optimizer
    constr = d.constraints
    # augment grad, hess
    grad = partial(defl.deflated_grad, grad_func=d.grad, x_defl=x_defl, radius=r_defl)
    hess = partial(defl.deflated_hess, grad_func=d.grad, hess_func=d.hess, x_defl=x_defl, radius=r_defl)

    # call local methods
    if method == "dNewton":
        dnewton_max_iter = DNEWTON_DEFAULT_MAX_ITER if max_iter is None else max_iter
        x, f, g, _, _ = DNewton(d.func, grad, hess, bounds, x0, dnewton_max_iter, tol, *args,
                                saddle_free=(d.mode == "minimization"))
    elif type(method) == str:
        options = {"disp": False}
        if max_iter is not None:
            options["maxiter"] = max_iter
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(d.func, x0, args=args, method=method, jac=grad, hess=hess,
                           bounds=bounds, constraints=constr, tol=tol,
                           options=options)
        x, f, g = res["x"], res["fun"], res["jac"]
    elif callable(method):
        res = method(d.func, grad, hess, bounds, x0, *args)
        x, f, g = res["x"], res["fun"], res["jac"]
    else:
        raise Exception("no local method specified")

    h = np.asarray(d.hess(x, *args), dtype=float)
    eig = np.linalg.eigvalsh(0.5 * (h + h.T))
    curvature = np.min(eig) if d.mode == "minimization" else np.min(np.abs(eig))
    local_success = bool(np.all(np.isfinite(g)) and np.linalg.norm(g) < 1e-6 and curvature > 1e-6)
    if local_success:
        max_radius = MAX_RADIUS_FRACTION * np.linalg.norm(bounds[:, 1] - bounds[:, 0])
        r = min(1. / curvature, max_radius)
    else:
        eig = np.array([0.0])
        r = 0.0
    return x, f, g, eig, r, local_success
###########################################################################
