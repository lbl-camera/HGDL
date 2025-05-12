import numpy as np
from distributed import get_client
from loguru import logger
from scipy.optimize import minimize

from . import bump_function as defl
from .. import misc
from .dNewton import DNewton as DNewton
import warnings


def run_local(d, optima, x0):
    x_defl, f_defl, radii = optima.get_deflation_points(len(optima.list))
    return run_local_optimizer(d, x0, x_defl, radii)


###########################################################################
def run_local_optimizer(d, x0, x_defl=[], radii=[]):
    """
    this function runs a deflated local methos for
    all the walkers.
    The loop below goes over every walker
    input:
        2d numpy array of initial positions
        2d numpy array of positions of deflations (optional, default = [])
    return:
        optima_locations, func values, gradient norms, eigenvalues, local_success(bool)
    """
    dim = d.dim
    number_of_walkers = d.number_of_walkers

    if len(x0) < number_of_walkers:
        x0 = np.row_stack([x0, misc.random_population(d.bounds, number_of_walkers - len(x0))])

    client = get_client()
    tasks = []
    bf = client.scatter(d, workers=d.workers["walkers"])
    for i in range(min(len(x0), number_of_walkers)):
        logger.debug("Worker ", i, " submitted")
        worker = d.workers["walkers"][(int(i - ((i // number_of_walkers) * number_of_walkers)))]
        data = {"d": bf, "x0": x0[i], "x_defl": x_defl, "radius": radii}
        tasks.append(client.submit(local_method, data, workers=worker))

    results = client.gather(tasks)
    number_of_walkers = len(tasks)
    x = np.empty((number_of_walkers, dim))
    f = np.empty((number_of_walkers))
    g = np.empty((number_of_walkers, dim))
    eig = np.empty((number_of_walkers, dim))
    r = np.empty((number_of_walkers))
    local_success = np.empty((number_of_walkers), dtype=bool)

    for i in range(len(tasks)):
        x[i], f[i], g[i], eig[i], r[i], local_success[i] = results[i]
        for j in range(i):
            if np.linalg.norm(np.subtract(x[i], x[j])) < r[i] and local_success[j] == True:
                logger.warning("points converged too close to each other in HGDL; point removed")
                local_success[i] = False
        for j in range(len(x_defl)):
            if np.linalg.norm(np.subtract(x[i], x_defl[j])) < radii[j] and all(g[i] < 1e-5):
                logger.warning("local method converged within 2 x radius of a deflated position in HGDL")
                local_success[i] = False
    return x, f, g, eig, r, local_success


def local_method(data, method="dNewton"):
    from functools import partial
    d = data["d"]
    x0 = np.array(data["x0"])
    e = np.inf
    local_success = False
    tol = d.tolerance
    x_defl = data["x_defl"]
    r_defl = data["radius"]
    bounds = d.bounds
    max_iter = d.local_max_iter
    args = d.args
    method = d.local_optimizer
    constr = d.constr
    # augment grad, hess
    grad = partial(defl.deflated_grad, grad_func=d.grad, x_defl=x_defl, radius=r_defl)
    hess = partial(defl.deflated_hess, grad_func=d.grad, hess_func=d.hess, x_defl=x_defl, radius=r_defl)

    # call local methods
    if method == "dNewton":
        x, f, g, eig, local_success = DNewton(d.func, grad, hess, bounds, x0, max_iter, tol, *args)
        if np.linalg.norm(g) < 1e-6 and np.min(eig) > 1e-6:
            local_success = True
            r = 1. / np.min(eig)
        else:
            eig = np.array([0.0])
            r = 0.0

    elif type(method) == str:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(d.func, x0, args=args, method=method, jac=grad, hess=hess,
            bounds=bounds, constraints=constr, tol = tol, options={"disp": False})
        x = res["x"]
        f = res["fun"]
        g = res["jac"]
        eig = np.linalg.eig(hess(x, *args))[0]

        if np.linalg.norm(g) < 1e-6 and np.min(eig) > 1e-6:
            local_success = True
            r = 1. / np.min(eig)
        else:
            eig = np.array([0.0])
            r = 0.0


    elif callable(method):
        res = method(d.func, grad, hess, bounds, x0, *args)
        x = res["x"]
        f = res["fun"]
        g = res["jac"]
        if np.linalg.norm(g) < 1e-6 and np.min(eig) > 1e-6:
            local_success = True
            eig = np.linalg.eig(hess(x, *args))[0]
            r = 1. / np.min(eig)
        else:
            eig = np.array([0.0])
            r = 0.0

    else:
        raise Exception("no local method specified")

    return x, f, g, np.real(eig), np.abs(r), local_success
###########################################################################
