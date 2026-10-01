import numpy as np
from loguru import logger
from .. import misc


def saddle_free_step(hessian, gradient):
    """
    Saddle-free Newton step -|S|^-1 gradient (Dauphin et al., 2014), with S the
    symmetric part of the (deflated) Hessian and |S| its eigenvalues replaced by their
    absolute values, floored so a (near-)singular S still gives a finite step.
    |S| is positive definite, so the step always descends: walkers are pushed away
    from maxima and saddle points. Near a minimum |S| = S, i.e. plain Newton.
    """
    lam, V = np.linalg.eigh(0.5 * (hessian + hessian.T))
    floor = 1e-8 * max(1.0, np.max(np.abs(lam)))
    return -(V @ ((V.T @ gradient) / np.maximum(np.abs(lam), floor)))


def _eigenvalues(hess, x, *args):
    try:
        return np.linalg.eig(hess(x, *args))[0]
    except np.linalg.LinAlgError:
        return np.full(len(x), np.nan)


def DNewton(func, grad, hess, bounds, x0, max_iter, tol, *args, saddle_free=False, should_stop=None):
    """
    Damped Newton on the (deflated) gradient. With saddle_free=False this is plain
    Newton and converges to any stationary point; with saddle_free=True it only
    converges to minima. should_stop is called after every iteration; if it returns
    True the iteration ends unconverged.
    """
    e = np.inf
    gradient = np.ones((len(x0))) * np.inf
    counter = 0
    x = np.array(x0)
    grad_list = []
    while e > tol or np.max(abs(gradient)) > tol:
        x = misc.project_onto_bounds(x, bounds)
        x[abs(x) < 1e-16] = 0.
        gradient = grad(x, *args)
        gradient[abs(gradient) < 1e-16] = 0.
        hessian = hess(x, *args)
        hessian[abs(hessian) < 1e-16] = 0.
        grad_list.append(np.max(gradient))
        if saddle_free:
            try:
                gamma = saddle_free_step(hessian, gradient)
            except np.linalg.LinAlgError:
                gamma = np.full(len(x), np.nan)  # aborts via the finiteness check below
        else:
            try:
                gamma = np.linalg.solve(hessian, -gradient)
            except Exception as error:
                gamma, a, b, c = np.linalg.lstsq(hessian, -gradient, rcond=None)
        if not np.all(np.isfinite(gamma)): return x, func(x, *args), gradient, \
        _eigenvalues(hess, x, *args), False
        x += gamma
        e = np.max(abs(gamma))
        logger.debug("dNewton step size: ", e, " max gradient: ", np.max(abs(gradient)))
        if counter > max_iter: return x, func(x, *args), gradient, _eigenvalues(hess, x, *args), False
        counter += 1
        if should_stop is not None and should_stop():
            return x, func(x, *args), gradient, _eigenvalues(hess, x, *args), False
    return x, func(x, *args), gradient, _eigenvalues(hess, x, *args), True
