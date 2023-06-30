import numpy as np
from loguru import logger
from .. import misc


def DNewton(func, grad, hess, bounds, x0, max_iter, tol, *args):
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
        # hessian = 0.5 * hessian @ hessian.T
        hessian[abs(hessian) < 1e-16] = 0.
        grad_list.append(np.max(gradient))
        try:
            gamma = np.linalg.solve(hessian, -gradient)
        except Exception as error:
            gamma, a, b, c = np.linalg.lstsq(hessian, -gradient, rcond=None)
        if any(gamma == np.nan) or any(gamma == np.inf): return x, func(x, *args), gradient, \
        np.linalg.eig(hess(x, *args))[0], False
        x += gamma
        e = np.max(abs(gamma))
        logger.debug("dNewton step size: ", e, " max gradient: ", np.max(abs(gradient)))
        if counter > max_iter: return x, func(x, *args), gradient, np.linalg.eig(hess(x, *args))[0], False
        counter += 1
    return x, func(x, *args), gradient, np.linalg.eig(hess(x, *args))[0], True
