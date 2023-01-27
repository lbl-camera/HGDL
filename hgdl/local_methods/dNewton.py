import numpy as np
import hgdl.misc as misc
import hgdl.local_methods.bump_function as defl
import dask.distributed as distributed
from loguru import logger


def DNewton(func,grad,hess,bounds,x0,max_iter,tol,*args):
    e = np.inf
    gradient = np.ones((len(x0))) * np.inf
    counter = 0
    x = np.array(x0)
    grad_list = []
    while e > tol or np.max(abs(gradient)) > tol:
        x = misc.project_onto_bounds(x,bounds)
        x[abs(x)<1e-16] = 0.
        gradient = grad(x,*args)
        gradient[abs(gradient)<1e-16] = 0.
        if hess: hessian  = hess(x,*args)
        else: hessian = approximate_hessian(x,grad,*args)
        hessian[abs(hessian) < 1e-16] = 0.
        grad_list.append(np.max(gradient))
        try: gamma = np.linalg.solve(hessian,-gradient)
        except Exception as error: gamma,a,b,c = np.linalg.lstsq(hessian,-gradient,rcond=None)
        if any(gamma == np.nan) or any(gamma == np.inf): return x,func(x, *args),gradient,np.linalg.eig(hess(x, *args))[0], False
        x += gamma
        e = np.max(abs(gamma))
        logger.debug("dNewton step size: ", e, " max gradient: ",np.max(abs(gradient)))
        #print("dNewton step size: ", e, " max gradient: ",np.max(abs(gradient)))
        if counter > max_iter: return x,func(x, *args),gradient,np.linalg.eig(hess(x, *args))[0], False, "max_iter reached"
        counter += 1
    return x,func(x, *args),gradient,np.linalg.eig(hess(x, *args))[0], True, "converged"



def approximate_hessian(x, grad, *args):
    ##implements a first-order approximation
    len_x = len(x)
    hess = np.zeros((len_x,len_x))
    epsilon = 1e-6
    grad_x = grad(x, *args)
    for i in range(len_x):
        x_temp = np.array(x)
        x_temp[i] = x_temp[i] + epsilon
        hess[i,i:] = ((grad(x_temp,*args) - grad_x)/epsilon)[i:]
    return hess + hess.T - np.diag(np.diag(hess))







