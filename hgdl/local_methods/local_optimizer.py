import numpy as np
import time

from loguru import logger

import hgdl.misc as misc
import dask.distributed
from distributed import Client, get_client, secede, rejoin, protocol
import dask.distributed as distributed
from hgdl.local_methods.dNewton import DNewton
from scipy.optimize import minimize
import hgdl.local_methods.bump_function as defl


def run_local(d,optima,x0):
    x_defl,f_defl = optima.get_deflation_points(len(optima.list))
    x,f,grad_norm,eig,local_success = run_local_optimizer(d,x0,x_defl)
    if not np.any(local_success) and len(optima.list["x"]) == 0:
        local_success[:] = True
    optima.fill_in_optima_list(x,f,grad_norm,eig,local_success)
    return optima
###########################################################################
def run_local_optimizer(d,x0,x_defl = []):
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
        x0 = np.row_stack([x0,misc.random_population(d.bounds,number_of_walkers - len(x0))])

    client = get_client()
    tasks = []
    bf = client.scatter(d,workers = d.workers["walkers"])
    for i in range(min(len(x0),number_of_walkers)):
        worker = d.workers["walkers"][(int(i - ((i // number_of_walkers) * number_of_walkers)))]
        data = {"d":bf,"x0":x0[i],"x_defl":x_defl}
        tasks.append(client.submit(local_method,data,workers = worker))
        #print("HGDL says: local optimizer submitted to worker ", worker," ; method: ", d.local_optimizer," tol: ",d.tolerance, flush = True)
    results = client.gather(tasks)
    number_of_walkers = len(tasks)
    x = np.empty((number_of_walkers, dim))
    f = np.empty((number_of_walkers))
    grad_norm = np.empty((number_of_walkers))
    eig = np.empty((number_of_walkers,dim))
    local_success = np.empty((number_of_walkers), dtype = bool)
    for i in range(len(tasks)):
        x[i],f[i],grad_norm[i],eig[i],local_success[i] = results[i]
        client.cancel(tasks[i])
        for j in range(i):
            #exchange for function def too_close():
            if np.linalg.norm(np.subtract(x[i],x[j])) < 2.0 * d.radius and local_success[j] == True:
                logger.warning("points converged too close to each other in HGDL; point removed")
                local_success[j] = False; break
        for j in range(len(x_defl)):
            if np.linalg.norm(np.subtract(x[i],x_defl[j])) < 2.0 * d.radius\
            and grad_norm[i] < 1e-5:
                logger.warning("local method converged within 2 x radius of a deflated position in HGDL")
                local_success[i] = False
                #print("point found: ",x[i]," deflated point: ",x_defl[j])
                #print("gradient at the point: ",grad_norm[i])
                #print("distance between the points: ",np.linalg.norm(np.subtract(x[i],x_defl[j])))
                #print("--")
    return x, f, grad_norm, eig, local_success
###########################################################################

def local_method(data, method = "dNewton"):
    from functools import partial
    d = data["d"]
    x0 = np.array(data["x0"])
    e = np.inf
    local_success = True
    tol = d.tolerance
    x_defl = data["x_defl"]
    bounds = d.bounds
    max_iter = d.local_max_iter
    args = d.args
    method = d.local_optimizer
    #augment grad, hess
    grad = partial(defl.deflated_grad, grad_func = d.grad, x_defl = x_defl, radius = d.radius)
    if callable(d.hess):
        hess = partial(defl.deflated_hess, grad_func = d.grad,
                       hess_func = d.hess, x_defl = x_defl, radius = d.radius)
    else:
        hess = d.hess
    #call local methods
    if method == "dNewton":
        x,f,g,eig,local_success = DNewton(d.func,grad,hess,bounds,x0,max_iter,tol,*args)
    elif type(method) == str:
        res = minimize(d.func,x0,args = args,method = method,jac = grad, hess = hess,
                       bounds = bounds, constraints = d.constr, options = {"disp":False})
        x = res["x"]
        f = res["fun"]
        g = np.linalg.norm(res["jac"])
        local_success = res["success"]
        eig = np.ones(x.shape) * np.nan
    elif callable(method):
        res = method(d.func,grad,hess,bounds,x0,*args)
        x = res["x"]
        f = res["fun"]
        g = np.linalg.norm(res["jac"])
        local_success = res["success"]
        eig = np.ones(x.shape) * np.nan
    else: raise Exception("no local method specified")
    return x,f,g,eig,local_success
###########################################################################


