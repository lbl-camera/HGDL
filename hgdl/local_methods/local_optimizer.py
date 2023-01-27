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
    return run_local_optimizer(d,x0,x_defl)
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
        logger.debug("Worker ",i," submitted")
        #print("Worker ",i," submitted", flush = True)
        worker = d.workers["walkers"][(int(i - ((i // number_of_walkers) * number_of_walkers)))]
        data = {"d":bf,"x0":x0[i],"x_defl":x_defl}
        tasks.append(client.submit(local_method,data,workers = worker))

    results = client.gather(tasks)
    number_of_walkers = len(tasks)
    x = np.empty((number_of_walkers, dim))
    f = np.empty((number_of_walkers))
    Lg = np.empty((number_of_walkers, dim))
    eig = np.empty((number_of_walkers,dim))
    local_success = np.empty((number_of_walkers), dtype = bool)
    messages = np.empty((number_of_walkers), dtype = str)

    for i in range(len(tasks)):
        x[i],f[i],Lg[i],eig[i],local_success[i], messages[i] = results[i]
        client.cancel(tasks[i])
        for j in range(i):
            if np.linalg.norm(np.subtract(x[i],x[j])) < 2.0 * d.radius and local_success[j] == True:
                logger.warning("points converged too close to each other in HGDL; point removed")
                local_success[j] = False; break
        for j in range(len(x_defl)):
            if np.linalg.norm(np.subtract(x[i],x_defl[j])) < 2.0 * d.radius and all(Lg[i] < 1e-5):
                logger.warning("local method converged within 2 x radius of a deflated position in HGDL")
                local_success[i] = False
    return x, f, Lg, eig, local_success, messages

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
    Lgrad = partial(defl.deflated_grad, grad_func = d.Lgrad, x_defl = x_defl, radius = d.radius)
    if callable(d.Lhess):
        Lhess = partial(defl.deflated_hess, grad_func = d.Lgrad,
                       hess_func = d.Lhess, x_defl = x_defl, radius = d.radius)
    else:
        Lhess = d.Lhess

    #call local methods
    if method == "dNewton":
        x,L,Lg,eig,local_success,message = DNewton(d.L,Lgrad,Lhess,bounds,x0,max_iter,tol,*args)
        f = d.func(x,*args)

    elif type(method) == str:
        if d.constr: logger.warning("Deflated constraints are only supported by local optimizer 'DNewton'")
        res = minimize(d.func, x0, args = args, method = method, jac = Lgrad, hess = Lhess,
                       bounds = bounds, options = {"disp":False})
        x = res["x"]
        f = res["fun"]
        eig = np.ones(x.shape) * np.nan
        Lg = res["jac"]
        local_success = res["success"]

    elif callable(method):
        res = method(d.func,Lgrad,Lhess,bounds,x0,*args)
        x = res["x"]
        f = res["fun"]
        eig = np.ones(x.shape) * np.nan
        Lg = res["jac"]
        local_success = res["success"]


    else: raise Exception("no local method specified")

    return x,f,Lg,np.real(eig),local_success,message
###########################################################################
