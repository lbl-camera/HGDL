import numpy as np
import time
import hgdl.misc as misc
import dask.distributed
from distributed import Client, get_client, secede, rejoin, protocol
import dask.distributed as distributed
from hgdl.local_methods.dNewton import DNewton
from scipy.optimize import minimize
import hgdl.local_methods.bump_function as defl


def run_local(d,optima,x0):
    x_init = np.array(x0)
    x_defl,f_defl = optima.get_deflation_points(len(optima.list))
    x,f,grad_norm,eig,success = run_local_optimizer(d,x0,x_defl)
    optima.fill_in_optima_list(x,f,grad_norm,eig,success)
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
        optima_locations, func values, gradient norms, eigenvalues, success(bool)
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
        print("HGDL says: local optimizer submitted to worker ", worker," ; method: ", d.local_optimizer)
    results = client.gather(tasks)
    number_of_walkers = len(tasks)
    x = np.empty((number_of_walkers, dim))
    f = np.empty((number_of_walkers))
    grad_norm = np.empty((number_of_walkers))
    eig = np.empty((number_of_walkers,dim))
    success = np.empty((number_of_walkers))
    for i in range(len(tasks)):
        x[i],f[i],grad_norm[i],eig[i],success[i] = results[i]
        client.cancel(tasks[i])
        for j in range(i):
            #exchange for function def too_close():
            if np.linalg.norm(np.subtract(x[i],x[j])) < 2.0 * d.radius and success[j] == True:
                print("CAUTION: points too close to each other in HGDL; point removed")
                success[j] = False; break
        for j in range(len(x_defl)):
            if np.linalg.norm(np.subtract(x[i],x_defl[j])) < 2.0 * d.radius\
            and grad_norm[i] < 1e-5:
                print("CAUTION: local method converged within 2 x radius of a deflated position in HGDL")
                success[i] = False
                print(x[i],x_defl[j])
                print(grad_norm[i])
                print("--")
    return x, f, grad_norm, eig, success
###########################################################################

from functools import *
def local_method(data, method = "dNewton"):
    d = data["d"]
    x0 = np.array(data["x0"])
    e = np.inf
    success = True
    tol = 1e-6
    x_defl = data["x_defl"]
    bounds = d.bounds
    max_iter = d.local_max_iter
    args = d.args
    method = d.local_optimizer
    #augment grad, hess
    grad = partial(defl.deflated_grad, grad_func = d.grad, x_defl = x_defl, radius = d.radius)
    if d.hess is callable:
        hess = partial(defl.deflated_hess, grad_func = d.grad,
        hess_func = d.hess, x_defl = x_defl, radius = d.radius)
    else:
        hess = d.hess
    #call local methods
    if method == "dNewton":
        x,f,g,eig,success = DNewton(d.func,grad,hess,bounds,x0,max_iter,*args)
    elif type(method) == str:
        res = minimize(d.func,x0,args = args,method = method,jac = grad,bounds = bounds, constraints = d.constr)
        x = res["x"]
        f = res["fun"]
        g = np.linalg.norm(res["jac"])
        success = res["success"]
        eig = np.ones(x.shape) * np.nan
    elif method is callable:
        res = method(d.func,grad,hess,bounds,x0,*args)
        x = res["x"]
        f = res["fun"]
        g = np.linalg.norm(res["jac"])
        success = res["success"]
        eig = np.ones(x.shape) * np.nan
    else: raise Exception("no local method specified")
    #return
    return x,f,g,eig,success
###########################################################################


