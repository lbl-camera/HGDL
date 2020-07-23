###local optimizer for hgdl
import numpy as np
import hgdl.misc as misc
import asyncio
def bump_function(x,x0,r = 40.0):
    """
    evaluates the bump function
    x ... 2d numpy array of points
    x0 ... 1d numpy array of location of bump function
    """
    if x.ndim == 1: x = np.array([x])
    x = x[0]
    d = np.linalg.norm(x-x0)
    if d >= r: return 0.0
    else: return np.exp(1.0) * np.exp(-1.0/(1.0-((d/r)**2)))
###########################################################################
def bump_function_gradient(x,x0, r = 40.0):
    if x.ndim == 1: x = np.array([x])
    x = x[0]
    d = np.linalg.norm(x-x0)
    d2= (x - x0)
    if d >= r: return 0.0
    else: return np.exp(1.0) * ((-2.0*d2)/((1.0-((d/r)**2))**2)) * np.exp(-1.0/(1.0-((d/r)**2)))
    return gr
def deflation_function(x,x0):
    if len(x0) == 0: return 1.0
    s = np.array([bump_function(x,x0[i]) for i in range(len(x0))])
    return (1.0/(1.0-sum(s)))
###########################################################################
def deflation_function_gradient(x,x0):
    if len(x0) == 0: return np.zeros((len(x)))
    s1 = np.array([bump_function(x,x0[i]) for i in range(len(x0))])
    s2 = np.array([bump_function_gradient(x,x0[i]) for i in range(len(x0))])
    return (1.0/((1.0-sum(s1))**2))*np.sum(s2)
###########################################################################
async def DNewton(func, grad, hess, x0,x_defl,bounds,tol = 1e-6 ,max_iter = 20, kwargs = {}):
    #w = np.random.rand() * 10.0
    #print("DNewton from point:", x0," started, waits: ", w)
    #await asyncio.sleep(w)
    e = np.inf
    success = True
    counter = 0
    x = np.array(x0)
    while e > tol:
        counter += 1
        if counter >= max_iter or misc.out_of_bounds(x,bounds):
            #print("DNewton from point:", x0," not converged",w)
            return x,func(x, kwargs),e,np.linalg.eig(hessian)[0],False
        gradient = grad(x, kwargs)
        e = np.linalg.norm(gradient)
        hessian = hess(x, kwargs)
        d = deflation_function(x,x0)
        dg = deflation_function_gradient(x,x0)
        gamma = np.linalg.solve(hessian + (np.outer(gradient,dg)/d),-gradient)
        x += gamma
        #print("current position: ",x,"epsilon: ",e)
    #print("DNewton from point:", x0," converged to ",x, "after waiting: ", w)
    return x,func(x, kwargs),e,np.linalg.eig(hessian)[0], success
