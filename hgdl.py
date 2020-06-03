# coding: utf-8
# # HGDL
#     * Hybrid - uses both local and global optimization
#     * G - uses global optimizer
#     * D - uses deflation
#     * L - uses local extremum localMethod
#  imports

import numpy as np
import numba as nb
from math import ceil
from psutil import cpu_count
from functools import partial
from multiprocessing import Pool

from .newton import newton, in_bounds

def random_sample(N,k,bounds):
    sample = np.random.random((N, k))
    sample *= bounds[:,1] - bounds[:,0]
    sample += bounds[:,0]
    return sample

def alreadyFound(newMinima, oldMinima, radius_squared, k):
    """
    check if the new minimum is within range of the old ones
    """
    c = oldMinima[:,:k] - newMinima[0,:k]
    return (np.sum(c*c,1)<radius_squared).any()

# This is my implementation of a genetic algorithm
def GeneticStep(X, y, bounds):
    """
    Input:
    X is the individuals - points on a surface
    y is the performance - f(X)
    Notes:
    the children can be outside of the bounds!
    """
    unfairness = 5.
    wildness = 0.01
    N, k = X.shape
    # normalize the performances to (0,1)
    y -= np.amin(y)
    amax = np.amax(y)
    # if the distribution of performance has no width,
    #   give everyone an equal shot
    if np.isclose(amax,0.):
        p = np.ones(N)*1./N
    else:
        y /= np.amax(y)
        y *= -1.
        y -= np.amin(y)
        y += 1
        p = y/np.sum(y)
    #This chooses from the sample based on the power law,
    #  allowing replacement means that the the individuals
    #  can have multiple kids
    p = unfairness*np.power(p,unfairness-1)
    p /= np.sum(p)
    if np.isnan(p).any():
        raise Exception("got isnans in GeneticStep")
    moms = np.random.choice(N, N, replace=True, p=p)
    dads = np.random.choice(N, N, replace=True, p=p)
    # calculate a perturbation to the median
    #   of each individual's parents
    perturbation = np.random.normal(
            loc = 0.,
            scale=wildness*(bounds[:,1]-bounds[:,0]),
            size=(N,k))
    # the children are the median of their parents plus a perturbation
    norm = p[moms]+p[dads]
    weights = (p[moms]/norm, p[dads]/norm)
    weighted_linear_sum = weights[0].reshape(-1,1)*X[moms] + weights[0].reshape(-1,1)*X[dads]
    children = weighted_linear_sum + perturbation
    return children

def deflated_local(starts, results_all, results_minima, gradient, hessian, bounds, workers,r,alpha):
    for j in range(5):
        percent_none = 0.
        tmp_results = workers.imap_unordered(
                partial(newton,minima=results_minima,gradient=gradient,hessian=hessian,bounds=bounds,r=r,alpha=alpha),
                starts)
        for x in tmp_results:
            if not x["success"]:
                percent_none += 1./starts.shape[0]
            else:
                if alreadyFound(x["x"].reshape(1,-1), results_all, r**2, starts.shape[1]):
                    percent_none += 1./starts.shape[0]
                else:
                    if x["edge"]:
                        results_all = np.append(results_all, x["x"].reshape(1,-1), 0)
                    else:
                        results_all = np.append(results_all, x["x"].reshape(1,-1), 0)
                        results_minima = np.append(results_minima, x["x"].reshape(1,-1), 0)
            if percent_none > 0.2:
                return results_all, results_minima
    return results_all, results_minima

def HGDL(func, grad, hess, bounds, r=.3, alpha=.1):
    k = len(bounds)
    starts = random_sample(5, k, bounds)
    func_vals = np.array([func(x) for x in starts])
    results_all = np.empty((0,k))
    results_minima = np.empty((0,k))
    workers = Pool(processes=cpu_count(logical=False)-1)
    for i in range(5):
        print('starts',starts.round(4), func_vals.round(4))
        print('minima - all',results_all.round(4))
        print('minima - true',results_minima.round(4))
        starts = GeneticStep(starts, func_vals, bounds)
        func_vals = np.array([func(x) for x in starts])
        #results_all, results_minima = deflated_local(starts, results_all, results_minima, grad, hess, bounds, workers, r, alpha)

    func_vals_all = np.array([func(x) for x in results_all])
    x = np.append(results_all, starts, 0)
    y = np.append(func_vals_all, func_vals)
    if len(x) == 0:
        return {"success":False}
    c = np.argsort(y)
    x, y = x[c][:5], y[c][:5]
    return {"success":True,"x":x,"func":y}

#from scipy.optimize import rosen, rosen_der, rosen_hess
#b = np.array([[-20, 20],[-30,30.]])
#HGDL(rosen, rosen_der, rosen_hess, b)
