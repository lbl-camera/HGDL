#!/usr/bin/env python
# coding: utf-8

# # HGDL
#     * Hybrid - uses both local and global optimization
#     * G - uses global optimizer
#     * D - uses deflation
#     * L - uses local extremum method
# The goal of this is to be modular and robust to a variety of functions
# ## Imports

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial
from math import ceil
import numba as nb
from multiprocessing import cpu_count
from scipy.optimize import minimize


# In[ ]:


# Set this to true to test functions
DEBUG = False


# ## Minimization
# ### Parameters - passing a bunch of params is unseemly

# In[ ]:


def defaultParams():
    parameters = np.dtype([('numWorkers','i2'), # how many processes to run
                            ('radius_squared','f8'), # how close is close enough for bump function
                            ('maxCount','i4'), # how many iterations of local method
                            ('alpha','f8'), # alpha parameter of bump function
                            ('unfairness','f8'), # how unfair is the global method
                            ('wildness','f8'), # how much randomness is in the global method
                           # Note - set wildness high to be high randomness
                            ('N','i4'), # how many individuals to have
                            ('keepLastX','i2'), # converge when best of last x runs is the same
                            ('maxRuns','i4'), # maximum number of iterations of local method
                            ('returnedThreshold','f8'), # what threshold of % local failed is enough
                            ('verbose','b'), # print out best at each step
                            ('k','i4'), # reserved by algorithm 
                            ('numGenerations','i4'), # reserved by algorithm
                          ])

    parameters = np.recarray(1, parameters)
    
    parameters.numWorkers = -1
    parameters.maxCount = 100
    parameters.alpha = .1
    parameters.unfairness = 2.5
    parameters.wildness = 1
    parameters.N = 100
    parameters.keepLastX = 10
    parameters.maxRuns = 10
    parameters.returnedThreshold=0.7
    parameters.verbose = False
    parameters.numGenerations = 0
    
    parameters.k = -2
    parameters.radius_squared = -2.
    
    return parameters[0]


# ### Define deflation operator and derivatives

# In[ ]:


@nb.vectorize(nopython=True, cache=True)
def bump_function(dist2center, radius_squared, alpha):
    """
    This actually takes the squared distances to the center, |x-x0|^2
    This is vectorized over distances
    Marcus's bump function
        - find it at https://www.sciencedirect.com/science/article/pii/S037704271730225X
    """
    if dist2center==radius_squared: return 0
    #if dist2center<1e-7: return 10000
    bump_term = np.exp( (-alpha)/(radius_squared - dist2center) + (alpha/radius_squared) )
    return 1./(1.-bump_term)


# In[ ]:


if DEBUG:
    x = np.arange(0, 2.5, 1e-2)
    plt.plot(x, bump_function(x, 2.5, .1))


# In[ ]:


@nb.vectorize(nopython=True, cache=True)
def bump_derivative(dist2center, radius_squared, alpha):
    """
    This actually takes the squared distances to the center, |x-x0|^2
    This is vectorized over distances
    Marcus's bump function
        - find it at https://www.sciencedirect.com/science/article/pii/S037704271730225X
    """
    if dist2center==radius_squared: return 0
    bump_der = np.exp( (-alpha)/(radius_squared - dist2center) + (alpha/radius_squared) )
    bump_der *= -2*alpha*np.sqrt(dist2center)/np.power(radius_squared-dist2center,2)
    return np.power(bump_function(dist2center,radius_squared,alpha),2)*bump_der


# In[ ]:


if DEBUG:
    x = np.arange(0, 2.4, 1e-2)
    plt.plot(x, bump_derivative(x, 2.5, .1))


# In[ ]:


@nb.jit(nopython=True, cache=True)
def deflation_factor(x, minima, radius_squared, alpha):
    """
    This calculates:
        * what minima is this x in range of
        * for the minima in range, what is their deflation factor
        * combined defaltion factor
    """
    # initialize scaling factor
    factor = 1.
    xLen = len(x)
    zLen = len(minima)
    
    # doing all the math in matrix form is much faster 
    c = x-minima[:,:xLen]
    dists2center = np.sum(c*c,axis=1)
    withinRange = dists2center < radius_squared
    return np.prod(bump_function(dists2center[withinRange], radius_squared, alpha)) 


# In[ ]:


if DEBUG:
    assert np.isinf(deflation_factor(np.ones(3), np.ones((1,3)), 1., 1.,)), "deflation factor is not blowing up"


# In[ ]:


if DEBUG:
    for i in range(100):
        stop = np.random.random()*100
        alpha = np.random.random()*3 + .1
        x = np.arange(0, stop, 1e-3)
        y = bump_function(x, stop, alpha)
        assert np.array([y[i]<=y[i-1] for i in range(1,len(y))]).all(), "bump function is not monotonically decreasing for alpha "+str(alpha)+' and stop: '+str(stop)
        y = bump_derivative(x, stop, alpha)
        assert np.array([y[i]>=y[i-1] for i in range(2,len(y))]).all(), "bump derivative is not monotonically increasing for alpha "+str(alpha)+' and stop: '+str(stop)


# In[ ]:


@nb.jit(nopython=True, cache=True)
def deflation_derivative(x, minima, radius_squared, alpha):
    """
    This calculates:
        * what minima is this x in range of
        * for the minima in range, what is their deflation factor
        * combined defaltion factor
    """
    # initialize scaling factor
    factor = 1.
    xLen = len(x)
    zLen = len(minima)
    
    # doing all the math in matrix form is much faster 
    c = x-minima[:,:xLen]
    dists2center = np.sum(c*c,axis=1)
    withinRange = dists2center < radius_squared
    return np.prod(bump_derivative(dists2center[withinRange], radius_squared, alpha))


# In[ ]:


if DEBUG:
    assert np.isinf(deflation_derivative(np.ones(3), np.ones((1,3))+1e-10, 1., 1.,)), "deflation derivative is not blowing up"


# ### Define checks (necessary bc of parallelism)

# In[ ]:


@nb.jit(nopython=True, cache=True)
def alreadyFound(newMinima, oldMinima, radius_squared, k):
    """
    check if the new minimum is within range of the old ones
    """
    c = oldMinima[:,:k] - newMinima[0,:k]
    return (np.sum(c*c,1)<radius_squared).any()


# In[ ]:


if DEBUG:
    assert alreadyFound(10*np.ones((1, 4)), 10*np.ones((1,4)), 10, 3), "not noticing already found minima"


# ### Define wrappers to make interface generic

# In[ ]:


def deflated_gradient(x, gradient, minima, radius_squared, alpha):
    """
    This just combines these two functions together
    """
    return gradient(x)*deflation_factor(x, minima, radius_squared, alpha)


# In[ ]:


def deflated_hessian(x, gradient, hessian, minima, radius_squared, alpha):
    """
    This just combines these two functions together
    """
    term1 = hessian(x)*deflation_derivative(x, minima, radius_squared, alpha)
    term2 = gradient(x)*deflation_factor(x, minima, radius_squared, alpha)
    return term1 + term2


# In[ ]:


def minimize_wrapper(x0, objective):
    """
    The partial function requires x to be in front, so this just does
    a switcheroo with w(x,f,..) <=> w(f,x)
    """
    return minimize(fun=objective, x0=x0)


# ### Define parallelized deflated local stepdefaultParams

# In[ ]:


def minimizer(x, objective, method, jac, hess, tol, options):
    return minimize(objective, x, method=method, jac=jac, hess=hess, tol=tol, options=options)


# In[ ]:


def wrapper(x, objective, **kwargs):
    return minimize(objective, x, **kwargs)


# In[ ]:


def walk_individuals(individuals, bounds, objective, gradient, Hessian, workers, parameters, method, minima=None):
    """
    Do the actual iteration through the deflated local optimizer
    """
    
    if minima is None: minima = np.empty((0, parameters.k+1))
    
    newMinima = np.empty((0, parameters.k+1))
        
    for i in range(parameters.maxRuns):
        numNone = 0
        
        # construct the jacobian, hessian, and objective. This is inside the loop because of the minima
        jac = partial(deflated_gradient, gradient=gradient, minima=minima, radius_squared=parameters.radius_squared, 
                      alpha=parameters.alpha)
        
        if Hessian is not None: Hessian = partial(deflated_hessian, gradient=gradient, hessian=Hessian, minima=minima,
                                                  radius_squared=parameters.radius_squared, alpha=parameters.alpha)
        
        minimizer = partial(wrapper, objective=objective, method=method, jac=jac, hess=Hessian, 
                    tol=parameters.radius_squared, options={'maxiter':parameters.maxCount})

        # chunk up and iterate through
        walkerOutput = workers.imap_unordered(minimizer, individuals, chunksize=ceil(0.3*len(individuals)/cpu_count()))

        # process results
        for i,x_found in enumerate(walkerOutput):
            if x_found.success==False: 
                numNone += 1
                if numNone/parameters.N > parameters.returnedThreshold:
                    return minima
            else:
                if not in_bounds(x_found.x, bounds): 
                    numNone += 1
                    if numNone/parameters.N > parameters.returnedThreshold:
                        return minima                        
                else:
                    score = x_found.fun;
                    if not np.isscalar(score): score = score[0]
                    newMinima = np.array([*x_found.x, x_found.fun]).reshape(1,-1)
                    if len(minima)==0:
                        minima = np.concatenate((newMinima, minima), axis=0)
                    elif not alreadyFound(newMinima, minima, radius_squared=parameters.radius_squared, k=parameters.k): 
                        minima = np.concatenate((newMinima, minima), axis=0)
    return minima


# ### Define a global optimizer

# In[ ]:


def Procreate(X, y, parameters):
    """
    Input:
    X is the individuals - points on a surface
    y is the performance - f(X)
    
    Parameters:
    unfairness is a metric from (1,infinity), where higher numbers skew more towards well performing individuals having more kids
    cauchy wildness is a scaling factor for the standard cauchy distribution, where higher values mean less variation
        (although it's still a nonstationary function...)
    
    Notes:
    some of the children can (and will) be outside of the bounds!
    """
    
    # normalize the performances to (0,1)
    p = y - np.amin(y)
    maxVal = np.amax(p)
    if maxVal == 0: p = np.ones(len(p))/len(p) # if the distribution of performance has no width, give everyone an equal shot
    else: p /= maxVal

    #This chooses from the sample based on the power law, allowing replacement means that the the individuals can have multiple kids
    p = parameters.unfairness*np.power(p,parameters.unfairness-1)
    p /= np.sum(p)
    if np.isnan(p).any():
        return p, X, y
    moms = np.random.choice(np.arange(len(p)), parameters.N, replace=True, p = p)
    dads = np.random.choice(np.arange(len(p)), parameters.N, replace=True, p = p)
    
    # calculate a perturbation to the median of each individuals parents
    perturbation = np.random.standard_normal(size=(parameters.N,parameters.k))
    perturbation *= parameters.wildness
    
    # the children are the median of their parents plus a perturbation (with a chance to deviate wildly)
    children = (X[moms]+X[dads])/2. + perturbation*(X[moms]-X[dads])

    return children


# ### Check boundary conditions

# In[ ]:


def in_bounds(x, bounds):
    if (bounds[:,1]-x > 0).all() and (bounds[:,0] - x < 0).all(): return True
    return False


# ### Sample within bounds

# In[ ]:


def random_sample(N,bounds,parameters):
    sample = np.random.random((N,parameters.k))
    sample *= bounds[:,1]-bounds[:,0]
    sample += bounds[:,0]
    return sample


# ### Wrap everything together

# In[ ]:


def HXDY(fun, bounds, jac, method=None, hess=None, x0=None, 
         parameters=None, rms = .01, extraStoppingCriterion=None ):
    """
    This is a bounded hybrid local optimizer that uses scipy's minimize,
      deflation, and a genetic optimizer.
      
    This function does the following:
    
    while global condition is not met:
        while local condition is not met:
            run deflated local optimization
            gather up found optima to use in deflation
    
    parameters:
        takes a parameters object that is fairly transparent if you look at HXDY.ipynb
        
    returns:
        result - a dict with 'x', 'f', and 'success' which give found x values
          their corresponding function ('f') values, and whether or not the method failed
    
    """
    # Initialization
    if parameters == None:
        parameters = defaultParams()

    parameters.k = len(bounds)
 
    if parameters.numWorkers == -1: parameters.numWorkers = cpu_count()

    if x0 is None: starts = random_sample(parameters.N,bounds,parameters)
    else: starts = x0
        
    objective = fun
    hessian = hess
    gradient = jac
    parameters.radius_squared = parameters.k*(rms**2)
    
    if extraStoppingCriterion is None: extraStoppingCriterion = lambda x: True
        
    workers = Pool(parameters.numWorkers)
     
    res = walk_individuals(starts, bounds, objective, gradient, hessian, workers, parameters, method=method)
    
    parameters.numGenerations = 0
    
    # processing
    res = res[res[:,-1].argsort()]
    best = np.inf*np.ones(parameters.keepLastX); 
    if len(res)==0: best[0] = np.nan_to_num(np.inf)-1
    else: best[0] = res[0,-1]
        
    while not np.allclose(best[0],best) and extraStoppingCriterion(res):

        parameters.numGenerations += 1

        if res.shape[0]!=0: 
            new_starts = Procreate(res[:,:parameters.k], res[:,-1], parameters)
            for i in range(len(new_starts)): 
                if not in_bounds(new_starts[i], bounds): new_starts[i] = random_sample(1,bounds,parameters)
        else:
            new_starts = random_sample(parameters.N,bounds,parameters)

        res = walk_individuals(new_starts, bounds, objective, gradient, hessian, workers, parameters, minima=res, method=method)
        
        res = res[res[:,-1].argsort()]
        if res.shape[0]!=0: best[parameters.numGenerations%parameters.keepLastX] = res[0,-1]
        if parameters.verbose: print(best.round())
        
    workers.close()
    
    if hess is not None:
        # keep only if eigenvalues all > -epsilon
        H = np.array([hess(x) for x in res[:,:-1]])
        L, V = np.linalg.eigh(H)
        mask = np.array([(l>-1e-3).all() for l in L])
        res = res[mask]
    
    result = {}
    result['x'] = res[:,:-1]
    result['f'] = res[:,-1]
    result['success'] = res.shape[0]>0
    
    return result


# In[ ]:




