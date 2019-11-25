#!/usr/bin/env python
# coding: utf-8

# # HGDL
#     * Hybrid - uses both local and global optimization
#     * G - uses global optimizer
#     * D - uses deflation
#     * L - uses local extremum method
# The goal of this is to be modular and robust to a variety of functions
# ## Imports

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial
from math import ceil
import numba as nb
from multiprocessing import cpu_count
from scipy.optimize import minimize


# ## Minimization
# ### Define deflation operator and derivatives

# In[2]:


@nb.vectorize(nopython=True, cache=True)
def bump_function(dist2center, radius_squared, alpha):
    """
    This actually takes the squared distances to the center, |x-x0|^2
    This is vectorized over distances
    Marcus's bump function
        - find it at https://www.sciencedirect.com/science/article/pii/S037704271730225X
    """
    if dist2center==radius_squared: return 0
    bump_term = np.exp( (-alpha)/(radius_squared - dist2center) + (alpha/radius_squared) )
    return 1./(1.-bump_term)


# In[3]:


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
    bump_der *= -2*alpha*np.power(dist2center,.5)/np.power(radius_squared-dist2center,2)
    return -1.*np.power(bump_function(dist2center,radius_squared,alpha),2)*bump_der


# In[4]:


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


# In[5]:


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


# ### Define wrappers to make interface generic

# In[6]:


def deflated_gradient(x, gradient, minima, radius_squared, alpha):
    return gradient(x)*deflation_factor(x, minima, radius_squared, alpha)


# In[7]:


def deflated_hessian(x, gradient, hessian, minima, radius_squared, alpha):
    term1 = hessian(x)*deflation_derivative(x, minima, radius_squared, alpha)
    term2 = gradient(x)*deflation_factor(x, minima, radius_squared, alpha)
    return term1 + term2


# In[8]:


def minimize_wrapper(x0, fun, *args, **kwargs):
    return minimize(fun, x0)


# ### Define checks (necessary bc of parallelism)

# In[19]:


@nb.jit(nopython=True, cache=True)
def alreadyFound(newMinima, oldMinima, xLen, squared_radius):
    c = oldMinima[:,:xLen] - newMinima[0,:xLen]
    return (np.sum(c*c,1)<squared_radius).any()


# ### Define parallelized deflated local step

# In[20]:


def walk_individuals(individuals, bounds, objective, gradient, Hessian, radius_squared, 
                     workers, epsilon, maxCount, alpha, maxRuns, returnedThreshold, minima=None,
                     args=(), method='L-BFGS-B'):
    

    xLen = len(individuals[0])
    N = len(individuals)
    if minima is None: minima = np.empty((0, xLen+1))
    
    newMinima = np.empty((0, xLen+1))
    
    numNone = 0
    for i in range(maxRuns):
        numNone = 0
        
        jac = partial(deflated_gradient, gradient=gradient, minima=minima, radius_squared=radius_squared, alpha=alpha)
        hess = Hessian
        if hess is not None: hess = partial(deflated_hessian, gradient=gradient, hessian=Hessian, minima=minima, radius_squared=radius_squared, alpha=alpha)
            
        walkerOutput = workers.imap_unordered(partial(minimize_wrapper, fun=objective, args=args, method=method, 
                                                      jac=jac, 
                                                      hess=hess, 
                                                      bounds=bounds, tol=radius_squared, options={'maxiter':maxCount}),
                                              individuals, chunksize=ceil(0.3*len(individuals)/cpu_count()))
                                                      
        # redo this where it can stop before everything returns
        for i,x_found in enumerate(walkerOutput):
            if x_found.success==False: 
                numNone += 1
                if numNone/N > returnedThreshold:
                    # check for redundant optima
                    #newMinima = reduceOutput(newMinima, xLen, radius_squared)
                    #redundant = alreadyFound(newMinima, minima, xLen, radius_squared)
                    #minima = np.concatenate((newMinima[np.logical_not(redundant)], minima), axis=0)
                    return minima
            else:
                if not in_bounds(x_found.x, bounds): 
                    numNone += 1
                    if numNone/N > returnedThreshold:
                        # check for redundant optima
                        #newMinima = reduceOutput(newMinima, xLen, radius_squared)
                        #redundant = alreadyFound(newMinima, minima, xLen, radius_squared)
                        #minima = np.concatenate((newMinima[np.logical_not(redundant)], minima), axis=0)
                        return minima                        
                else:
                    newMinima = np.array([*x_found.x, x_found.fun]).reshape(1,-1)
                    if not alreadyFound(newMinima, minima, xLen, radius_squared): 
                        minima = np.concatenate((newMinima, minima), axis=0)
        # check for redundant optima
        #redundant = alreadyFound(newMinima, minima, xLen, radius_squared)
        
    return minima


# ### Define a global optimizer

# In[14]:


def Procreate(X, y, unfairness, cauchy_wildness):
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
    N = len(X)
    k = len(X[0])
    
    # normalize the performances to (0,1)
    p = y - np.amin(y)
    maxVal = np.amax(p)
    if maxVal == 0: p = np.ones(len(p))/len(p) # if the distribution of performance has no width, give everyone an equal shot
    else: p /= maxVal

    #This chooses from the sample based on the power law, allowing replacement means that the the individuals can have multiple kids
    p = unfairness*np.power(p,unfairness-1)
    p /= np.sum(p)
    if np.isnan(p).any():
        return p, X, y
    moms = np.random.choice(np.arange(len(p)), N, replace=True, p = p)
    dads = np.random.choice(np.arange(len(p)), N, replace=True, p = p)
    
    # calculate a perturbation to the median of each individuals parents
    perturbation = np.random.standard_cauchy(size=(N,k))
    perturbation /= cauchy_wildness
    
    # the children are the median of their parents plus a perturbation (with a chance to deviate wildly)
    children = (X[moms]+X[dads])/2. + perturbation*(X[moms]-X[dads])

    return children


# ### Check boundary conditions

# In[15]:


def in_bounds(x, bounds):
    if (bounds[:,1]-x > 0).all() and (bounds[:,0] - x < 0).all(): return True
    return False


# ### Sample within bounds

# In[16]:


def random_sample(N,k,bounds):
    sample = np.random.random((N,k))
    sample *= bounds[:,1]-bounds[:,0]
    sample += bounds[:,0]
    return sample


# ### Wrap everything together

# In[17]:


def HXDY(fun, bounds, jac, tol, args=(), method='L-BFGS-B', hess=None, x0=None,
    numWorkers = -1,
    epsilon = 1e-8,
    maxCount = 20,
    alpha = 1.,
    unfairness = 5,
    cauchy_wildness = 60,
    minImprovement = 1.1,
    N = 100,
    keepLastX = 5, maxRuns=30, returnedThreshold=0.6,
    extraStoppingCriterion=None,
    verbose = False
        ):

    
    k = len(bounds)
    if numWorkers == -1: numWorkers = cpu_count()
    N = int(N)
    if x0 is None: starts = random_sample(N,k,bounds)
    else: starts = x0
    objective = fun
    hessian = hess
    gradient = jac
    radius_squared = tol
    if extraStoppingCriterion is None: extraStoppingCriterion = lambda x: True
        
    workers = Pool(numWorkers)
      
    res = walk_individuals(starts, bounds, objective, gradient, hessian, radius_squared, workers, epsilon, maxCount, alpha, maxRuns, returnedThreshold, args=args, method=method)

    numGenerations = 0

    res = res[res[:,-1].argsort()]
    best = np.inf*np.ones(keepLastX); 
    if len(res)==0: best[0] = np.nan_to_num(np.inf)-1
    else: best[0] = res[0,-1]
        
    try:
        while not (best[0]==best).all() and extraStoppingCriterion(res):
            numGenerations += 1

            if res.shape[0]!=0: 
                new_starts = Procreate(res[:,:k], res[:,-1], unfairness=unfairness, cauchy_wildness=cauchy_wildness)
                for i in range(len(new_starts)): 
                    if not in_bounds(new_starts[i], bounds): new_starts[i] = random_sample(1,k,bounds)
            else:
                new_starts = random_sample(N,k,bounds)
            #print(new_starts, res[:10])
            res = walk_individuals(new_starts, bounds, objective, gradient, hessian, radius_squared, workers, epsilon, maxCount, alpha, maxRuns, returnedThreshold, minima=res, args=args, method=method)
            res = res[res[:,-1].argsort()]
            best[numGenerations%keepLastX] = res[0,-1]
            if verbose: print(best)
        workers.close()
        return res
    
    except KeyboardInterrupt:
        res = res[res[:,-1].argsort()]
        return res


# In[ ]:





# In[ ]:





# 

# @nb.jit(nopython=True, cache=True)
# def newton_step(x0, grad, hess):    
#     return x0 + np.linalg.solve(hess, -grad)

# def newton(x0, minima, objective, gradient, Hessian, radius_squared, epsilon, maxCount, alpha):
#     """
#     Apply Newton's method until convergence
#     * assumes Hessian is not singular
#     """
#     
#     counter = 1.
#     xLen = len(x0)
#     
#     grad = gradient(x0)*deflation_factor(x0, minima, radius_squared, alpha)
#     if np.isinf(grad).any(): return counter
#     hess = Hessian(x0)
#     xNew = newton_step(x0, grad, hess)
# 
#     xReported = np.empty(xLen+2) # needs to store the x and f(x)
#     
#     while np.dot(grad,grad)>epsilon and counter<maxCount:
# 
#         x0 = xNew
#         grad = gradient(x0)*deflation_factor(x0, minima, radius_squared, alpha)
#         if np.isinf(grad).any(): return counter
#         hess = Hessian(x0)
#         xNew = newton_step(x0, grad, hess)       
# 
#         counter += 1
#         
#     if counter==maxCount: 
#         return counter
#     
#     # store the found x
#     xReported[:xLen] = xNew
#         
#     # store the function value at the found x
#     xReported[xLen] = objective(xNew)
#     xReported[xLen+1] = counter
#     
#     return xReported.reshape(1,-1)

# @nb.jit(nopython=True, cache=True)
# def alreadyFound(newMinima, oldMinima, xLen, squared_radius):
#     N = len(newMinima)
#     result = np.empty(N, dtype=np.bool8)
#     for i in range(N):
#         c = oldMinima[:,:xLen] - newMinima[i,:xLen]
#         result[i] = (np.sum(c*c,1)<squared_radius).any()
#     return result

# @nb.jit(nopython=True, cache=True)
# def eliminate_duplicates(minima, xLen, squared_radius):
#     N = len(newMinima)
#     result = np.empty(N, dtype=np.bool8)
#     for i in range(N):
#         c = oldMinima[:,:xLen] - newMinima[i,:xLen]
#         result[i] = (np.sum(c*c,1)<squared_radius).any()
#         print(np.sum(c*c,1)<squared_radius)
#     return result

# @nb.jit(nopython=True, cache=True)
# def reduceOutput_base(minima, xLen, squared_radius):
#     N = len(minima)
# 
#     duplicate = np.zeros(N, dtype=np.bool8)
#     best_choice = np.empty(N, dtype=np.int64)
#     skippable = set()
#     Bad = np.zeros(N, dtype=np.bool8)
# 
#     for i in range(N):
# 
#         if i in skippable: continue
#         else:
#             c = minima[:,:xLen] - minima[i,:xLen]
#             withinRange = (np.sum(c*c,1)<squared_radius)
# 
#             duplicate[i] = withinRange.sum()!=1
# 
#             if duplicate[i]:
#                 best_choice[i] = minima[withinRange,-1].argmin()
#                 for j in range(len(withinRange)): skippable.add(j); Bad[j]=True;
# 
#     return best_choice, duplicate, Bad

# def reduceOutput(minima, xLen, squared_radius):
#     best_choice, duplicate, Bad = reduceOutput_base(minima, xLen, squared_radius)
#     return np.append(minima[best_choice[duplicate]], minima[np.logical_not(Bad)], 0)

# In[ ]:




