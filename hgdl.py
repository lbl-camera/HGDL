# coding: utf-8
# # HGDL
#     * Hybrid - uses both local and global optimization
#     * G - uses global optimizer
#     * D - uses deflation
#     * L - uses local extremum localMethod
#  imports
import numpy as np
import numba as nb
from multiprocessing import Pool
from functools import partial
from math import ceil
from psutil import cpu_count
import scipy.optimize

# main class
class HGDL(object):
    """
    This is a bounded hybrid local optimizer that uses
        scipy's minimize, deflation, and a global optimizer.
    This is the class that wraps up all the functions
    """
    def __init__(self, func, bounds, localArgs={}, globalArgs={}):
        """
        This function does the following:
        + while global condition is not met:
            - while local condition is not met:
                * run deflated local optimization
                * gather up found optima to use in deflation
        takes:
            required:
                function: a function that takes only a numpy array x
                gradient: function's derivative in the same format
                    (if you need to give args that are the same every
                        time, use functools.partial)
                bounds: a numpy arrary of shape (dim, 2)
                    of the lower and upper bounds for each dimension
            kwargs (and the defaults):
                localArgs:
                globalArgs:
        returns:
            OptimizationResult
        """
        # initialize constants
        self.rng = np.random.default_rng()
        self.k = len(bounds)
        self.N = 5
        self.unfairness = 2.5
        self.wildness = .01
        self.obj = func
        self.fArgs = localArgs.get('args',{})
        # save user input
        self.func = localMethod(func, bounds, localArgs)
        self.bounds = bounds
        self.globalArgs = globalArgs
        self.localArgs = localArgs

    def run(self):
        x0 = self.random_sample(self.N)
        res, err  = self.func.compute(x0)
        print(res, err)

    # This is my implementation of a genetic algorithm
    def GeneticStep(self, X, y):
        """
        Input:
        X is the individuals - points on a surface
        y is the performance - f(X)
        Notes:
        the children can be outside of the bounds!
        """
        numChoose = min(len(y), self.N)
        # normalize the performances to (0,1)
        y = -y
        y = y - np.amin(y)
        y = y + 1
        p = y/np.sum(y)
        # This chooses from the sample based on the power law,
        #  allowing replacement means that the the individuals
        #  can have multiple kids
        p = self.unfairness*np.power(p,self.unfairness-1)
        p /= np.sum(p)
        if np.isnan(p).any():
            raise ValueError("isnan in performances")
        parents = self.rng.choice(
                np.arange(len(p)), size=(2,numChoose), p=p)
        # calculate a perturbation to the weighted median
        #   of each individual's parents 
        scaling = (p[parents[0]]+p[parents[1]])
        mom = X[parents[0]].T * (p[parents[0]]/scaling)
        dad = X[parents[1]].T * (p[parents[1]]/scaling)
        children = (mom+dad).T

        perturbation = self.rng.normal(
                scale=self.wildness*(self.bounds[:,1]-self.bounds[:,0]),
                size=(numChoose,self.k))
        # the children are the median of their parents plus a perturbation
        return children + perturbation

    ## Utility Functions
    def random_sample(self, N):
        sample = self.rng.random((N, self.k))
        sample *= self.bounds[:,1] - self.bounds[:,0]
        sample += self.bounds[:,0]
        return sample

class  localMethod(object):
    def __init__(self, func, bounds, localArgs):
        self.func = func
        self.bounds = bounds
        self.localArgs = localArgs
        self.alpha = .1
        self.radius_squared = .01
        self.maxLocalSteps = 2

    def base_function(self, x):
        return scipy.optimize.minimize(
                    fun = self.func,
                    x0 = x,
                    **self.localArgs)

    def compute(self, x0):
        numCpu = cpu_count(logical=False)
        workers = Pool(numCpu-1)
        newMinima = []
        for i in range(self.maxLocalSteps):
            numNone = 0
            numAlreadyFound = 0
            numOob = 0
            singleStepResults = []
            # chunk up and iterate through
            chunksize = ceil(len(x0)/(numCpu-1))
            walkerOutput = workers.imap_unordered(
                    self.base_function,
                    x0,
                    chunksize=chunksize)
            # check for stopping criterion
            for i, x_found in enumerate(walkerOutput):
                if x_found.success == False:
                    numNone += 1
                    if (numNone+numAlreadyFound+numOob)/len(x0) > 0.7:
                        newMinima += singleStepResults
                        return newMinima, (numNone, numAlreadyFound, numOob)
                elif self.alreadyFound(
                        x_found.x,
                        [x.x for x in singleStepResults],
                        self.radius_squared):
                    numAlreadyFound += 1
                    if (numNone+numAlreadyFound)/len(x0) > 0.7:
                        newMinima += singleStepResults
                        return newMinima, (numNone, numAlreadyFound, numOob)
                elif not self.in_bounds(x_found.x, self.bounds):
                    numOob += 1
                    if (numNone+numAlreadyFound+numOob)/len(x0) > 0.7:
                        newMinima += singleStepResults
                        return newMinima, (numNone, numAlreadyFound, numOob)
                else:
                    singleStepResults.append(x_found)
            newMinima += singleStepResults
        return newMinima, (numNone, numAlreadyFound, numOob)

    @staticmethod
    def alreadyFound(newPt, oldPts, radius_squared):
        """
        check if the new minimum is within range of the old ones
        """
        print(newPt,oldPts)
        for i in range(len(oldPts)):
            r = newPt - oldPts[i]
            if np.dot(r,r)<radius_squared:
                return True
        return False
    @staticmethod
    def in_bounds(x, bounds):
        if (bounds[:,1]-x > 0).all() and (bounds[:,0] - x < 0).all():
            return True
        return False




'''
    def epoch_step(self):
        self.epoch += 1
        # take a single genetic step or random sample if no info
        if len(self.res) != 0:
            new_starts = self.GeneticStep(self.res[:,:-1], self.res[:,-1])
            for i in range(len(new_starts)):
                if not self.in_bounds(new_starts[i], self.bounds):
                    new_starts[i] = self.random_sample(1)
        else:
            new_starts = self.random_sample(self.N)
        # run the local minimizers
        self.walk_individuals(new_starts)
        # sort the results and update the best
        self.res = self.res[self.res[:,-1].argsort()]
        if len(self.res) != 0:
            self.best[(self.epoch-1)%self.keepLastX] = self.res[0,-1]
        if self.verbose: print('epoch:',self.epoch,
                '\n\tres:\n\t',self.res.round(2),'\n\tbest:\n\t',self.best.round())

    def walk_individuals(self, individuals):
        """
        Do the actual iteration through the deflated local optimizer
        """
        newMinima = np.empty((0, self.k+1))
        for i in range(self.maxLocalSteps):
            numNone = 0
            # construct the jacobian, hessian, and objective.
            #   This is inside the loop because of the minima
            minimizer = self.deflate()
            # chunk up and iterate through
            chunksize = ceil(self.N/self.numWorkers)
            walkerOutput = self.workers.imap_unordered(
                    minimizer, individuals, chunksize=chunksize)
            # check for stopping criterion
            for i, x_found in enumerate(walkerOutput):
                if x_found.success == False:
                    numNone += 1
                    if numNone/self.N > self.maxLocalNonePerc:
                        if self.verbose:
                            print('\t hit maximum % Nones')
                        return
                else:
                    if not self.in_bounds(x_found.x, self.bounds):
                        numNone += 1
                        if numNone/self.N > self.maxLocalNonePerc:
                            if self.verbose:
                                print('\t hit maximum % Nones')
                            return
                    else:
                        if not np.isscalar(x_found.fun): x_found.fun = x_found.fun[0]
                        newMinima = np.array([*x_found.x, x_found.fun]).reshape(1,-1)
                        if len(self.res)==0:
                            self.res = np.concatenate(
                                    (newMinima, self.res), axis=0)
                        elif not self.alreadyFound(newMinima, self.res,
                                radius_squared=self.radius_squared, k=self.k):
                            self.res = np.concatenate((newMinima, self.res), axis=0)
        if self.verbose: print('\t hit max local steps')

    ## Deflation Process
    def deflate(self):
        if self.hessian is not None:
            Hess = partial(deflated_hessian,
                    gradient = self.gradient,
                    hessian = self.hessian,
                    minima = self.res[:,:-1],
                    radius_squared = self.radius_squared,
                    alpha = self.alpha)
        else:
            Hess = None
        if self.localMethod in ['L-BFGS-B', 'TNC', 'SLSQP', 'trust-constr']:
            g = partial(minimize,
                    jac = partial(deflated_gradient,
                        gradient = self.gradient,
                        minima = self.res[:,:-1],
                        radius_squared = self.radius_squared,
                        alpha = self.alpha),
                    method=self.localMethod,
                    hess = Hess,
                    tol = self.tol,
                    bounds = self.bounds,
                    options = {"maxiter":self.maxLocalSteps})
        else:
             g = partial(minimize,
                    jac = partial(deflated_gradient,
                        gradient = self.gradient,
                        minima = self.res[:,:-1],
                        radius_squared = self.radius_squared,
                        alpha = self.alpha),
                    method=self.localMethod,
                    hess = Hess,
                    tol = self.tol,
                    options = {"maxiter":self.maxLocalSteps})
        return partial(wrapped_scipy, f=self.objective, g=g)

# wrapper that can't be a local object
def wrapped_scipy(x, f, g):
    return g(f, x)

## Math functions
# the bump function itself
@nb.vectorize([nb.float64(nb.float64,nb.float64,nb.float64)],
        nopython=True, cache=True)
def bump_function(dist2center, radius_squared, alpha):
    """ vectorized, eager. 1./1-b(x-x0) """
    bump_term = np.exp( (-alpha)/(radius_squared - dist2center)
            + (alpha/radius_squared) )
    return 1./(1.-bump_term)
# the derivative of the bump function
@nb.vectorize([nb.float64(nb.float64,nb.float64,nb.float64)],
        nopython=True)
def bump_derivative(dist2center, radius_squared, alpha):
    """ vectorized, eager. (1./1-b(x-x0))' """
    bump_der = np.exp((-alpha)/(radius_squared - dist2center)
            + (alpha/radius_squared))
    bump_der *= -2*alpha*np.sqrt(dist2center) \
            /np.power(radius_squared-dist2center,2)
    return  np.power(bump_function(dist2center,
        radius_squared,alpha),2)*bump_der
    # the bump function for a bunch of minimai
@nb.njit(nb.float64(nb.float64[:], nb.float64[:,:],
    nb.float64, nb.float64))
def deflation_factor(x, x0s, radius_squared, alpha):
    r = np.sum(np.power(x0s-x,2),1)
    return np.prod(bump_function(r[r<radius_squared],
        radius_squared, alpha))
    # the bump function's derivative for a bunch of minima
def deflation_derivative(x, x0s, radius_squared, alpha):
    r = np.sum(np.power(x0s-x,2),1)
    return np.prod(bump_derivative(r[r<radius_squared],
        radius_squared, alpha))
    ## Wrappers
def deflated_gradient(x, gradient, minima,
        radius_squared, alpha):
    return gradient(x) * deflation_factor(x, minima, radius_squared, alpha)
def deflated_hessian(x, gradient, hessian, minima,
        radius_squared, alpha):
    term1 = hessian(x) * deflation_derivative(x, minima, radius_squared, alpha)
    term2 = gradient(x) * deflation_factor(x, minima, radius_squared, alpha)
    return term1 + term2
'''
# ---------------------------------------------------------------------
## test
import scipy.optimize as sOpt

def rosen(x, b, c=None):
    if c is None:
        print('bummer, my dood')
        return
    return sOpt.rosen(x)

def rosen_der(x, b, c=None):
    if c is None:
        print('bummer, my dood')
        return
    return sOpt.rosen_der(x)

def rosen_hess(x, b, c=None):
    if c is None:
        print('bummer, my dood')
        return
    return sOpt.rosen_hess(x)




b = np.array([[-2,2],[-2,2],[-2,2]])
opt = HGDL(rosen, b, localArgs={'jac':rosen_der,'args':(None, 1)})
print(opt.run())

'''
print('new run\n')
opt = HGDL(rosen, b, globalArgs={'popsize':100}, localArgs={'args':(None, 1)} )
print(opt.run())
print('new run\n')
opt = HGDL(rosen, b, localArgs={'jac':rosen_der,'options':{'verbose':True},'args':(None, 1)})
print(opt.run())

print('new run\n')
opt = HGDL(rosen, b, localArgs={'jac':rosen_der, 'hess':rosen_hess,'args':(None,1)})
print(opt.run())

print('new run\n')
opt = HGDL(rosen, b,
    localArgs={'jac':rosen_der, 'hess':rosen_hess,'args':(None,1)},
    globalArgs={'popsize':200})
print(opt.run())
'''
