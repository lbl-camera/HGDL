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
from multiprocessing import cpu_count
from scipy.optimize import minimize

## Math functions
@nb.vectorize([nb.float64(nb.float64,nb.float64,nb.float64)],
        nopython=True, cache=True)
def bump_function(dist2center, radius_squared, alpha):
    """ vectorized, eager. 1./1-b(x-x0) """
    bump_term = np.exp( (-alpha)/(radius_squared - dist2center)
            + (alpha/radius_squared) )
    return 1./(1.-bump_term)

@nb.vectorize([nb.float64(nb.float64,nb.float64,nb.float64)],
        nopython=True, cache=True)
def bump_derivative(dist2center, radius_squared, alpha):
    """ vectorized, eager. (1./1-b(x-x0))' """
    bump_der = np.exp((-alpha)/(radius_squared - dist2center)
                            + (alpha/radius_squared))
    bump_der *= -2*alpha*np.sqrt(dist2center) \
            /np.power(radius_squared-dist2center,2)
    return  np.power(bump_function(dist2center,radius_squared,alpha),
                2)*bump_der

@nb.njit(nb.float64(nb.float64[:], nb.float64[:,:],
        nb.float64, nb.float64), cache=True)
def deflation_factor(x, x0s, radius_squared, alpha):
    r = np.sum(np.power(x0s-x,2),1)
    return np.prod(bump_function(r[r<radius_squared],
        radius_squared, alpha))

def deflation_derivative(x, x0s, radius_squared, alpha):
    r = np.sum(np.power(x0s-x,2),1)
    return np.prod(bump_derivative(r[r<radius_squared],
        radius_squared, alpha))


'''
# main class
class HGDL(object):
    """
    This is a bounded hybrid local optimizer that uses
        scipy's minimize, deflation, and a global optimizer.
    This is the class that wraps up all the functions
    """
    def __init__(self, function, gradient, bounds, **kwargs):
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
                hess: hessian matrix function, same format as function
                    and gradient. passed to scipy. (none)
                x0: ndarray of starting guesses (random)
                N: the number of local minimizer processes (100)
                localMethod: localMethod arg to scipy.minimize ('LBFGS')
                    this can be bounded or not - the out of bounds
                    results will get killed either way
                maxLocalSteps: the max number of
                    local minimizer steps (30)
                maxLocalNonePerc: the maximum number of local mimizers
                    that return None before ending an epoch (0.7)
                maxEpochs: the max number of epochs (20)
                unfairness: how unfair the global method is, where
                    higher values are more unfair (1.)
                wildness: likelihood of wild mutations in the global
                    method - high wildness is more mutations (0.6)
                alpha: alpha in the bump function (.1)
                keepLastX: number of epochs where the best result
                    doesn't change before quitting (5)
                numWorkers: number of subprocesses (number of cores)
                rms: the root mean square across dimensions for two
                    minima to be considered the same - set this to the
                    largest rms difference points that makes them
                    experimentally equivalent (0.05)
                tol: tol parameter passed to scipy (0.01)
                earlyStop: a lambda function that takes a result['x']
                    array and return True if criterion is met (none)
                verbose: print out as much as possible (False)
        returns:
            result - a dict with 'x', 'f', and 'success'
        """
        # initialize constants
        self.rng = np.random.default_rng()
        self.epoch = 0
        self.k = len(bounds)
        # initialize mandatory options
        self.objective = function
        self.gradient = gradient
        self.bounds = bounds
        # process kwargs/use defaults
        self.hessian = kwargs.get('hess', None)
        self.N = kwargs.get('N', 10)
        self.x0 = kwargs.get('x0', self.random_sample(self.N))
        self.localMethod = kwargs.get('localMethod', 'L-BFGS-B')
        self.maxLocalSteps = kwargs.get('maxLocalSteps', 30)
        self.maxLocalNonePerc = kwargs.get('maxLocalNonePerc', 0.7)
        self.maxEpochs = kwargs.get('maxEpochs', 20)
        self.unfairness = kwargs.get('unfairness', 1.)
        self.wildness = kwargs.get('wildness', .6)
        self.alpha = kwargs.get('alpha', 0.1)
        self.keepLastX = kwargs.get('keepLastX', 5)
        self.numWorkers = kwargs.get('numWorkers', cpu_count())
        self.radius_squared = self.k*(kwargs.get('rms', 0.5)**2)
        self.tol = kwargs.get('tol', 0.001)
        self.earlyStop = kwargs.get('earlyStop', lambda x: False)
        self.verbose = kwargs.get('verbose', True)
        # initialize worker pool
        self.workers = Pool(self.numWorkers)
        # initialize record of best results
        self.best = np.inf*np.ones(self.keepLastX)
        self.res = np.empty((0, self.k+1))

    def run(self):
        # check for a.) convergence (ignoring when it doesn't find
        #  anything. b) early stopping. c) max epochs
        while (
            not (np.allclose(self.best[0],self.best)
            and not np.isinf(self.best).all())
            and not self.earlyStop(self.res)
            and self.epoch<self.maxEpochs):
                self.epoch_step()
        self.workers.close()
        result = {'x':self.res[:,:-1], 'f':self.res[:,-1]}
        result['success'] = len(self.res)>0
        return result

    def epoch_step(self):
        if self.verbose: print('epoch:',self.epoch,
                '\nres:\n',self.res,'\nbest:\n',self.best.round())
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

    # This is my implementation of a genetic algorithm 
    def GeneticStep(self, X, y):
        """
        Input:
        X is the individuals - points on a surface
        y is the performance - f(X)
        Notes:
        the children can be outside of the bounds!
        """
        # normalize the performances to (0,1)
        p = y - np.amin(y)
        maxVal = np.amax(p)
        # if the distribution of performance has no width, 
        #   give everyone an equal shot
        if maxVal == 0: p = np.ones(len(p))/len(p)
        else: p /= maxVal
        #This chooses from the sample based on the power law, 
        #  allowing replacement means that the the individuals 
        #  can have multiple kids
        p = self.unfairness*np.power(p,self.unfairness-1)
        p /= np.sum(p)
        if np.isnan(p).any():
            return p, X, y
        moms = self.rng.choice(
                np.arange(len(p)), self.N, replace=True, p=p)
        dads = self.rng.choice(
                np.arange(len(p)), self.N, replace=True, p=p)
        # calculate a perturbation to the median 
        #   of each individual's parents
        perturbation = self.rng.normal(
                scale=self.wildness*(self.bounds[:,1]-self.bounds[:,0]),
                size=(self.N,self.k))
        # the children are the median of their parents plus a perturbation
        children = (X[moms]+X[dads])/2. + perturbation*(X[moms]-X[dads])
        return children

    def walk_individuals(self, individuals):
        """
        Do the actual iteration through the deflated local optimizer
        """
        newMinima = np.empty((0, self.k+1))
        for i in range(self.maxLocalSteps):
            numNone = 0
            # construct the jacobian, hessian, and objective. 
            #   This is inside the loop because of the minima
            jac = partial(deflated_gradient,
                    gradient = self.gradient,
                    minima=self.res,
                    radius_squared = self.radius_squared,
                    alpha=self.alpha)
            if self.hessian is not None:
                Hessian = partial(deflated_hessian,
                            gradient=jac,
                            hessian=self.hessian, minima=self.res,
                            radius_squared=self.radius_squared,
                            alpha=self.alpha)
            else: Hessian = None
            minimizer = partial(scipy_minimize,
                            objective=self.objective,
                            localMethod=self.localMethod,
                            jac=jac, hess=Hessian,
                            tol=self.tol,
                            bounds=self.bounds,
                            options={'maxiter':self.maxLocalSteps})
            if len(self.res)==0:
                jac, Hessian = self.gradient, self.hessian
            # chunk up and iterate through
            chunksize = ceil(self.N/self.numWorkers)
            walkerOutput = self.workers.imap_unordered(
                    minimizer, individuals, chunksize=chunksize)
            print('hi')
            # check for stopping criterion 
#            for i in range(len(individuals)):
#                print(individuals[i]) 
#                x_found = minimizer(individuals[i])

#            for i, x_found in enumerate(walkerOutput):
            if True:
                x_found = next(walkerOutput)
                print(x_found)
                if x_found.success == False:
                    numNone += 1
                    if numNone/self.N > self.maxLocalNonePerc:
                        return
                else:
                    if not self.in_bounds(x_found.x, self.bounds):
                        numNone += 1
                        if numNone/self.N > self.maxLocalNonePerc:
                            return
                    else:
                        if not np.isscalar(x_found.fun): x_found.fun = x_found.fun[0]
                        newMinima = np.array([*x_found.x, x_found.fun]).reshape(1,-1)
                        if len(self.res)==0:
                            self.res = np.concatenate(
                                    (newMinima, self.res), axis=0)
                        elif not alreadyFound(newMinima, self.res,
                                radius_squared=self.radius_squared, k=self.k):
                            self.res = np.concatenate((newMinima, self.res), axis=0)
    ## Utility Functions
    @staticmethod
    def in_bounds(x, bounds):
        if (bounds[:,1]-x > 0).all() and (bounds[:,0] - x < 0).all():
            return True
        return False
    def random_sample(self, N):
        sample = self.rng.random((N, self.k))
        sample *= self.bounds[:,1] - self.bounds[:,0]
        sample += self.bounds[:,0]
        return sample
@nb.jit(nopython=True, cache=True)
def alreadyFound(newMinima, oldMinima, radius_squared, k):
    """
    check if the new minimum is within range of the old ones
    """
    c = oldMinima[:,:k] - newMinima[0,:k]
    return (np.sum(c*c,1)<radius_squared).any()

## Wrappers for deflation
def deflated_gradient(x, gradient, minima, radius_squared, alpha):
    """ This just combines these two functions together """
    return gradient(x)*deflation_factor(x, minima, radius_squared, alpha)

def deflated_hessian(x, gradient, hessian, minima, radius_squared,
        alpha):
    """ This just combines these two functions together """
    term1 = hessian(x)*deflation_derivative(x, minima, radius_squared, alpha)
    term2 = gradient(x)*deflation_factor(x, minima, radius_squared, alpha)
    return term1 + term2

## Wrapper for scipy's minimimze 
def scipy_minimize(x, objective, localMethod, jac, hess,
        tol, bounds, options):
    print('hello there') 
    def minimize_select_bounded(method, bounds):
        if method in ['L-BFGS-B', 'TNC', 'SLSQP', 'trust-constr']:
            return partial(minimize, bounds=bounds)
        else:
            return minimize
    return minimize_select_bounded(localMethod, bounds)\
            (objective, x, method=localMethod,jac=jac,hess=hess,
                    tol=tol, options=options)



# ---------------------------------------------------------------------

## test
f = np.sin
f_p = np.cos
b = np.array([[-10,10]])
opt = HGDL(f, f_p, b)
print(opt.run())

print('new run\n\n\n')
def hess(x):
    return -1.*np.sin(x)
opt = HGDL(f, f_p, b, hess=hess)
opt.run()
'''
