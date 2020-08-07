# coding: utf-8

#  imports
import numpy as np
from .results import Results
from .bump import deflation, deflation_der

class info(object):
    """
    hold onto the necessary info for HGDL
    """
    def __init__(
            self, func, grad, bounds,
            hess=None, client=None, fix_rng=True,
            r=.3, alpha=.1, num_epochs=5, bestX=5,
            num_individuals=15, max_local=4, num_workers=None,
            x0=None, global_method='genetic', local_method='scipy',
            local_args=(), local_kwargs={}, global_args=(), global_kwargs={}):
        """
        Mandatory Parameters:
            * func - should return a scalar given a numpy array x
            -- note: use functools.partial if you have optional params
            * grad - gradient vector at x
            * hess - hessian array at x
            * bounds - numpy array of bounds in same format as scipy.optimize
        Optional Parameters:
            * r (0.3) - the radius of the deflation operator
            * alpha (0.1) - the alpha term of the bump function
            * maxEpochs (5) - the maximum number of epochs
            * numIndividuals (15) - the number of individuals to run
            * maxLocal (5) - the maximum number of local runs to do
            * numWorkers (logical cpu cores -1) - how many processes to use
        Returns:
            a dict of the form
            either {"success":False} if len(x) is 0
            or {"success":True, "x",x, "y",y} with the bestX x's and their y's
        """
        # disable scipy minimize's ftol check 
        self.func = func
        self._grad = grad
        self._hess = hess
        self.bounds = bounds
        if fix_rng: seed = 42
        else: seed = None
        self.rng = np.random.default_rng(seed)
        self.r = r
        self.alpha = alpha
        self.num_epochs = num_epochs
        self.bestX = 5
        self.max_local = max_local
        self.num_individuals = num_individuals
        self.num_workers = num_workers
        self.global_method = global_method
        self.local_method = local_method
        self.local_args = local_args
        self.local_kwargs = local_kwargs
        self.global_args = global_args
        self.global_kwargs = global_kwargs
        self.k = len(bounds)
        self.results = Results(self)
        self.use_dask_map = False
        if num_workers is None:
            from psutil import cpu_count
            self.num_workers = cpu_count(logical=False)-1
        if x0 is None:
            x0 = self.random_sample(self.num_individuals)
        self.x0 = x0
        self.results.update_global(x0)
        self.r2 = r**2

    @property
    def minima(self):
        return self.results.minima_x

    def grad(self, x):
        j = self._grad(x)
        defl = deflation(x, self.minima, self.r2, self.alpha)
        return j*defl

    def hess(self, x):
        h = self._hess(x)
        j = self._grad(x)
        defl = deflation(x, self.minima, self.r2, self.alpha)
        defl_der = deflation_der(x, self.minima, self.r2, self.alpha)
        return h*defl + np.outer(defl_der, j)

    def update_global(self, x):
        self.results.update_global(x)

    def update_minima(self, x):
        self.results.update_minima(x)

    def random_sample(self, N):
        return self.rng.uniform(
                low = self.bounds[:,0],
                high = self.bounds[:,1],
                size = (N,self.k))

    def in_bounds(self, x):
        return (self.bounds[:,0]<x).all() and (x<self.bounds[:,1]).all()



