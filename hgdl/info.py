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
            r=.3, alpha=.1, num_epochs=10, bestX=5,
            num_individuals=25, max_local=5,
            x0=None, global_method='genetic', local_method='scipy',
            local_args=(), local_kwargs={}, global_args=(), global_kwargs={},
            verbose=False):
        """
        HGDL
            * Hybrid - uses both local and global optimization
            * G - uses global optimizer
            * D - uses deflation
            * L - uses local extremum localMethod
            Mandatory Parameters:
                * func - should return a scalar given a numpy array x
                -- note: use functools.partial if you have optional params
                * grad - gradient vector at x
                * bounds - numpy array of bounds in same format as scipy.optimize
            Optional Parameters:
                * Overall Parameters -----------------------------------
                * hess (None) - hessian array at x - you may need this depending on the local method
                * client - (None->HGDL initializes if None) dask.distributed.Client object
                    -- this lets you interface with clusters via dask with Client(myCluster)
                * num_epochs (10) - the number of epochs. 1 epoch is 1 global step + 1 local run
                * fix_rng (True) - sets random numbers to be fixed (for reproducibility)
                * bestX (5) - maximum number of minima and global results to put in get_final()
                * num_individuals (25) - the number of individuals to run for both global and local methods
                * x0 (None) starting points to probe
                * global_method ('genetic') - these control what global and local methods
                * local_method ('my_newton') -    are used by HGDL
                * verbose (False) - what it says

                * Global Method Parameters -----------------------------
                * global_args ((,)) - arguments to global method
                * global_kwargs({}) - kwargs for global method
                    -- note: these let you pass custom info to your method of choice

                * Deflation Parameters ---------------------------------
                * alpha (0.1) - the alpha term of the bump function
                * r (.3) - the radius for the bump function
                    -- these define how deflation behaves

                * Local Method Parameters ------------------------------
                * local_args ((,)) - arguments to global method
                * local_kwargs({}) - kwargs for global method
                    -- note: these let you pass custom info to your method of choice
                * max_local (5) - the maximum number of local runs to do

            Returns:
                an HGDL object that has the following functions:
                get_best(): yields a dict of the form
                    {"best_x":best_x_ndarray, "best_y":best_y_value}
                get_final(): yields a dict of the form
                    {"best_x":best_x_ndarray, "best_y":best_y_value,
                    "minima_x":minima_x_ndarray, "minima_y":minima_y_values,
                    "global_x":global_x_ndarray, "global_y":global_y_values,
                    }

        """
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
        self.bestX = bestX
        self.max_local = max_local
        self.num_individuals = num_individuals
        self.global_method = global_method
        self.local_method = local_method
        self.local_args = local_args
        self.local_kwargs = local_kwargs
        self.global_args = global_args
        self.global_kwargs = global_kwargs
        self.verbose = verbose
        self.k = len(bounds)
        self.results = Results(self)
        self.use_dask_map = True
        if x0 is None:
            x0 = self.random_sample(self.num_individuals)
        self.x0 = x0
        self.results.update_global(x0)
        self.r2 = r**2
        # find if the user provided a client
        if client is None:
            from dask.distributed import Client
            client = Client(scheduler_port=0, worker_dashboard_address=':0')
            self.scheduler_file = 'scheduler.json'
        elif client.scheduler_file:
            self.scheduler_file = client.scheduler_file
        else:
            self.scheduler_file = 'scheduler.json'

        client.scheduler_file = None
        client.write_scheduler_file(self.scheduler_file)

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



