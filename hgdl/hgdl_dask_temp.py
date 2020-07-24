# coding: utf-8

#  imports
import numpy as np
from Global.run_global import run_global
from Local.run_local import run_local
from results import Results
from multiprocessing import Process, Queue, Lock
import dask.distributed
import asyncio

class HGDL(object):
    """
    HGDL
        * Hybrid - uses both local and global optimization
        * G - uses global optimizer
        * D - uses deflation
        * L - uses local extremum localMethod
    """
    def __init__(
            self, func, grad, hess, bounds, client,
            r=.3, alpha=.1, max_epochs=5,
            num_individuals=15, max_local=4, num_workers=None, bestX=5,
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
            * bestX (5) - return the best X x's
        Returns:
            a dict of the form
            either {"success":False} if len(x) is 0
            or {"success":True, "x",x, "y",y} with the bestX x's and their y's
        """

        self.func = func
        self.grad = grad
        self.hess = hess
        self.bounds = bounds
        self.client = client
        self.r = r
        self.alpha = alpha
        self.max_epochs = max_epochs
        self.max_local = max_local
        self.num_individuals = num_individuals
        self.num_workers = num_workers
        self.bestX = bestX
        self.global_method = global_method
        self.local_method = local_method
        self.local_args = local_args
        self.local_kwargs = local_kwargs
        self.global_args = global_args
        self.global_kwargs = global_kwargs

        self.rng = np.random.default_rng(42)
        self.k = len(bounds)
        self.results = Results(self)

        if num_workers is None:
            from psutil import cpu_count
            self.num_workers = cpu_count(logical=False)-1
        if x0 is None:
            x0 = self.random_sample(self.num_individuals, self.k, self.bounds)
        self.x0 = x0
        self.results.update_global(self.x0)
        # this is a flag that goes up after one epoch is made 
        self.event = asyncio.Event()
        # what these will do as they're added
        # tasks[0] - run one epoch - made at startup
        # tasks[1] - wait until 1 epoch, make task[2] - made at startup
        # tasks[2] - run rest of epochs - made at end of 1st epoch by tasks[1]
        self.tasks = []
        self.loop = asyncio.get_event_loop()
        self.loop.run_until_complete(self.first_epoch())

    # user access functions
    def get_final(self):
        # wait until everything is done 
        self.loop.run_until_complete(asyncio.gather(*self.tasks))
        self.close()
        return self.best

    def get_best(self):
        # wait until at least one epoch is done 
        self.loop.run_until_complete(asyncio.gather(self.tasks[0]))
        return self.best
    # auxilary functions
    # explanation of the asyncio stuff:
    #  * self.tasks contains:
    #     - a task for the first epoch of computation
    #     - a task that waits on the first epoch then adds the last task
    #     - a task for the rest of the computation
    #  * what this satisfies:
    #     - if the user does nothing, the first task should run
    #        which wakes up the second task to add the last task
    #        so that all the loops are run 
    #     - if the user asks for the best, it must wait for the first
    #        epoch to finish before return self.best
    async def first_epoch(self):
        self.tasks.append(asyncio.create_task(self.epoch()))
        self.tasks.append(asyncio.create_task(self.wait_until_first_epoch()))
    async def wait_until_first_epoch(self):
        # wait until one epoch is done, then add next epochs (or epochs) 
        await self.event.wait()
        # create task starts the task running, so we need to wait until
        #  we get the results from the first epoch before running the next
        self.tasks.append(asyncio.create_task(self.rest_of_epochs()))

    # work functions
    async def rest_of_epochs(self):
        for i in range(1,self.max_epochs):
            self.best = await self.epoch()
            if self.results.mimia_x.shape[0]>1: print(i) # debug
        self.best = self.results.roll_up()
    #a single epoch
    async def epoch(self):
        await asyncio.sleep(10)
        self.x0 = run_global(self)
        self.results.update_global(self.x0)
        run_local(self)
        self.best = self.results.epoch_end()
        # raise flag saying that you made an epoch 
        self.event.set()

    # utility functions
    def close(self):
        self.client.shutdown()
    def random_sample(self, N, k,bounds):
        sample = self.rng.random((N, k))
        sample *= bounds[:,1] - bounds[:,0]
        sample += bounds[:,0]
        return sample
    def in_bounds(self, x):
        if (self.bounds[:,1]-x > 0).all() and (self.bounds[:,0] - x < 0).all():
            return True
        return False

