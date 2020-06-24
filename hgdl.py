# coding: utf-8

#  imports
import numpy as np
from Global.run_global import run_global
from Local.run_local import run_local
from utility import in_bounds
from results import Results
from multiprocessing import Process, Queue, Lock
from time import sleep

class HGDL(object):
    def __init__(self, *args, **kwargs):
        self.queue = Queue()
        self.lock = Lock()
        self.hgdl = HGDL_worker(
            hgdl=self,
            *args, **kwargs)
        self.worker = Process(target=self.hgdl.run)
        self.worker.start()
        self.done = False

    def get_best(self):
        if self.done:
            result = {
                    "best_x":self.result["best_x"],
                    "best_y":self.result["best_y"],
                    }
        else:
            self.lock.acquire()
            res = self.queue.get()
            self.lock.release()
            if res["final"]:
                self.done = True
                self.result = res
            result = {
                    "best_x":res["best_x"],
                    "best_y":res["best_y"]
                    }
        return result

    def get_final(self):
        self.worker.join()
        if self.done:
            return self.result
        else:
            self.lock.acquire()
            res = self.queue.get()
            self.lock.release()
            return res

class HGDL_worker(object):
    """
    HGDL
        * Hybrid - uses both local and global optimization
        * G - uses global optimizer
        * D - uses deflation
        * L - uses local extremum localMethod
    """
    def __init__(
            self, func, grad, hess, bounds, hgdl, r=.3, alpha=.1, max_epochs=5,
            num_individuals=15, max_local=5, num_workers=None, bestX=5,
            x0=None, global_method='genetic', local_method='scipy',
            ):
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
        self.rng = np.random.default_rng(42)
        self.lock = hgdl.lock
        self.queue = hgdl.queue
        self.func = func
        self.grad = grad
        self.hess = hess
        self.bounds = bounds
        self.k = len(bounds)
        self.r = r
        self.alpha = alpha
        self.max_epochs = max_epochs
        self.max_local = max_local
        self.num_individuals = num_individuals
        if num_workers is None:
            from psutil import cpu_count
            num_workers = cpu_count(logical=False)-1
        self.num_workers = num_workers
        self.bestX = bestX
        self.results = Results(self)
        if x0 is None:
            x0 = self.random_sample(self.num_individuals, self.k, self.bounds)
        self.x0 = x0
        self.global_method = global_method
        self.local_method = local_method
        self.results.update_genetic(self.x0)
        self.in_bounds = in_bounds

    def run(self):
        for i in range(self.max_epochs):
            self.x0 = run_global(self)
            self.results.update_genetic(self.x0)
            run_local(self)
            self.lock.acquire()
            while not self.queue.empty(): self.queue.get()
            self.queue.put(self.results.epoch_end())
            self.lock.release()
            sleep(10)
        self.lock.acquire()
        while not self.queue.empty(): self.queue.get()
        self.queue.get()
        self.queue.put(self.results.roll_up())
        self.lock.release()

    def random_sample(self, N, k,bounds):
        sample = self.rng.random((N, k))
        sample *= bounds[:,1] - bounds[:,0]
        sample += bounds[:,0]
        return sample

