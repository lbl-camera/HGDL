import numpy as np
import torch as t
import time
import hgdl.misc as misc
from hgdl.local_methods.local_optimizer import run_local_optimizer
from hgdl.global_methods.global_optimizer import run_global
from hgdl.local_methods.local_optimizer import run_local
import dask.distributed as distributed
import dask.multiprocessing
from dask.distributed import as_completed
from hgdl.result.optima  import optima
from hgdl.meta_data  import meta_data
import pickle
###authors: David Perryman, Marcus Noack
###institution: CAMERA @ Lawrence Berkeley National Laboratory


"""
TODO:   *the radius is still ad hoc, should be related to curvature and unique to a deflation point
"""

class HGDL:
    """
    This is the HGDL class, a class to do asynchronous HPC-ready optimization
    type help(name you gave at import)
    e.g.:
    from hgdl.hgdl import HGDL
    help(HGDL)
    H ... Hybrid
    G ... Global
    D ... Deflated
    L ... Local

    """
    def __init__(self,func,grad,hess,
            bounds, num_epochs=100000,
            global_optimizer = "genetic",
            radius = 0.1, global_tol = 1e-4,
            local_max_iter = 20,
            args = (), verbose = False):
        """
        intialization for the HGDL class

        required parameters:
        ---------------
            func:  the objective function
            grad:  the objective function gradient
            hess:  the objective function hessian
            bounds:the bounds of the optimization

        optional parameters:
        ---------------
            num_epochs = 100000
            global_optimizer = "genetic"   "genetic"/"gauss"/user defined function,
                                           use partial to communicate args to the function
            radius = 20
            global_tol = 1e-4
            local_max_iter = 20
            args = (), a n-tuple of parameters, will be communicated to func, grad, hess
            verbose = False
        """

        self.func = func
        self.grad= grad
        self.hess= hess
        self.bounds = np.asarray(bounds)
        self.radius = radius
        self.dim = len(self.bounds)
        self.global_tol = global_tol
        self.local_max_iter = local_max_iter
        self.num_epochs = num_epochs
        self.global_optimizer = global_optimizer
        self.args = args
        self.verbose = verbose
        self.optima = optima(self.dim)
    ###########################################################################
    ###########################################################################
    ############USER FUNCTIONS#################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    def optimize(self, dask_client = True, 
            x0 = None, number_of_walkers = None,
            number_of_starting_positions = None):
        """
        optional input:
        -----
            dask_client = True = dask.distributed.Client()
            x0 = None = random.rand()   starting positions
            number_of_walkers: make sure you have enough workers for
                               your walkers ( walkers + 1 <= workers)
                               otherwise ignore for the correct assignment
            number_of_starting_positions: number of walkers for first epoch if different
        """

        client = self._init_dask_client(dask_client,number_of_walkers,number_of_starting_positions)
        x0 = self._prepare_starting_positions(x0)
        self.meta_data = meta_data(self)
        #####run first local######################
        print("HGDL engine started")
        print("This will block the main thread until the first epoch has concluded its work")
        x,f,grad_norm,eig,success = self._run_first_local_optimization(client,x0)
        print("I found ",len(np.where(success == True)[0])," optima in my first run")
        if len(np.where(success == True)[0]) == 0: success[:] = True; self.optima.list["success"] = False
        else: self.optima.list["success"] = True
        self.optima.fill_in_optima_list(x,f,grad_norm,eig, success)
        if self.verbose == True: print(optima_list)
        #################################
        ####run epochs###################
        #################################
        self._run_epochs(client)
    ###########################################################################
    def get_client_info(self):
        return self.workers
    ###########################################################################
    def get_latest(self, n):
        """
        get n best results

        input:
        -----
            n: number of results requested
        """
        try:
            data, frames = self.transfer_data.get()
            self.optima = distributed.protocol.deserialize(data,frames)
        except:
            self.optima = self.optima
        optima_list = self.optima.list
        n = min(n,len(optima_list["x"]))
        return {"x": optima_list["x"][0:n], \
                "func evals": optima_list["func evals"][0:n],
                "classifier": optima_list["classifier"][0:n],
                "eigen values": optima_list["eigen values"][0:n],
                "gradient norm":optima_list["gradient norm"][0:n],
                "success":optima_list["success"]}
    ###########################################################################
    def get_final(self,n):
        """
        get n final results

        input:
        -----
            n: number of results requested
        """
        try:
            self.optima = self.main_future.result()
        except:
            pass
        optima_list = self.optima.list
        n = min(n,len(optima_list["x"]))
        return {"x": optima_list["x"][0:n], \
                "func evals": optima_list["func evals"][0:n],
                "classifier": optima_list["classifier"][0:n],
                "eigen values": optima_list["eigen values"][0:n],
                "gradient norm":optima_list["gradient norm"][0:n],
                "success":optima_list["success"]}
    ###########################################################################
    def cancel_tasks(self):
        """
        cancel tasks but leave client alive
        return:
        -------
            latest results
        """
        res = self.get_latest(-1)
        self.break_condition.set(True)
        while self.main_future.status != "finished":
            time.sleep(0.1)
        self.client.cancel(self.main_future)
        print("Status of HGDL task: ", self.main_future.status)
        print("This leaves the client alive.")
        return res
    ###########################################################################
    def kill(self, n= -1):
        """
        kill tasks and shutdown client
        return:
        -------
            latest results
        """
        print("Kill initialized ...")
        res = self.get_latest(n)
        try:
            self.break_condition.set(True)
            while self.main_future.status == "pending":
                time.sleep(0.1)
            self.client.gather(self.main_future)
            self.client.cancel(self.main_future)
            del self.main_future
            self.client.shutdown()
            self.client.close()
            print("kill successful")
        except Exception as err:
            print(err)
            print("kill failed")
        time.sleep(0.1)
        return res
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ############USER FUNCTIONS END#############################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    def _prepare_starting_positions(self,x0):
        if x0 is None: x0 = misc.random_population(self.bounds,self.number_of_walkers)
        elif len(x0) < self.number_of_walkers: 
            x0 = np.empty((self.number_of_walkers,len(x0[0])))
            x0[0:len(x0)] = x0
            x0[len(x0):] = misc.random_population(self.bounds,self.number_of_walkers - len(x0))
        elif len(x0) > self.number_of_walkers:
            x0 = x0[0:self.number_of_walkers]
        else: x0 = x0
        return x0
    ###########################################################################
    def _prepare_dask_client(self,dask_client):
        if dask_client is True: dask_client = dask.distributed.Client()
        client = dask_client
        return client
    def _init_dask_client(self,dask_client,number_of_walkers,number_of_starting_positions):
        if dask_client is None:
            raise Exception("dask_client is None, can only be True/False or a distributed.Client(...)")
        client = self._prepare_dask_client(dask_client)
        worker_info = list(client.scheduler_info()["workers"].keys())
        self.workers = {"host": worker_info[0],
                "walkers": worker_info[1:]}
        if number_of_walkers is None: number_of_walkers = len(self.workers["walkers"])
        if number_of_starting_positions is None: number_of_starting_positions = number_of_walkers
        self.number_of_starting_positions = number_of_starting_positions
        self.number_of_walkers = number_of_walkers
        return client
    ###########################################################################
    def _run_first_local_optimization(self,client,x0):
        if client is False:
            x,f,grad_norm,eig,success = run_local_optimizer(d)
        else:
            d = self.meta_data
            bf = client.scatter(d,workers = [self.workers["host"]]+self.workers["walkers"], broadcast = True)
            self.main_future = client.submit(run_local_optimizer,bf,x0, workers = self.workers["host"])
            x,f,grad_norm,eig,success = self.main_future.result()
        return x,f,grad_norm,eig,success
    ###########################################################################
    def _run_epochs(self,client):
        dask_client = client
        if self.verbose == True: print("Submitting main hgdl task")
        if dask_client is False:
            self.transfer_data = False
            self.break_condition = False
            data = {"transfer data":self.transfer_data,
                 "break condition":self.break_condition,
                 "optima":self.optima, "d":self.meta_data}
            hgdl(data)
        else:
            self.break_condition = distributed.Variable("break_condition",client)
            self.transfer_data = distributed.Variable("transfer_data",client)
            a = distributed.protocol.serialize(self.optima)
            self.transfer_data.set(a)
            self.break_condition.set(False)
            data = {"transfer data":self.transfer_data,
                 "break condition":self.break_condition,
                 "optima":self.optima, "d":self.meta_data}
            bf = client.scatter(data, workers = self.workers["host"])
            self.main_future = client.submit(hgdl, bf, workers = self.workers["host"])
            self.client = client
###########################################################################
###########################################################################
##################hgdl functions###########################################
###########################################################################
###########################################################################
def hgdl(data):
    d = data["d"]
    transfer_data = data["transfer data"]
    break_condition=data["break condition"]
    optima = data["optima"]
    if d.verbose is True: print("    Starting ",d.num_epochs," epochs.")
    for i in range(d.num_epochs):
        if break_condition is not False: bc = break_condition.get()
        else: bc = False
        if bc is True: print("Epoch ",i," was cancelled");break
        print("Computing epoch ",i," of ",d.num_epochs)
        optima = run_hgdl_epoch(d,optima)
        if transfer_data is not False:
            a = distributed.protocol.serialize(optima)
            transfer_data.set(a)
        if d.verbose is True: print("    Epoch ",i," finished")
    return optima
###########################################################################
def run_hgdl_epoch(d,optima):
    """
    an epoch is one local run and one global run,
    where one local run are several convergence runs of all workers from
    the x_init points
    """
    optima_list = optima.list
    nn = min(len(optima_list["x"]),d.number_of_walkers)
    if d.verbose is True: print("    global step started")
    x0 = run_global(\
            np.array(optima_list["x"][0:nn,:]),
            np.array(optima_list["func evals"][0:nn]),
            d.bounds, d.global_optimizer,d.number_of_walkers,d.verbose)
    if d.verbose is True: print("    global step finished")
    if d.verbose is True: print("    local step started")
    x0 = np.array(x0)
    optima = run_local(d,optima,x0)
    optima.list["success"] = True
    if d.verbose is True: print("    local step finished")
    return optima
