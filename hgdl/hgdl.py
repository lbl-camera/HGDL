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
from hgdl.optima  import optima
from hgdl.meta_data  import meta_data
import pickle
###authors: David Perryman, Marcus Noack
###institution: CAMERA @ Lawrence Berkeley National Laboratory



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
    def __init__(self,func,grad,hess = None,
            bounds = None, num_epochs=100000,
            global_optimizer = "genetic",
            local_optimizer = "dNewton",
            number_of_optima = 100,
            radius = 1.0, global_tol = 1e-4,
            local_max_iter = 20,
            args = (), constr = ()):
        """
        intialization for the HGDL class

        required parameters:
        ---------------
            func:  the objective function
            grad:  the objective function gradient

        optional parameters:
        ---------------
            hess:  the objective function hessian, default = None
            bounds:the bounds of the optimization, default = None

            num_epochs = 100000
            global_optimizer = "genetic"   "genetic"/"gauss"/user defined function (soon Bayes),
                                           use partial() to communicate args to the function
            local_optimizer = "dNewton"    use dNewton, any scipy local optimizer 
                                           (recommended: L-BFGS-B, SLSQP, TNC (those allow for bounds)), or your own callable
            number_of_optima               how many optima will be recorded and deflated
            radius = 0.1
            global_tol = 1e-4
            local_max_iter = 20
            args = (), an n-tuple of parameters, will be communicated to func, grad, hess
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
        self.local_optimizer = local_optimizer
        self.args = args
        self.constr = constr
        self.optima = optima(self.dim, number_of_optima)
    ###########################################################################
    ###########################################################################
    ############USER FUNCTIONS#################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    def optimize(self, dask_client = None,
            x0 = None, number_of_walkers = None):
        """
        optional input:
        -----
            dask_client = None = dask.distributed.Client()
            x0 = None = random.rand()   starting positions
            number_of_walkers = None, make sure you have enough workers for
                               your walkers ( walkers + 1 <= workers)
                               otherwise ignore for the assignment
        """

        client = self._init_dask_client(dask_client,number_of_walkers)
        x0 = self._prepare_starting_positions(x0)
        #f = np.asarray([self.func(x0[i], *self.args) for i in range(len(x0))])
        print("HGDL starts with: ", x0)
        self.optima.fill_in_optima_list(x0,np.ones((len(x0))) * 1e6, np.ones((len(x0))) * 1e6,np.ones((len(x0),self.dim)),[True]*len(x0))
        self.meta_data = meta_data(self)
        self._run_epochs(client)
    ###########################################################################
    def get_client_info(self):
        return self.workers
    ###########################################################################
    def get_latest(self, n = -1):
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
    def get_final(self,n = -1):
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
        print("HGDL is cancelling all tasks...")
        res = self.get_latest(-1)
        self.break_condition.set(True)
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
        print("HGDL kill initialized ...")
        res = self.get_latest(n)
        try:
            self.break_condition.set(True)
            self.client.gather(self.main_future)
            self.client.cancel(self.main_future)
            del self.main_future
            self.client.shutdown()
            self.client.close()
            print("HGDL kill successful")
        except Exception as err:
            print(err)
            print("HGDL kill failed")
        time.sleep(0.1)
        return res
    ###########################################################################
    ############USER FUNCTIONS END#############################################
    ###########################################################################
    def _prepare_starting_positions(self,x0):
        if x0 is None: x0 = misc.random_population(self.bounds,self.number_of_walkers)
        if x0.ndim == 1: x0 = np.array([x0])
        elif len(x0) < self.number_of_walkers:
            x0_aux = np.zeros((self.number_of_walkers,len(x0[0])))
            x0_aux[0:len(x0)] = x0
            x0_aux[len(x0):] = misc.random_population(self.bounds,self.number_of_walkers - len(x0))
            x0 = x0_aux
        elif len(x0) > self.number_of_walkers:
            x0 = x0[0:self.number_of_walkers]
        else: x0 = x0
        return x0
    ###########################################################################
    def _init_dask_client(self,dask_client,number_of_walkers):
        if dask_client is None: dask_client = dask.distributed.Client()
        client = dask_client
        worker_info = list(client.scheduler_info()["workers"].keys())
        self.workers = {"host": worker_info[0],
                "walkers": worker_info[1:]}
        if number_of_walkers is None: number_of_walkers = len(self.workers["walkers"])
        self.number_of_walkers = number_of_walkers
        return client
    ###########################################################################
    def _run_epochs(self,client):
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
        #hgdl(data)
        self.client = client
###########################################################################
###########################################################################
##################hgdl functions###########################################
###########################################################################
###########################################################################
def hgdl(data):
    d = data["d"]
    transfer_data = data["transfer data"]
    break_condition = data["break condition"]
    optima = data["optima"]
    for i in range(d.num_epochs):
        bc = break_condition.get()
        if bc is True: print("HGDL Epoch ",i," was cancelled");break
        print("HGDL computing epoch ",i+1," of ",d.num_epochs)
        optima = run_hgdl_epoch(d,optima)
        a = distributed.protocol.serialize(optima)
        transfer_data.set(a)
    return optima
###########################################################################
def run_hgdl_epoch(d,optima):
    optima_list = optima.list
    n = min(len(optima_list["x"]),d.number_of_walkers)
    x0 = run_global(\
            np.array(optima_list["x"][0:n,:]),
            np.array(optima_list["func evals"][0:n]),
            d.bounds, d.global_optimizer,d.number_of_walkers)
    x0 = np.array(x0)
    optima = run_local(d,optima,x0)
    return optima
