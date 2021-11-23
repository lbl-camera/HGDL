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
###authors: Marcus Noack,David Perryman
###institution: CAMERA @ Lawrence Berkeley National Laboratory



class HGDL:
    """
    This is the HGDL, a class to do asynchronous HPC-ready optimization
    type help(name you gave at import)
    e.g.:
    from hgdl.hgdl import HGDL
    help(HGDL)
    H ... Hybrid
    G ... Global
    D ... Deflated
    L ... Local
    """
    def __init__(self, func, grad, bounds,
            hess = None, num_epochs=100000,
            global_optimizer = "genetic",
            local_optimizer = "dNewton",
            number_of_optima = 1000000,
            radius = None,
            local_max_iter = 100,
            args = (), constr = ()):
        """
        Initialization for the HGDL class

        required parameters:
        ---------------
            func:  the objective function, callable R^dim --> R
            grad:  the objective function gradient, callable R^dim --> R^dim
            bounds: optimization bounds, 2d numpy array

        optional parameters:
        ---------------
            hess = None:  the objective function hessian, R^dim --> R^dim*dim
            num_epochs = 100000: run until num_epochs are completed
            global_optimizer = "genetic"   "genetic"/"gauss"/user defined function (soon Bayes),
                                           use partial() to communicate args to the function
            local_optimizer = "dNewton"    use dNewton or any scipy local optimizer
                                           (recommended: L-BFGS-B, SLSQP, TNC (those allow for bounds)), or your own callable
            number_of_optima               how many optima will be recorded and deflated
            radius = None, means it will be set to the mean of the domain size/100, felation radius
            local_max_iter = 100
            args = (), an n-tuple of parameters, will be communicated to func, grad, hess
            constr = (), define constraints following the format in scipy.optimize.minimize (only foir certain local optimizers)
        """
        self.func = func
        self.grad= grad
        self.hess= hess
        self.bounds = np.asarray(bounds)
        if radius is None: self.radius = np.min(bounds[:,1]-bounds[:,0])/1000.0
        else: self.radius = radius
        self.dim = len(self.bounds)
        self.local_max_iter = local_max_iter
        self.num_epochs = num_epochs
        self.global_optimizer = global_optimizer
        self.local_optimizer = local_optimizer
        self.args = args
        self.constr = constr
        self.optima = optima(self.dim, number_of_optima)
        print("HGDL successfully initiated", flush = True)
        print("deflation radius set to ",self.radius, flush = True)
        if callable(self.hess): print("Hessian was provided by the user:", self.hess)
        print("========================")
    ###########################################################################
    ###########################################################################
    ############USER FUNCTIONS#################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    def optimize(self, dask_client = None, x0 = None):
        """
        optional input:
        -----
            dask_client = None = dask.distributed.Client()
            x0 = None = random.rand()   starting positions
        """

        client = self._init_dask_client(dask_client)
        print(client, flush = True)
        self.x0 = self._prepare_starting_positions(x0)
        #f = np.asarray([self.func(x0[i], *self.args) for i in range(len(x0))])
        print("HGDL starts with: ", self.x0, flush = True)
        #self.optima.fill_in_optima_list(x0,np.ones((len(x0))) * 1e6, np.ones((len(x0))) * 1e6,np.ones((len(x0),self.dim)),[True]*len(x0))
        self.meta_data = meta_data(self)
        self._run_epochs(client)
    ###########################################################################
    def get_client_info(self):
        return self.workers
    ###########################################################################
    def get_latest(self, n = None):
        """
        get n best results

        input:
        -----
            n: number of results requested
        """
        try:
            data, frames = self.transfer_data.get()
            self.optima = distributed.protocol.deserialize(data,frames)
            print("HGDL called get_latest() successfully")
        except:
            self.optima = self.optima
            print("HGDL get_latest failed due to ", str(err))
            print("optima list unchanged")

        optima_list = self.optima.list
        if n is not None: n = min(n,len(optima_list["x"]))
        else: n = len(optima_list["x"])
        print("HGDL get_latest() returned: ",{"x": optima_list["x"][0:n], \
                "func evals": optima_list["func evals"][0:n],
                "classifier": optima_list["classifier"][0:n],
                "eigen values": optima_list["eigen values"][0:n],
                "gradient norm":optima_list["gradient norm"][0:n],
                "success":optima_list["success"]}, flush = True)
        return {"x": optima_list["x"][0:n], \
                "func evals": optima_list["func evals"][0:n],
                "classifier": optima_list["classifier"][0:n],
                "eigen values": optima_list["eigen values"][0:n],
                "gradient norm":optima_list["gradient norm"][0:n],
                "success":optima_list["success"]}
    ###########################################################################
    def get_final(self,n = None):
        """
        get n final results

        input:
        -----
            n: number of results requested
        """
        try:
            self.optima = self.main_future.result()
        except Exception as err:
            print("HGDL get_final failed due to ", str(err))
        optima_list = self.optima.list
        if n is not None: n = min(n,len(optima_list["x"]))
        else: n = len(optima_list["x"])
        return {"x": optima_list["x"][0:n], \
                "func evals": optima_list["func evals"][0:n],
                "classifier": optima_list["classifier"][0:n],
                "eigen values": optima_list["eigen values"][0:n],
                "gradient norm":optima_list["gradient norm"][0:n],
                "success":optima_list["success"]}
    ###########################################################################
    def cancel_tasks(self, n = None):
        """
        cancel tasks but leave client alive
        return:
        -------
            latest results
        """
        print("HGDL is cancelling all tasks...")
        res = self.get_latest(n)
        self.break_condition.set(True)
        self.client.cancel(self.main_future)
        print("Status of HGDL task: ", self.main_future.status)
        print("This leaves the client alive.")
        return res
    ###########################################################################
    def kill_client(self, n = None):
        """
        kill tasks and shutdown client
        return:
        -------
            latest results
        """
        print("HGDL kill client initialized ...")
        res = self.get_latest(n)
        try:
            self.break_condition.set(True)
            self.client.gather(self.main_future)
            self.client.cancel(self.main_future)
            del self.main_future
            self.client.shutdown()
            self.client.close()
            print("HGDL kill client successful")
        except Exception as err:
            print("HGDL kill failed")
            print(str(err))
        time.sleep(0.1)
        return res
    ###########################################################################
    ############USER FUNCTIONS END#############################################
    ###########################################################################
    def _prepare_starting_positions(self,x0):
        if x0 is None: x0 = misc.random_population(self.bounds,self.number_of_walkers)
        elif x0.ndim == 1: x0 = np.array([x0])
        if len(x0) < self.number_of_walkers:
            x0_aux = np.zeros((self.number_of_walkers,len(x0[0])))
            x0_aux[0:len(x0)] = x0
            x0_aux[len(x0):] = misc.random_population(self.bounds,self.number_of_walkers - len(x0))
            x0 = x0_aux
        elif len(x0) > self.number_of_walkers:
            x0 = x0[0:self.number_of_walkers]
        else: x0 = x0
        return x0
    ###########################################################################
    def _init_dask_client(self,dask_client):
        if dask_client is None: 
            dask_client = dask.distributed.Client()
            print("No dask client provided to HGDL. Using the local client", flush = True)
        else: print("dask client provided to HGDL", flush = True)
        client = dask_client
        worker_info = list(client.scheduler_info()["workers"].keys())
        if not worker_info: raise Exception("No workers available")
        self.workers = {"host": worker_info[0],
                "walkers": worker_info[1:]}
        print("Host ",self.workers["host"]," has ", len(self.workers["walkers"])," workers.")
        self.number_of_walkers = len(self.workers["walkers"])
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
                "optima":self.optima, "metadata":self.meta_data}
        bf = client.scatter(data, workers = self.workers["host"])
        self.main_future = client.submit(hgdl, bf, workers = self.workers["host"])
        self.client = client
###########################################################################
###########################################################################
##################hgdl functions###########################################
###########################################################################
###########################################################################
def hgdl(data):
    metadata = data["metadata"]
    transfer_data = data["transfer data"]
    break_condition = data["break condition"]
    optima = data["optima"]
    print("HGDL computing epoch 1 of ",metadata.num_epochs, flush = True)
    optima = run_local(metadata,optima,metadata.x0)
    a = distributed.protocol.serialize(optima)
    transfer_data.set(a)

    for i in range(1,metadata.num_epochs):
        bc = break_condition.get()
        if bc is True: print("HGDL Epoch ",i," was cancelled", flush = True);break
        print("HGDL computing epoch ",i+1," of ",metadata.num_epochs, flush = True)
        optima = run_hgdl_epoch(metadata,optima)
        a = distributed.protocol.serialize(optima)
        transfer_data.set(a)
    return optima
###########################################################################
def run_hgdl_epoch(metadata,optima):
    optima_list = optima.list
    n = min(len(optima_list["x"]),metadata.number_of_walkers)
    x0 = run_global(\
            np.array(optima_list["x"][0:n,:]),
            np.array(optima_list["func evals"][0:n]),
            metadata.bounds, metadata.global_optimizer,metadata.number_of_walkers)
    x0 = np.array(x0)
    optima = run_local(metadata,optima,x0)
    return optima
