import numpy as np
import torch as t
import time
import hgdl.misc as misc
from hgdl.local_methods.local_optimizer import run_local_optimizer
from hgdl.global_methods.global_optimizer import run_global
from hgdl.local_methods.local_optimizer import run_local
from hgdl.result.optima  import optima
import dask.distributed as distributed
import dask.multiprocessing
from dask.distributed import as_completed
###authors: David Perryman, Marcus Noack
###institution: CAMERA @ Lawrence Berkeley National Laboratory
import pickle


"""
TODO:   *currently walkers that walk out in Newton are discarded. We should do a line search instead
        *the radius is still ad hoc, should be related to curvature
"""


class HGDL:
    """
    doc string here
    """
    def __init__(self,obj_func,grad_func,hess_func, 
            bounds, maxEpochs=100000,
            local_optimizer = "newton",
            global_optimizer = "genetic",
            radius = 0.1, global_tol = 1e-4,
            local_max_iter = 20, global_max_iter = 120,
            number_of_walkers = 20, x0 = None, 
            args = (), verbose = False):
        """
        intialization for the HGDL class

        required input:
        ---------------
            obj_func
            grad_func
            hess_func
            bounds
        optional input:
        ---------------
            maxEpochs = 100000
            local_optimizer = "newton"   "newton"/user defined function, 
                                         use partial to communicate args to the function
            global_optimizer = "genetic"   "genetic"/"gauss"/user defined function,
                                           use partial to communicate args to the function
            radius = 20
            global_tol = 1e-4
            local_max_iter = 20
            global_max_iter = 20
            number_of_walkers: make sure you have enough workers for
                               your walkers ( walkers + 1 <= workers)
            x0 = np.rand.random()
            args = (), a n-tuple of parameters, will be communicated to obj_func, grad, hess
            verbose = False
        """
        self.obj_func = obj_func
        self.grad_func= grad_func
        self.hess_func= hess_func
        self.bounds = np.asarray(bounds)
        self.r = radius
        self.dim = len(self.bounds)
        self.global_tol = global_tol
        self.local_max_iter = local_max_iter
        self.global_max_iter = global_max_iter
        self.number_of_walkers = number_of_walkers
        self.maxEpochs = maxEpochs
        self.local_optimizer = local_optimizer
        self.global_optimizer = global_optimizer
        if x0 is None: x0 = misc.random_population(self.bounds,self.number_of_walkers)
        if len(x0) != self.number_of_walkers: exit("number of initial position != number of walkers")
        self.x0 = x0
        self.args = args
        self.verbose = verbose
        ########################################
        #init optima list:
        self.optima = optima(self.dim)
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    def optimize(self, dask_client = None):
        """
        optional input:
        -----
            dask_client = dask.distributed.Client()
        """
        if dask_client is None: dask_client = dask.distributed.Client()
        client = dask_client
        if dask_client is False:
            ###this should be run local_optimizer
            x,f,grad_norm,eig,success = run_local_optimizer(self.obj_func,
                self.grad_func,self.hess_func,
                self.bounds,self.r,self.local_max_iter,self.local_optimizer,
                self.x0,self.args)
        else:
            self.main_future = client.submit(run_local_optimizer,self.obj_func,
                self.grad_func,self.hess_func,
                self.bounds,self.r,self.local_max_iter,self.local_optimizer,
                self.x0,self.args)
            x,f,grad_norm,eig,success = self.main_future.result()
        print("HGDL engine started: ")
        #print(self.x0)
        #print("")
        #print("")
        print("")
        print("I found ",len(np.where(success == True)[0])," optima in my first run")
        if len(np.where(success == True)[0]) == 0:
            print("no optima found")
            success[:] = True
        print("They are now stored in the optima_list")
        self.optima.fill_in_optima_list(x,f,grad_norm,eig, success)
        if self.verbose == True: print(optima_list)
        #################################
        if self.verbose == True: print("Submitting main hgdl task")
        if dask_client is False and self.maxEpochs != 0:
            self.transfer_data = False
            hgdl(self.transfer_data,self.optima,self.obj_func,
                self.grad_func,self.hess_func,
                self.bounds,self.maxEpochs,self.r,self.local_max_iter,
                self.global_max_iter,self.local_optimizer,self.global_optimizer,
                self.number_of_walkers,self.args, self.verbose)
        elif dask_client is not False and self.maxEpochs != 0:
            self.transfer_data = distributed.Variable("transfer_data",client)
            self.main_future = client.submit(hgdl,self.transfer_data,self.optima,self.obj_func,
                self.grad_func,self.hess_func,
                self.bounds,self.maxEpochs,self.r,self.local_max_iter,
                self.global_max_iter,self.local_optimizer,self.global_optimizer,
                self.number_of_walkers,self.args, self.verbose)
            self.client = client
        else:
            client.cancel(self.main_future)
            client.shutdown()

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
        except:
            return self.optima.list
        self.optima = distributed.protocol.deserialize(data,frames)
        optima_list = self.optima.list
        n = min(n,len(optima_list["x"]))
        return {"x": optima_list["x"][0:n], \
                "func evals": optima_list["func evals"][0:n],
                "classifier": optima_list["classifier"][0:n],
                "eigen values": optima_list["eigen values"][0:n],
                "gradient norm":optima_list["gradient norm"][0:n]}
    ###########################################################################
    def get_final(self,n):
        """
        get n final results

        input:
        -----
            n: number of results requested
        """
        self.optima = self.main_future.result()
        optima_list = self.optima.list
        n = min(n,len(optima_list["x"]))
        return {"x": optima_list["x"][0:n], \
                "func evals": optima_list["func evals"][0:n],
                "classifier": optima_list["classifier"][0:n],
                "eigen values": optima_list["eigen values"][0:n],
                "gradient norm":optima_list["gradient norm"][0:n]}
    ###########################################################################
    def cancel_tasks(self):
        """
        cancel tasks but leave client alive
        return:
        -------
            latest results
        """
        res = self.get_latest(-1)
        self.client.cancel(self.main_future)
        print("All HGDL tasks cancelled.")
        print("This leaves the client alive.")
        return res
    ###########################################################################
    def kill(self):
        """
        kill tasks and shutdown client
        return:
        -------
            latest results
        """
        print("Kill initialized ...")
        res = self.get_latest(-1)
        self.client.cancel(self.main_future)
        self.client.shutdown()
        return res

###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
def hgdl(transfer_data,optima,obj_func,grad_func,hess_func,
                bounds,maxEpochs,r,local_max_iter,
                global_max_iter,local_method,global_method,number_of_walkers,args, verbose):
    if verbose is True: print("    Starting ",maxEpochs," epochs.")
    for i in range(maxEpochs):
        print("Computing epoch ",i," of ",maxEpochs)
        optima = run_hgdl_epoch(
            obj_func,grad_func,hess_func,bounds,optima,
            r,local_max_iter,global_max_iter,local_method,global_method,
            number_of_walkers,args,verbose)
        if transfer_data is not False:
            a = distributed.protocol.serialize(optima)
            transfer_data.set(a)
        if verbose is True: print("    Epoch ",i," finished")
    return optima
###########################################################################
def run_hgdl_epoch(func,grad,hess,bounds,optima_obj,radius,
        local_max_iter,global_max_iter,local_method,
        global_method,number_of_walkers,args,verbose):
    """
    an epoch is one local run and one global run,
    where one local run are several convergence runs of all workers from
    the x_init point
    """
    optima_list  = optima_obj.list
    n = len(optima_list["x"])
    nn = min(n,number_of_walkers)
    if verbose is True: print("    global step started")
    x0 = run_global(\
            np.array(optima_list["x"][0:nn,:]),
            np.array(optima_list["func evals"][0:nn]),
            bounds, global_method,number_of_walkers,verbose)
    if verbose is True: print("    global step finished")
    if verbose is True: print("    local step started")
    optima = run_local(func,grad,hess,bounds,radius,
            local_max_iter, global_max_iter,local_method,
            x0,optima_obj,args,verbose)
    if verbose is True: print("    local step finished")
    return optima


