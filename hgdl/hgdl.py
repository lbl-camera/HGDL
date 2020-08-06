import numpy as np
import torch as t
import time
import hgdl.misc as misc
import hgdl.local as local
import hgdl.glob as glob
import hgdl.hgdl_functions as hgdl_functions
from functools import partial
from multiprocessing import Process, Lock
from multiprocessing import Queue as mQueue
import dask.distributed as distributed
import asyncio
from psutil import cpu_count
import threading
import dask.multiprocessing
from multiprocessing.pool import ThreadPool
from dask.distributed import as_completed
###authors: David Perryman, Marcus Noack
###institution: CAMERA @ Lawrence Berkeley National Laboratory


"""
TODO:   *currently walkers that walk out in Newton are discarded. We should do a line search instead
        *the radius is still ad hoc, should be related to curvature
        *work on the shut down
"""
class HGDL:
    """
    doc string here
    """
    def __init__(self,obj_func,grad_func,hess_func, bounds,dask_client = None, maxEpochs=10,
            radius = 20.0,local_tol = 1e-4, global_tol = 1e-4,
            local_max_iter = 20, global_max_iter = 120,
            number_of_walkers = 20,
            number_of_workers = None, x0 = None, 
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
            dask_client = give custom dask client or it will be intialized to Client()
            maxEpochs = 10
            radius = 20
            local_tol  = 1e-4
            global_tol = 1e-4
            local_max_iter = 20
            global_max_iter = 20
            x0 = np.rand.random()
            args = (), a n-tuple of parameters, will be communicated to obj func, grad, hess
        """
        self.obj_func = obj_func
        self.grad_func= grad_func
        self.hess_func= hess_func
        self.bounds = np.asarray(bounds)
        self.client = dask_client
        self.r = radius
        self.dim = len(self.bounds)
        self.local_tol = local_tol
        self.global_tol = global_tol
        self.local_max_iter = local_max_iter
        self.global_max_iter = global_max_iter
        self.number_of_walkers = number_of_walkers
        self.maxEpochs = maxEpochs
        if dask_client is None: dask_client = dask.distributed.Client()
        self.client = dask_client
        if x0 is None: x0 = misc.random_population(self.bounds,self.number_of_walkers)
        if len(x0) != self.number_of_walkers: exit("number of initial position != number of walkers")
        self.args = args
        self.verbose = verbose
        ########################################
        #init optima list:
        optima_list = {"x": np.empty((0,self.dim)), 
                "func evals": np.empty((0)), 
                "classifier": [], "eigen values": np.empty((0,self.dim)), 
                "gradient norm":np.empty((0))}
        ####################################
        self.main_future = self.client.submit(hgdl_functions.run_dNewton,obj_func,
                grad_func,hess_func,
                np.array(bounds),radius,local_max_iter,
                x0,args)
        x,f,grad_norm,eig,success = self.main_future.result()
        print("HGDL starting positions: ")
        print(x0)
        print("")
        print("")
        print("")
        print("I found ",len(np.where(success == True)[0])," optima in my first run")
        if len(np.where(success == True)[0]) == 0: 
            print("no optima found")
            success[:] = True
        print("They are now stored in the optima_list")
        optima_list = hgdl_functions.fill_in_optima_list(optima_list,1e-6,x,f,grad_norm,eig, success)
        if verbose == True: print(optima_list)
        #################################
        self.transfer_data = distributed.Variable("transfer_data",self.client)
        #self.break_out = distributed.Variable("break_out",self.client)
        if verbose == True: print("Submitting main hgdl task")
        self.main_future = self.client.submit(hgdl_functions.hgdl,self.transfer_data,optima_list,obj_func,
                grad_func,hess_func,
                np.array(bounds),maxEpochs,radius,local_max_iter,
                global_max_iter,number_of_walkers,args, verbose)
        ####no multithreading:
        #hgdl_functions.hgdl(optima_list,obj_func, grad_func,hess_func,
        #        np.array(bounds),maxEpochs,radius,local_max_iter,
        #        global_max_iter,number_of_walkers,args, verbose)
    ###########################################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    def get_latest(self, n):
        data, frames = self.transfer_data.get()
        optima_list = distributed.protocol.deserialize(data,frames)
        return {"x": optima_list["x"][0:n], \
                "func evals": optima_list["func evals"][0:n],
                "classifier": optima_list["classifier"][0:n],
                "eigen values": optima_list["eigen values"][0:n],
                "gradient norm":optima_list["gradient norm"][0:n]}
    ###########################################################################
    def get_final(self,n):
        optima_list = self.main_future.result()
    ###########################################################################
    def kill(self):
        print("Shutdown initialized ...")
        res = self.get_latest(-1)
        self.client.cancel(self.main_future)
        self.client.shutdown()
        return res

