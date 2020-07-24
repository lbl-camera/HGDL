import numpy as np
import torch as t
import time
import hgdl.misc as misc
import hgdl.local as local
import hgdl.glob as glob
from multiprocessing import Process, Queue, Lock
import dask.distributed
import asyncio
from psutil import cpu_count

###authors: David Perryman, Marcus Noack
###institution: CAMERA @ Lawrence Berkeley National Laboratory


"""
TODO:   *currently walkers that walk out in Newton are discarded. We should do a line search instead
        *currently individuals are replaced randomly, we should do the genetic step or the global_gaussian_pdf step
        *the radius is still ad hoc, should be related to curvature
        *need a good global break condition
"""

class HGDL:
    """
    doc string here
    """
    def __init__(self,obj_func,grad_func,hess_func, bounds,dask_client = None, maxEpochs=10,
            radius = 20.0,local_tol = 1e-4, global_tol = 1e-4,
            local_max_iter = 20, global_max_iter = 120,
            number_of_walkers = 20, number_of_optima = 10,
            number_of_workers = None, x0 = None, 
            argument_dict = None):
        """
        intialization for the HGDL class

        input:
        ------
        Nothing has to be input. If "HGDL()" then Schefel's
        function is used as example. If this is not a test and 
        a user defined function has to be called the parameters are:
            obj_func: function that calculates the scalar function value at a ppoint x
            grad_func: function that calculates the gradient at a point x
            hess_func: function that calculates the hessian at a point x
            bounds(numpy array): array of bounds, len is the dimensionality of the problem
        """
        self.obj_func = obj_func
        self.grad_func = grad_func
        self.hess_func = hess_func
        self.bounds = np.asarray(bounds)
        self.client = dask_client
        self.r = radius
        self.dim = len(self.bounds)
        self.local_tol = local_tol
        self.global_tol = global_tol
        self.local_max_iter = local_max_iter
        self.global_max_iter = global_max_iter
        self.number_of_walkers = number_of_walkers
        self.number_of_optima = number_of_optima
        self.maxEpochs = maxEpochs
        if dask_client is None: client = dask.distributed.Client()
        self.client = client
        if number_of_workers is None: number_of_workers = cpu_count(logical=False)-1
        self.number_of_workers = number_of_workers
        if x0 is None:x0 = misc.random_population(self.bounds,self.number_of_walkers)
        self.x0 = x0
        if len(self.x0) != self.number_of_walkers: exit("number of initial position != number of walkers")
        self.argument_dict = argument_dict
        ########################################
        #init optima list:
        self.optima_list = {"x": np.empty((0,self.dim)), \
                "func evals": np.empty((0)), \
                "classifier": [], "eigen values": np.empty((0,self.dim)), \
                "gradient norm":np.empty((0))}
        ####################################
        #first run
        x,f,grad_norm,eig,success = self.run_dNewton(self.x0)
        print("HGDL starting positions: ")
        print(self.x0)
        print("I found ",len(np.where(success == True)[0])," optima in my first run")
        print("They are now stored in the optima_list")
        self.optima_list = self.fill_in_optima_list(self.optima_list,x,f,grad_norm,eig, success)
        #################################
        #exit()

        #self.tasks = [None] * self.maxEpochs
        #self.hgdl()
        #self.tasks = [self.loop.create_task(self.hgdl())]
        self.tasks = []
        self.loop = asyncio.get_event_loop()
        #self.loop.run_until_complete(*self.tasks)
        self.loop.run_until_complete(self.create_list())
        #print(res)
    ###########################################################################
    async def create_list(self):
        await self.hgdl()
    ###########################################################################
    async def hgdl(self):
        for i in range(self.maxEpochs):
            await self.run_hgdl_epoch()
    ###########################################################################
    def get_final(self):
        # wait until everything is done
        self.loop.run_until_complete(asyncio.gather(*self.tasks))
        self.close()
        return self.optima_list
    ###########################################################################
    def get_best(self):
        # wait until at least one epoch is done 
        self.loop.run_until_complete(asyncio.gather(self.tasks[0]))
        return self.best
    ###########################################################################
    def get_latest(self):
        return self.optima_list
    ###########################################################################
    def kill(self):
        logging.info('received stop signal, cancelling tasks...')
        for task in asyncio.Task.all_tasks():
            task.cancel()
        logging.info('exiting hgdl')
        return self.optima_list
    ###########################################################################
    async def run_hgdl_epoch(self):
        """
        an epoch is one local run and one global run,
        where one local run are several convergence runs of all workers from
        the x_init point
        """
        #print("    Epoch initiated...")
        #print("    Running global step")
        self.x0 = glob.genetic_step(np.array(self.optima_list["x"]),
        np.array(self.optima_list["func evals"]), bounds = self.bounds, numChoose= len(self.optima_list["func evals"]))
        #print("    Running local step")
        self.run_local(self.x0, np.array(self.optima_list["x"]))
    ###########################################################################
    def run_local(self,x_init, x_defl):
        break_condition = False
        x_init = np.array(x_init)
        x_defl = np.array(x_defl)
        counter = 0
        while break_condition is False:
            counter += 1
            ##walk walkers with DNewton
            x,f,grad_norm,eig,success = self.run_dNewton(x_init,x_defl)
            ##assemble optima_list
            self.optima_list = self.fill_in_optima_list(self.optima_list, x,f,grad_norm,eig,success)
            x_defl = np.array(self.optima_list["x"])
            #print(self.optima_list)
            #input()
            if len(np.where(success == False)[0]) > len(success)/2.0: break_condition = True
            if counter == self.local_max_iter: break_condition = True
    ###########################################################################
    def run_dNewton(self,x_init,x_defl = []):
        """
        this function runs a deflated Newton for
        all the walkers.
        The loop below goes over every walker
        input:
            2d numpy array of initial positions
            2d numpy array of positions of deflations (optional, default = [])
        return:
            optima_locations, func values, gradient norms, eigenvalues, success(bool)
        """
        number_of_walkers = len(x_init)
        x = np.empty((number_of_walkers, self.dim))
        f = np.empty((number_of_walkers))
        grad_norm = np.empty((number_of_walkers))
        eig = np.empty((number_of_walkers,self.dim))
        success = np.empty((number_of_walkers))
        tasks = [None] * number_of_walkers
        #submit a bunch of tasks:
        #print("deflated points: ")
        #print(x_defl)
        for i in range(number_of_walkers):
            tasks[i]= self.client.submit(local.DNewton,self.obj_func, self.grad_func,self.hess_func,\
            x_init[i],x_defl,self.bounds,self.local_tol,self.local_max_iter)
        self.client.gather(tasks)
        #gather results and kick out optima that are too close:
        for i in range(number_of_walkers):
            x[i],f[i],grad_norm[i],eig[i],success[i] = \
            local.DNewton(self.obj_func, self.grad_func,self.hess_func,\
            x_init[i],x_defl,self.bounds,self.local_tol,self.local_max_iter)
            #x[i],f[i],grad_norm[i],eig[i],success[i] = tasks[i].result()
            for j in range(i):
                #exchange for function def too_close():
                if np.linalg.norm(np.subtract(x[i],x[j])) < 2.0 * self.r: success[i] = False; break
            for j in range(len(x_defl)):
                if np.linalg.norm(np.subtract(x[i],x_defl[j])) < 1e-5:
                    print("CAUTION: Newton converged to deflated position")
                    print(x[i],x_defl[j])
                    input()
        return x, f, grad_norm, eig, success
    ###########################################################################
    def fill_in_optima_list(self,optima_list,x,f,grad_norm,eig, success):
        clean_indices = np.where(success == True)
        clean_x = x[clean_indices]
        clean_f = f[clean_indices]
        clean_grad_norm = grad_norm[clean_indices]
        clean_eig = eig[clean_indices]
        classifier = []
        for i in range(len(x)):
            if grad_norm[i] > self.local_tol: classifier.append("degenerate")
            elif len(np.where(eig[i] > 0.0)[0]) == len(eig[i]): classifier.append("minimum")
            elif len(np.where(eig[i] < 0.0)[0]) == len(eig[i]): classifier.append("maximum")
            elif len(np.where(eig[i] == 0.0)[0])  > 0: classifier.append("zero curvature")
            elif len(np.where(eig[i] < 0.0)[0])  < len(eig[i]): classifier.append("sattle point")
            else: print("something is up with the eigen values: ", eig[i]); exit()

        optima_list = {"x":       np.vstack([optima_list["x"],clean_x]), \
                       "func evals":   np.append(optima_list["func evals"],clean_f), \
                       "classifier":   optima_list["classifier"] + classifier, \
                       "eigen values": np.vstack([optima_list["eigen values"],clean_eig]),\
                       "gradient norm":np.append(optima_list["gradient norm"],clean_grad_norm)}
        sort_indices = np.argsort(optima_list["func evals"])
        optima_list["x"] = optima_list["x"][sort_indices]
        optima_list["func evals"] = optima_list["func evals"][sort_indices]
        optima_list["classifier"] = [optima_list["classifier"][i] for i in sort_indices]
        optima_list["eigen values"] = optima_list["eigen values"][sort_indices]
        optima_list["gradient norm"] = optima_list["gradient norm"][sort_indices]
        return optima_list
    ###########################################################################

