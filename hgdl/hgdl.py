# coding: utf-8

#  imports
import numpy as np
from .global_methods.run_global import run_global
from .local_methods.run_local import run_local
from .info import info
import dask.distributed

class HGDL(object):
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
            * numWorkers (logical cpu cores -1) - how many processes to use
            * fix_rng (True) - sets random numbers to be fixed (for reproducibility)
            * bestX (5) - maximum number of minima and global results to put in get_final()
            * num_individuals (25) - the number of individuals to run for both global and local methods
            * x0 (None) starting points to probe
            * global_method ('gaussian') - these control what global and local methods
            * local_method ('scipy') -    are used by HGDL

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
    def __init__(self, *args, **kwargs):
        # find if the user provided a client
        for z in [*args, kwargs.values]:
            if type(z) == dask.distributed.Client:
                self.client = z
                break
        else:
            self.client = dask.distributed.Client(
                    dask.distributed.LocalCluster(processes=True, dashboard_address=None))
        data = info(*args, **kwargs)
        self.epoch_futures = [self.client.submit(run_epoch, data)]
        for i in range(data.num_epochs):
            self.epoch_futures.append(self.client.submit(run_epoch, self.epoch_futures[-1]))

    # user access functions
    def get_final(self):
        # wait until everything is done 
        result = self.epoch_futures[-1].result().results.roll_up()
        #self.client.cancel(self.epoch_futures)
        #self.client.close()
        return result

    def get_best(self):
        for future in self.epoch_futures[::-1]:
            if future.done():
                finished = future.result()
                break
        else:
            finished = self.epoch_futures[0].result()
        return finished.results.epoch_end()

    def get_latest(self, N):
        for future in self.epoch_futures[::-1]:
            if future.done():
                finished = future.result()
                break
        else:
            finished = self.epoch_futures[0].result()
        return finished.results.latest(N)

    def kill(self):
        for future in self.epoch_futures: future.cancel()

# run a single epoch
def run_epoch(data):
    if data.verbose: print('working on an epoch')
    data.update_global(run_global(data))
    data.update_minima(run_local(data))
    return data

