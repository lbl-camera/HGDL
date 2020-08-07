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
    """
    def __init__(self, *args, **kwargs):
        # find if the user provided a client
        for z in [*args, kwargs.values]:
            if type(z) == dask.distributed.Client:
                self.client = z
                break
        else:
            self.client = dask.distributed.Client()
        data = info(*args, **kwargs)
        self.epoch_futures = [self.client.submit(run_epoch, data)]
        for i in range(data.num_epochs):
            self.epoch_futures.append(self.client.submit(run_epoch, self.epoch_futures[-1]))

    # user access functions
    def get_final(self):
        # wait until everything is done 
        return self.epoch_futures[-1].result().results.roll_up()

    def get_best(self):
        for z in self.epoch_futures[::-1]:
            if z.done():
                result = z.result()
                break
        else:
            result = self.epoch_futures[0].result()
        return result.results.epoch_end()

# run a single epoch
def run_epoch(data):
    data.update_global(run_global(data))
    data.update_minima(run_local(data))
    return data

