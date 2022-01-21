import numpy as np
from hgdl.hgdl import HGDL
from support_functions import *
import time
import dask.distributed as distributed



def test_non_diff():
    #dask_client = distributed.Client("10.0.0.184:8786")
    a = HGDL(non_diff, non_diff_grad, 
             hess = non_diff_hess,bounds = [[-5,5],[-5,5]],
            radius = 0.1, num_epochs = 5)
    a.optimize(dask_client = None)
    res = a.optima.list

    #print(a.optima_list)
    print("main thread submitted HGDL and will now sleep for 10 seconds")
    time.sleep(10)
    print("main thread asks for 10 best solutions:")
    print(a.get_latest(10))
    print("main sleeps for another 10 seconds")
    time.sleep(10)
    print("main thread kills optimization")
    res = a.kill_client()
    print("hgdl was killed but I am waiting 2s")
    time.sleep(2)
    print("")
    print("")
    print("")
    print("")
    print(res)

if __name__ == '__main__':
    test_non_diff()
