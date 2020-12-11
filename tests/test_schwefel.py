import numpy as np
from hgdl.hgdl import HGDL as hgdl
from test_functions import *
import time
import dask.distributed as distributed

def main():
    arr  = 5
    brr  = 6
    #dask_client = distributed.Client("10.0.0.184:8786")
    a = hgdl(schwefel, schwefel_gradient, schwefel_hessian,[[-500,500],[-500,500]],
            global_optimizer = "genetic",
            args = (arr,brr), radius = 5.0, num_epochs = 1000, verbose = False)
    #a.optimize(dask_client = distributed.Client())
    x0 = np.random.uniform(size = (20,2))
    a.optimize(x0 = x0)
    #a.optimize(dask_client = False)
    #res = a.optima_list
    #print(res)


    #print(a.optima_list)
    print("main thread submitted HGDL and will now sleep for 10 seconds")
    time.sleep(10)
    print("main thread asks for 10 best solutions:")
    print(a.get_latest(10))
    #a.cancel_tasks()
    print("main sleeps for another 10 seconds")
    time.sleep(10)
    print("main thread kills optimization")
    res = a.kill()
    print("hgdl was killed but I am waiting 2s")
    time.sleep(2)
    print("")
    print("")
    print("")
    print("")
    print(res)

if __name__ == '__main__':
    main()
