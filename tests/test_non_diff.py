import numpy as np
from hgdl.hgdl import HGDL
from test_functions import *
import time
import dask.distributed as distributed

def main():
    #dask_client = distributed.Client("10.0.0.184:8786")
    a = HGDL(non_diff, non_diff_grad, 
             non_diff_hess,[[-5,5],[-5,5]],
            radius = 0.1, maxEpochs = 5, verbose = False)
    #a.optimize(dask_client = distributed.Client())
    #a.optimize(dask_client = True)
    a.optimize(dask_client = False)
    res = a.optima.list
    print(res)
    exit()


    #print(a.optima_list)
    print("main thread submitted HGDL and will now sleep for 10 seconds")
    time.sleep(10)
    print("main thread asks for 10 best solutions:")
    print(a.get_latest(10))
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
