import numpy as np
from hgdl.hgdl import HGDL
from hgdl.test_functions import *
import time

def main():
    arr  = 5
    brr  = 6
    a = HGDL(schwefel, schwefel_gradient, schwefel_hessian,[[-500,500],[-500,500]], 
            args = (arr,brr), maxEpochs = 100, dask_client = True)

    #print(a.optima_list)
    print("main thread submitted HGDL and will now sleep for 10 seconds")
    time.sleep(10)
    exit()
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
