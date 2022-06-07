import numpy as np
from hgdl.hgdl import HGDL as hgdl
from hgdl.support_functions import *
import time
import dask.distributed as distributed
import tracemalloc



def test_schwefel():
    arr  = 5
    brr  = 6
    bounds = np.array([[-500,500],[-500,500]])
    a = hgdl(schwefel, schwefel_gradient, bounds,
            hess = schwefel_hessian,
            global_optimizer = "genetic",
            local_optimizer = "BFGS",
            number_of_optima = 30000,
            args = (arr,brr), radius = None, num_epochs = 100)

    x0 = np.random.uniform(low = bounds[:, 0], high = bounds[:,1],size = (20,2))
    print("starting positions: ")
    print(x0)
    print("--------------------")
    a.optimize(x0 = x0)


    #print(a.optima_list)
    print("main thread submitted HGDL and will now sleep for 2 seconds")
    time.sleep(2)
    print("main thread asks for 10 best solutions:")
    print(a.get_latest())
    print("main sleeps for another 2 seconds")
    time.sleep(2)
    print("main thread kills optimization")
    res = a.kill_client()
    print("hgdl was killed but I am waiting 2s")
    print("")
    print("")
    print("")
    print("")
    print(res)

if __name__ == '__main__':
    test_schwefel()
