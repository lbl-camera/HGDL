import numpy as np
from hgdl.hgdl import HGDL
from schwefel_def  import *
import time

def main():
    a = HGDL(schwefel, schwefel_gradient, hess=schwefel_hessian,
            bounds = np.array([[-500,500],[-500,500]]),
            local_method='my_newton', num_epochs = 100,
            verbose=True
            )

    #print(a.optima_list)
    print("main thread submitted HGDL and will now sleep for 10 seconds")
    time.sleep(10)
    print("main thread asks for 10 best solutions:")
    print(a.get_latest(10))
    print("main sleeps for another 10 seconds")
    time.sleep(10)
    print("main thread kills optimization")
    a.kill()

if __name__ == '__main__':
    main()
