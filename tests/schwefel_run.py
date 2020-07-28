import numpy as np
from hgdl.hgdl import HGDL
from hgdl.test_functions import *
import time

def main():
    a = HGDL(schwefel, schwefel_gradient, schwefel_hessian, bounds = [[-500,500],[-500,500]], maxEpochs = 100)

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
