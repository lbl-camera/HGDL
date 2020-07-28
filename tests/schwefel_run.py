import numpy as np
from hgdl.hgdl import HGDL
from hgdl.test_functions import *
import time

def main():
    a = HGDL(schwefel, schwefel_gradient, schwefel_hessian, bounds = [[-500,500],[-500,500]], maxEpochs = 100)

    #print(a.optima_list)
    print("submitted, sleeping")
    time.sleep(10)
    print(a.get_latest(10))
    time.sleep(10)
    a.kill()

if __name__ == '__main__':
    main()
