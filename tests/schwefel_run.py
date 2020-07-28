import numpy as np
from hgdl.hgdl import HGDL
from hgdl.test_functions import *
import time

def main():
    a = HGDL(schwefel, schwefel_gradient, schwefel_hessian, bounds = [[-500,500],[-500,500]])

    #print(a.optima_list)
    print("submitted, sleeping")
    time.sleep(5)
    print(a.optima_list)

if __name__ == '__main__':
    main()
