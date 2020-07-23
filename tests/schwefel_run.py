import numpy as np
from hgdl.hgdl import HGDL
from hgdl.test_functions import *
import time

a = HGDL(schwefel, schwefel_gradient, schwefel_hessian, bounds = [[-500,500],[-500,500]])

print("out")

time.sleep(15)

