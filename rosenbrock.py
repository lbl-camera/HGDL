from scipy.optimize import rosen, rosen_der, rosen_hess
import numpy as np
from hgdl import HGDL
from time import sleep
from sys import exit
b = np.array([[-2, 2],[-3,3.]])

hgdl = HGDL(rosen, rosen_der, rosen_hess, b)

#for i in range(2):
#    print(hgdl.get_best())

res = hgdl.get_final()

print(res)
