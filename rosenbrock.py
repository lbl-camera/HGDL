from scipy.optimize import rosen, rosen_der, rosen_hess
import numpy as np
from hgdl import HGDL
from time import sleep
from sys import exit
b = np.array([[-2, 2],[-3,3.]])

hgdl = HGDL(rosen, rosen_der, rosen_hess, b, max_epochs=10, num_workers=2)

for i in range(10):
    print(hgdl.get_best())
print(hgdl.get_final())
