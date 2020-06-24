from scipy.optimize import rosen, rosen_der, rosen_hess
import numpy as np
from hgdl import HGDL
from time import sleep

b = np.array([[-2, 2],[-3,3.]])

hgdl = HGDL(rosen, rosen_der, rosen_hess, b, max_epochs=5)

for i in range(8):
    sleep(11)
    print(hgdl.get_best())

print(hgdl.get_final())
