from scipy.optimize import rosen, rosen_der, rosen_hess
import numpy as np
from hgdl import HGDL
b = np.array([[-2, 2],[-3,3.]])
print(HGDL(rosen, rosen_der, rosen_hess, b, max_epochs=100).run())

