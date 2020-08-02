import numpy as np
def gaussian_step(hgdl, x, y):
    x = x.T
    c = 1.0
    y  = y -  np.min(y)
    y =  y/np.max(y)
    cov = np.cov(x, aweights = 1.0 - (y**c))
    mean = np.mean(x , 1)
    offspring = hgdl.rng.multivariate_normal(mean, cov, size=hgdl.num_individuals)
    return offspring

