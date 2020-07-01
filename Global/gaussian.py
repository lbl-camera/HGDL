import numpy as np
def gaussian_step(hgdl):
    x = hgdl.results.global_x.copy().T
    y = hgdl.results.global_y.copy()
    c = 1.0
    y  = y -  np.min(y)
    y =  y/np.max(y)
    cov = np.cov(x, aweights = 1.0 - (y**c))
    mean = np.mean(x , 1)
    offspring = np.random.multivariate_normal(mean, cov, size=hgdl.num_individuals)
    return offspring

