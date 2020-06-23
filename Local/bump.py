import numpy as np
import numba as nb

@nb.njit(cache=True)
def deflation(x, minima, r, alpha):
    r2 = r**2.
    for i in range(minima.shape[0]):
        dist_vec = x - minima[i]
        dist2 = np.sum(np.power(dist_vec,2))
        if dist2 > r2:
            continue
        else:
            for j in range(i+1,minima.shape[0]):
                if np.sum(np.power(x-minima[j],2))<r2:
                    raise NotImplementedError("there are two minima in range of this point. Exiting")
            exp_denom = r2-dist2
            bump = np.exp(-alpha/exp_denom + alpha/r2)
            deflation = 1/(1.-bump)
            return deflation
    return 0.

@nb.njit(cache=True)
def deflation_der(x, minima, r, alpha):
    r2 = r**2.
    for i in range(minima.shape[0]):
        dist_vec = x - minima[i]
        dist2 = np.sum(np.power(dist_vec,2))
        if dist2 > r2:
            continue
        else:
            for j in range(i+1,minima.shape[0]):
                if np.sum(np.power(x-minima[j],2))<r2:
                    raise NotImplementedError("there are two minima in range of this point. Exiting")
            exp_denom = r2-dist2
            bump = np.exp(-alpha/exp_denom + alpha/r2)
            der = -bump*2*alpha*dist_vec*np.power(exp_denom,-2)
            deflation = 1/(1.-bump)
            return 2*deflation*deflation*der

    return np.zeros_like(x)



