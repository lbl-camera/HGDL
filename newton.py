import numpy as np
import numba as nb

@nb.njit(cache=True)
def reduced_bump_derivative(x, minima, r, alpha):
    r2 = r**2.
    factors = np.ones_like(x)
    b = 1.
    modified = False
    for i in range(len(minima)):
        dist_vec = x-minima[i]
        dist2 = np.sum(np.power(dist_vec,2))
        if dist2 > r2:
            continue
        else:
            modified = True
            exp_denom = r2-dist2
            b = np.exp(-alpha/exp_denom + alpha/r2)
            der = -2*alpha*np.power(exp_denom,-2)
            for j in range(len(factors)):
                factors[j] *= b*der*dist_vec[j]/(1.-b)
    if not modified:
        return np.zeros_like(factors), 0.
    return factors, b

def newton(x, minima, gradient, hessian, bounds, r, alpha):
    for i in range(30):
        jac = gradient(x)
        hess = hessian(x)
        f, b = reduced_bump_derivative(x, minima, r, alpha)
        update = np.linalg.lstsq(hess+np.outer(jac,f), jac, rcond=None)[0]
        xNew = x - update
        if not in_bounds(xNew, bounds):
            for i in range(1,4):
                xNew = x - update/(2.**i)
                if in_bounds(xNew, bounds):
                    return {"success":True,"x":xNew,"edge":True}
            return {"success":False}
        x = xNew
        if np.linalg.norm(jac)*b < 1e-5:
            return {"success":True,"x":x,"edge":False}
    return {"success":False}

def in_bounds(x, bounds):
    if (bounds[:,1]-x > 0).all() and (bounds[:,0] - x < 0).all():
        return True
    return False

