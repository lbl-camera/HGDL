import numpy as np
import numba as nb

@nb.njit(cache=True)
def reduced_bump_derivative(x, minima, r, alpha):
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
            if bump==1.:
                return np.inf*np.ones_like(x), 1.
            der = -bump*2*alpha*dist_vec*np.power(exp_denom,-2)
            deflation = 1/(1.-bump)
            return 2*deflation*der, bump
    return np.zeros_like(x), 0.

def newton(x, minima, gradient, hessian, bounds, r, alpha):
    k = x.shape[0]
    for i in range(20):
        jac = gradient(x)
        hess = hessian(x)
        # if x is near 2 minima 
        try:
            f, b = reduced_bump_derivative(x, minima, r, alpha)
        except NotImplementedError:
            print("exited bc was near two points")
            return {"success":False}
        if np.isclose(np.linalg.norm(jac),0.) and np.isclose(b,0.):
            return {"success":True,"x":x,"edge":False}
        # newton step 
        try:
            update = np.linalg.lstsq(hess+np.outer(jac,f), jac, rcond=None)[0]
        # if you are right on top of a minima, there will be annoying infinities 
        # otherwise, just try to move over a little and keep trucking 
        except np.linalg.LinAlgError:
            x += np.random.normal(loc=0.,scale=3*r,size=x.shape)
            update = np.zeros_like(x)
        xNew = x - update
        # if you stepped out of bounds 
        if not in_bounds(xNew, bounds):
            for j in range(1,4):
                if i+j-1 >= 20:
                    return {"success":False}
                xNew = x - update/(2.**j)
                if in_bounds(xNew, bounds):
                    break
            else:
                return {"success":False}
        x = xNew
    return {"success":False}

def in_bounds(x, bounds):
    if (bounds[:,1]-x > 0).all() and (bounds[:,0] - x < 0).all():
        return True
    return False

