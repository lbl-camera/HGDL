import numpy as np
from scipy.optimize import minimize

def scipy_newton(hgdl):
    
    for i in range(hgdl.max_local):
        for j in range(len(hgdl.num_individuals)):
            res = minimze(fun=hgdl.func, jac=hgdl.jac, hess=hgdl.hess, method='SLSQP')
            if 
        jac = gradient(x)
        hess = hessian(x)
        # if x is near 2 minima 
        try:
            b = 0 #f, b = reduced_bump_derivative(x, minima, r, alpha)
        except NotImplementedError:
            print("exited bc was near two points")
            return {"success":False}
        if np.isclose(np.linalg.norm(jac),0.) and np.isclose(b,0.):
            return {"success":True,"x":x,"edge":False}
        # newton step 
        try:
            update = np.linalg.lstsq(hess, jac, rcond=None)[0]
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

