import numpy as np

def newton(x, func, jac, hess, in_bounds):
    for i in range(15):
        try:
            j, h = jac(x), hess(x)
        except ZeroDivisionError:
            return {"success":False}
        except NotImplementedError:
            return {"success":False}
        except:
            raise
        if np.isclose(np.linalg.norm(j),0.):
            return {"success":True,"x":x,"edge":False}
        # newton step 
        # if you are right on top of a minima, there will be annoying infinities 
        # otherwise, just try to move over a little and keep trucking 
        try:
            update = np.linalg.lstsq(h, j, rcond=None)[0]
        except np.linalg.LinAlgError:
            x += np.random.normal(loc=0.,scale=3*r,size=x.shape)
            update = np.zeros_like(x)
        xNew = x - update
        # if you stepped out of bounds 
        if not in_bounds(xNew):
            for j in range(1,4):
                if i+j-1 >= 20:
                    return {"success":False}
                xNew = x - update/(2.**j)
                if in_bounds(xNew):
                    break
            else:
                return {"success":False}
        x = xNew
    return {"success":False}

