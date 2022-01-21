import numpy as np
import hgdl.misc as misc
import hgdl.local_methods.bump_function as defl
import dask.distributed as distributed


def DNewton(func,grad,hess,bounds,x0,max_iter,tol,*args):
    e = np.inf
    gradient = np.ones((len(x0))) * np.inf
    counter = 0
    x = np.array(x0)
    success = True
    grad_list = []
    while e > tol and np.linalg.norm(gradient) > tol:
        gradient = grad(x,*args)
        hessian  = hess(x,*args)
        grad_list.append(np.max(gradient))
        try:
            gamma = np.linalg.solve(hessian,-gradient)
        except Exception as error:
            gamma,a,b,c = np.linalg.lstsq(hessian,-gradient)
        x += gamma
        e = np.max(abs(gamma))
        #print("gamma = ", gamma," gradient = ",gradient,flush = True)
        if misc.out_of_bounds(x,bounds):
            x = np.random.uniform(low = bounds[:,0], high = bounds[:,1], size = len(bounds))
            #print("out of bounds, replace")
        if counter > max_iter:
            #print("dNewton takes a long time to converge, possibly due to not finding any non-deflated optima...", flush = True)
            return x,func(x, *args),e,np.linalg.eig(hess(x, *args))[0], False
        counter += 1
    return x,func(x, *args),e,np.linalg.eig(hess(x, *args))[0], success

