import numpy as np
import hgdl.misc as misc
import hgdl.local_methods.bump_function as defl
import dask.distributed as distributed


def DNewton(func,grad,hess,bounds,x0,max_iter,*args):
    e = np.inf
    tol = 1e-6
    counter = 0
    x = np.array(x0)
    success = True
    grad_list = []
    while e > tol:
        gradient = grad(x,*args)
        hessian  = hess(x,*args)
        grad_list.append(np.max(gradient))
        try:
            gamma = np.linalg.solve(hessian,-gradient)
        except Exception as error:
            gamma,a,b,c = np.linalg.lstsq(hessian,-gradient)
        x += gamma
        #print(np.linalg.norm(gradient), np.max(abs(gamma)), flush = True)
        e = np.max(abs(gamma))
        if misc.out_of_bounds(x,bounds):
            x = np.random.uniform(low = bounds[:,0], high = bounds[:,1], size = len(bounds))
        if counter > max_iter:
            print("dNewton takes a long time to converge, possibly due to not finding any non-deflated optima...")
            return x,func(x, *args),e,np.linalg.eig(hess(x, *args))[0], False
        counter += 1
    return x,func(x, *args),e,np.linalg.eig(hess(x, *args))[0], success



def gradient_descent(ObjectiveFunction, GradientFunction,bounds,x0, radius, args):
    dim = len(bounds)
    bounds = np.array(bounds)
    epsilon = np.inf
    step_counter = 0
    beta = 0.5
    x = np.array(x0)
    success = True
    while epsilon > 1e-6:
        step_counter += 1
        aug_gradient = defl.deflated_grad(GradientFunction,x,x_defl,radius, *args)
        counter = 0
        step = 1.0
        while misc.out_of_bounds(x - (step * aug_gradient), bounds) or \
              ObjectiveFunction(x - (step * aug_gradient), *args) > \
              ObjectiveFunction(x, *args) - ((step / 2.0) * np.linalg.norm(aug_gradient) ** 2):
            step = step * beta
            counter += 1
            if counter > 10:
                return x, ObjectiveFunction(x, *args), False
        x = x - (step * aug_gradient)
        epsilon = np.linalg.norm(aug_gradient)
        if step_counter > 20:
            success = False
            break
    return x, ObjectiveFunction(x, *args), True

