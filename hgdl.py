import numpy as np
import torch as t
import time

###authors: David Perryman, Marcus Noack


class HGDL:
    def __init__(self.obj,grad,hess,tol = 1e-6):
        """
        intialization for the HGDL class
        input:
        ------
            obj_func:
            grad_func:
            hess_func:
        """
    def __call__(self):
        return self.res
    ###########################################################################
    def bump_function(x,x0):
        """
        evaluates the bump function
        x ... 2d numpy array of points
        x0 ... 1d numpy array of location of bump function
        """
        np.seterr(all = 'ignore')
        d = np.linalg.norm((x - x0),axis = 1)
        b = np.zeros((d.shape))
        indices = np.where(d<1.0)
        a = 1.0-(d[indices]**2)
        b[indices] = np.exp(-1.0/a)
        return b
    ###########################################################################
    def bump_function_gradient(x,x0,r):
        np.seterr(all = 'ignore')
        d = np.array([np.linalg.norm((x - x0),axis = 1),]*len(x[0])).T
        d2= (x - x0)
        gr = np.zeros((d2.shape))
        indices = np.where(d<1.0) 
        a = 1.0-(d[indices]**2)
        gr[indices] = ((-2.0*d2[indices])/(a**2)) * np.exp(-1.0/a)
    return gr
    ###########################################################################
    def DNewton(optima_list):
        e = np.inf
        while e > 0.0000001:
            gradient = grad(x0)
            e = np.linalg.norm(gradient)
            hessian  = hess(x0)
            gamma = np.linalg.solve(hessian,-gradient[0])
            x0 += gamma
            print("current position: ",x0,"epsilon: ",e)
    return x0,obj(x0),np.linalg.eig(hessian)[0]
    ###########################################################################
    def hgdl(obj,grad,hess,bounds, maxEpochs=5, 
            local_tol = 1e-4, global_tol = 1e-4,
            local_max_iter = 20, global_max_iter = 120,
            number_of_walkers = 20, number_of_optima = 10, inital_positions = None):
        if initial_conditions is None:
            x = np.random.uniform(low = bounds[:,0], high = bounds[:,1], size = (number_of_walkers,len(bounds)))
        break_condition = False
        while break_condition is False:
            ##walk walkers with DNewton
            for i in len(number_of_walkers):
                ...
            ##assemble optima_list
            ##if something break_condition is True
            ##replace walkers by global step
            ...
        return optima_list

    ###########################################################################
    ##################TEST FUNCTION############################################
    ###########################################################################
    ###########################################################################
    def schwefel(x):
        return 418.9829*len(x[0]) - np.sum(x*np.sin(np.sqrt(np.abs(x))),axis = 1)
    ###########################################################################
    def schwefel_gradient(x):
        indices = np.where(x==0)
        x[indices] = 0.00001
        return -(np.sin(np.sqrt(np.abs(x)))+(x*np.cos(np.sqrt(np.abs(x)))*(0.5/np.sqrt(np.abs(x)))*(np.sign(x))))
    ###########################################################################
    def schwefel_hessian(x):
        e = 1e-4
        dd = (schwefel_gradient(x+e)-schwefel_gradient(x-e))/e
        hessian=np.zeros((len(x[0]),len(x[0])))
        for i in range(len(hessian)):
            hessian[i,i] = (schwefel_gradient(x[:,i]+e)-schwefel_gradient(x[:,i]-e))/(2.0*e)
        return hessian
