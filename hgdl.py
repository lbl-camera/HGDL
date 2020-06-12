import numpy as np
import torch as t
import time

###authors: David Perryman, Marcus Noack
###institution: CAMERA @ Lawrence Berkeley National Laboratory


class HGDL:
    """
    doc string here
    """
    def __init__(self,obj_func = None,grad_func = None, hess_func = None,bounds = None, maxEpochs=5, 
            local_tol = 1e-4, global_tol = 1e-4,
            local_max_iter = 20, global_max_iter = 120,
            number_of_walkers = 20, number_of_optima = 10,
            initial_positions = None, argument_dict = None):
        """
        intialization for the HGDL class

        input:
        ------
        Nothing has to be input. If "HGDL()" then Schefel's
        function is used as example. If this is not a test and 
        a user defined function has to be called the parameters are:
            obj_func: function that calculates the scalar function value at a ppoint x
            grad_func: function that calculates the gradient at a point x
            hess_func: function that calculates the hessian at a point x
            bounds(numpy array): array of bounds, len is the dimensionality of the problem
        """
        if obj_func is None: self.obj_func = self.schwefel
        else: self.obj_func = obj_func
        if grad_func is None: self.grad_func = self.schwefel_gradient
        else: self.grad_func = grad_func
        if hess_func is None: self.hess_func = self.schwefel_hessian
        else: self.hess_func = hess_func
        if bounds is None: self.bounds = np.array([[-500,500],[-500,500]])
        else: self.bounds = bounds
        self.dim = len(self.bounds)
        self.local_tol = local_tol
        self.global_tol = global_tol
        self.local_max_iter = local_max_iter
        self.global_max_iter = global_max_iter
        self.number_of_walkers = number_of_walkers
        self.number_of_optima = number_of_optima
        if initial_positions is None:
            self.initial_positions = \
            np.random.uniform(low = self.bounds[:,0], high = self.bounds[:,1], 
            size = (self.number_of_walkers,len(self.bounds)))
        else: self.initial_positions = initial_positions
        if len(self.initial_positions) != self.number_of_walkers:
            exit("number of initial position does not match the number of walkers")
        self.argument_dict = argument_dict
        ##################################
        res = self.hgdl()
        print(res)
    ###########################################################################
    def bump_function(self,x,x0,r = 1.0):
        """
        evaluates the bump function
        x ... 2d numpy array of points
        x0 ... 1d numpy array of location of bump function
        """
        np.seterr(all = 'ignore')
        d = np.linalg.norm((x - x0),axis = 1)
        b = np.zeros((d.shape))
        indices = np.where(d<r)
        a = 1.0-((d[indices]/r)**2)
        b[indices] = np.exp(-1.0/a)
        return b
    ###########################################################################
    def bump_function_gradient(self,x,x0, r = 1.0):
        np.seterr(all = 'ignore')
        d = np.array([np.linalg.norm((x - x0),axis = 1),]*len(x[0])).T
        d2= (x - x0)
        gr = np.zeros((d2.shape))
        indices = np.where(d<r) 
        a = 1.0-((d[indices]/r)**2)
        gr[indices] = ((-2.0*d2[indices])/(a**2)) * np.exp(-1.0/a)
        return gr
    ###########################################################################
    def deflation_operator(self,x,x_0):
        return 1.0/(1.0-self.bump_function(x,x0))
    ###########################################################################
    def deflation_operator_gradient(self,x,x0):
        return self.bump_function_gradient(x,x0)
    ###########################################################################
    def deflation_function(self,x,x0):
        if len(x0) == 0: return 1.0
        s = np.array([self.deflation_operator(x,x0[i]) for i in range(len(x0))])
        return np.sum(s)
    ###########################################################################
    def deflation_function_gradient(self,x,x0):
        if len(x0) == 0: return np.zeros((len(x)))
        s = np.array([self.deflation_operator_gradient(x,x0[i]) for i in range(len(x0))])
        return np.sum(s)
    ###########################################################################
    def DNewton(self,x,x0,tol):
        e = np.inf
        while e > tol:
            gradient = self.grad_func(x, self.argument_dict)
            e = np.linalg.norm(gradient)
            hessian = self.hess_func(x, self.argument_dict)
            d = self.deflation_function(x,x0)
            dg = self.deflation_function_gradient(x,x0)
            gamma = np.linalg.solve(hessian+(np.outer(gradient,dg)/d),-gradient)
            x += gamma
            print("current position: ",x,"epsilon: ",e)
        return x,self.obj_func(x, self.argument_dict),e,np.linalg.eig(hessian)[0]
    ###########################################################################
    def hgdl(self):
        break_condition = False
        x = self.initial_positions
        f = np.empty((len(x)))
        e = np.empty((len(x)))
        eig = np.empty((len(x), self.dim))
        x0 = []
        optima_list = None
        while break_condition is False:
            ##walk walkers with DNewton
            for i in range(self.number_of_walkers):
                x[i],f[i],e[i], eig[i] = self.DNewton(x[i],x0,self.local_tol)
            ##assemble optima_list
            optima_list = self.fill_in_optima_list(optima_list, x,f,e,eig)
            print(optima_list)
            ##if something break_condition is True
            ##replace walkers by global step
            #x = self.global_step(genetic_step,x,Y)
        return x
    ###########################################################################
    def fill_in_optima_list(self,optima_list,x,f,grad_norm,eig):
        if optima_list is None:
            classifier = []
            for i in range(len(x)):
                if grad_norm[i] > self.local_tol: classifier.append("degenerate")
                else:
                    if len(np.where(eig[i] > 0.0)[0]) == len(eig[i]): classifier.append("minimum")
                    elif len(np.where(eig[i] < 0.0)[0]) == len(eig[i]): classifier.append("maximum")
                    elif len(np.where(eig[i] == 0.0)[0])  > 0: classifier.append("zero curvature")
                    elif len(np.where(eig[i] < 0.0)[0])  < len(eig[i]): classifier.append("sattle point")
                    else: print("something is up with the eigen values: ", eig[i]); exit()
            optima_list = {"points": x, "func evals": f, "classifier": classifier, "eigen values": eig, "gradient norm":grad_norm}
            #print("----------------------------------------------------")
            #print(optima_list)
            sort_indices = np.argsort(optima_list["func evals"])
            #print("----------------------------------------------------")
            #print(sort_indices)
            #print("----------------------------------------------------")
            optima_list["points"] = optima_list["points"][sort_indices]
            optima_list["func evals"] = optima_list["func evals"][sort_indices]
            optima_list["classifier"] = [optima_list["classifier"][i] for i in sort_indices]
            optima_list["eigen values"] = optima_list["eigen values"][sort_indices]
            optima_list["gradient norm"] = optima_list["gradient norm"][sort_indices]
            #print("----------------------------------------------------")
            #print(optima_list)
            #print("----------------------------------------------------")
            #exit()
        else:
            for i in range(len(x)):
                ...
        print(optima_list)
        exit()
        return optima_list
    ###########################################################################
    ##################TEST FUNCTION############################################
    ###########################################################################
    ###########################################################################
    def schwefel(self,x,args):
        return 418.9829*len(x) - np.sum(x*np.sin(np.sqrt(np.abs(x))))
    ###########################################################################
    def schwefel_gradient(self,x,args):
        indices = np.where(x==0)
        x[indices] = 0.00001
        return -(np.sin(np.sqrt(np.abs(x)))+(x*np.cos(np.sqrt(np.abs(x)))*(0.5/np.sqrt(np.abs(x)))*(np.sign(x))))
    ###########################################################################
    def schwefel_hessian(self,x,args):
        e = 1e-4
        hessian=np.zeros((len(x),len(x)))
        for i in range(len(hessian)):
            x_aux1 = np.array(x)
            x_aux1[i] =x[i] + e
            x_aux2 = np.array(x)
            x_aux2[i] =x[i] - e
            a = (((self.schwefel_gradient(x_aux1,args)-self.schwefel_gradient(x_aux2,args))/(2.0*e)))
            hessian[i,i] = a[i]
        return hessian
