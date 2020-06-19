import numpy as np
import torch as t
import time

###authors: David Perryman, Marcus Noack
###institution: CAMERA @ Lawrence Berkeley National Laboratory



####TODO:   *currently walkers that walk out in Newton are dicarded. We should do a line search then
            *currently individuals are replaced randomly, we should do the genetic step or the global_gaussian_pdf step
            *the radius is still ad hoc, should be related to curvature


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
    def bump_function(self,x,x0,r = 40.0):
        """
        evaluates the bump function
        x ... 2d numpy array of points
        x0 ... 1d numpy array of location of bump function
        """
        if x.ndim == 1: x = np.array([x])
        np.seterr(all = 'ignore')
        d = np.linalg.norm((x - x0),axis = 1)
        b = np.zeros((d.shape))
        indices = np.where(d<r)
        a = 1.0-((d[indices]/r)**2)
        b[indices] = np.exp(1.0) * np.exp(-1.0/a)
        return b
    ###########################################################################
    def bump_function_gradient(self,x,x0, r = 40.0):
        if x.ndim == 1: x = np.array([x])
        np.seterr(all = 'ignore')
        d = np.array([np.linalg.norm((x - x0),axis = 1),]*len(x[0])).T
        d2= (x - x0)
        gr = np.zeros((d2.shape))
        indices = np.where(d<r) 
        a = 1.0-((d[indices]/r)**2)
        gr[indices] = np.exp(1.0) * ((-2.0*d2[indices])/(a**2)) * np.exp(-1.0/a)
        return gr
    ###########################################################################
    #def deflation_operator(self,x,x0):
    #    return 1.0/(1.0-self.bump_function(np.array([x]),x0))
    ###########################################################################
    #def deflation_operator_gradient(self,x,x0):
    #    return self.bump_function_gradient(np.array([x]),x0)
    ###########################################################################
    def deflation_function(self,x,x0):
        if len(x0) == 0: return 1.0
        s = np.array([self.bump_function(x,x0[i]) for i in range(len(x0))])
        return (1.0/(1.0-sum(s)))
    ###########################################################################
    def deflation_function_gradient(self,x,x0):
        if len(x0) == 0: return np.zeros((len(x)))
        s1 = np.array([self.bump_function(x,x0[i]) for i in range(len(x0))])
        s2 = np.array([self.bump_function_gradient(x,x0[i]) for i in range(len(x0))])
        return (1.0/((1.0-sum(s1))**2))*np.sum(s2)
    ###########################################################################
    def DNewton(self,x,x0,tol):
        e = np.inf
        print("in newton: ",x,x0)
        success = True
        counter = 0
        while e > tol:
            counter += 1
            if counter >= self.local_max_iter or self.out_of_bounds(x):
                success = False
                print("out of bounds or iter limit reached: ", x, "after ", counter," iterations")
                x = np.zeros((x.shape))
                #input()
                return x,0,0,0,success
            gradient = self.grad_func(x, self.argument_dict)
            e = np.linalg.norm(gradient)
            hessian = self.hess_func(x, self.argument_dict)
            d = self.deflation_function(x,x0)
            dg = self.deflation_function_gradient(x,x0)
            gamma = np.linalg.solve(hessian + (np.outer(gradient,dg)/d),-gradient)
            x += gamma
            #if len(x0) != 0: self.plot_schwefel( deflation_points = np.array(x0), points = np.array([x]))
            print("current position: ",x,"epsilon: ",e)
        return x,self.obj_func(x, self.argument_dict),e,np.linalg.eig(hessian)[0], success
    ###########################################################################
    def hgdl(self):
        break_condition = False
        x = self.initial_positions
        print("initial points: ", x)
        temp = np.array(x)
        f = np.empty((len(x)))
        e = np.empty((len(x)))
        eig = np.empty((len(x), self.dim))
        success = np.empty((len(x)), dtype = bool)
        x0 = []
        optima_list = {"points": np.empty((0,2)), \
                "func evals": np.empty((0)), \
                "classifier": [], "eigen values": np.empty((0,self.dim)), \
                "gradient norm":np.empty((0))}
        while break_condition is False:
            ##walk walkers with DNewton
            for i in range(self.number_of_walkers):
                x[i],f[i],e[i], eig[i],success[i] = self.DNewton(x[i],x0,self.local_tol)
            #print("results of the newton: ",x,f)
            #print(success)
            #print("deflations @: ", x0)
            ##assemble optima_list
            optima_list = self.fill_in_optima_list(optima_list, x,f,e,eig,success)
            x0 = optima_list["points"]
            ###this is in place of the global replacement later:
            x = \
            np.random.uniform(low = self.bounds[:,0], high = self.bounds[:,1], 
            size = (self.number_of_walkers,len(self.bounds)))
            ########################################################
            #print(optima_list)
            self.plot_schwefel(points = optima_list["points"], deflation_points = optima_list["points"])
            print("=================================================")
            print("=================================================")
            print("=================================================")
            print("=================================================")
            #input()
            ##if something break_condition is True
            ##replace walkers by global step
            #x = self.global_step(genetic_step,x,Y)
        return x
    ###########################################################################
    def fill_in_optima_list(self,optima_list,x,f,grad_norm,eig, success):
        clean_indices = np.where(success == True)
        clean_x = x[clean_indices]
        clean_f = f[clean_indices]
        clean_grad_norm = grad_norm[clean_indices]
        clean_eig = eig[clean_indices]
        classifier = []
        for i in range(len(x)):
            if grad_norm[i] > self.local_tol: classifier.append("degenerate")
            elif len(np.where(eig[i] > 0.0)[0]) == len(eig[i]): classifier.append("minimum")
            elif len(np.where(eig[i] < 0.0)[0]) == len(eig[i]): classifier.append("maximum")
            elif len(np.where(eig[i] == 0.0)[0])  > 0: classifier.append("zero curvature")
            elif len(np.where(eig[i] < 0.0)[0])  < len(eig[i]): classifier.append("sattle point")
            else: print("something is up with the eigen values: ", eig[i]); exit()

        optima_list = {"points":       np.vstack([optima_list["points"],clean_x]), \
                       "func evals":   np.append(optima_list["func evals"],clean_f), \
                       "classifier":   optima_list["classifier"] + classifier, \
                       "eigen values": np.vstack([optima_list["eigen values"],clean_eig]),\
                       "gradient norm":np.append(optima_list["gradient norm"],clean_grad_norm)}
        sort_indices = np.argsort(optima_list["func evals"])
        optima_list["points"] = optima_list["points"][sort_indices]
        optima_list["func evals"] = optima_list["func evals"][sort_indices]
        optima_list["classifier"] = [optima_list["classifier"][i] for i in sort_indices]
        optima_list["eigen values"] = optima_list["eigen values"][sort_indices]
        optima_list["gradient norm"] = optima_list["gradient norm"][sort_indices]
        return optima_list
    ###########################################################################
    def out_of_bounds(self,x):
        for i in range(len(x)):
            if x[i] < self.bounds[i,0] or x[i] > self.bounds[i,1]:
                return True
        return False
    ###########################################################################
    def global_step(self,x,y):
        c = 1.0
        y  = y -  np.min(y)
        y =  y/np.max(y)
        cov = np.cov(x, aweights = 1.0 - (y**c))
        mean= np.mean(x , axis = 0)
        offspring = np.random.multivariate_normal(mean, cov, size = len(x))
        return offspring

    ###########################################################################
    ##################TEST FUNCTION############################################
    ###########################################################################
    ###########################################################################
    def schwefel(self,x,args = None):
        return 418.9829*len(x) - np.sum(x*np.sin(np.sqrt(np.abs(x))))
    ###########################################################################
    def schwefel_gradient(self,x,args = None):
        indices = np.where(x==0)
        x[indices] = 0.00001
        return -(np.sin(np.sqrt(np.abs(x)))+(x*np.cos(np.sqrt(np.abs(x)))*(0.5/np.sqrt(np.abs(x)))*(np.sign(x))))
    ###########################################################################
    def schwefel_hessian(self,x,args = None):
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
    def plot_schwefel(self,bounds = [[-500,500],[-500,500]], resolution = 100, points = None, deflation_points = None):
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib import cm

        X = np.linspace(bounds[0][0], bounds[0][1], resolution)
        Y = np.linspace(bounds[1][0], bounds[1][1], resolution)
        X, Y = np.meshgrid(X, Y)
        schwefel = np.empty((X.shape))
        gr = np.empty((X.shape))

        for i in range(len(X)):
            for j in range(len(Y)):
                schwefel[i,j] = self.schwefel(np.array([X[i,j],Y[i,j]]))
                if deflation_points is not None:
                    gr[i,j] = self.schwefel_gradient(np.array([X[i,j],Y[i,j]]))[0] * self.deflation_function(np.array([[X[i,j],Y[i,j]]]), deflation_points)
                    #gr[i,j] = self.deflation_function(np.array([[X[i,j],Y[i,j]]]), deflation_points)


        fig = plt.figure(0)
        a = plt.pcolormesh(X, Y, schwefel, cmap=cm.viridis)
        plt.colorbar(a)
        if points is not None: plt.scatter(points[:,0], points[:,1])

        if len(deflation_points) != 0: plt.scatter(deflation_points[:,0], deflation_points[:,1])
        #    fig = plt.figure(1)
        #    b = plt.pcolormesh(X, Y, gr, cmap=cm.viridis, vmin = -1.5, vmax = 1.5)
        #    plt.colorbar(b)
        plt.show()
