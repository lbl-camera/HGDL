import numpy as np
import torch as t
import time
import hgdl.misc as misc
import hgdl.local_methods as local
import hgdl.global_methods as glob
import dask.distributed as distributed
import dask.multiprocessing
from dask.distributed import as_completed
###authors: David Perryman, Marcus Noack
###institution: CAMERA @ Lawrence Berkeley National Laboratory

class optima:
    """
    stores all results and adaptations of it
    """
    def __init__(self,dim):
        """
        input:
        -----
            dim  the dimensionality of the space
        """
        self.dim = dim
        self.list = {"x": np.empty((0,self.dim)), 
                     "func evals": np.empty((0)), 
                     "classifier": [], "eigen values": np.empty((0,self.dim)), 
                     "gradient norm":np.empty((0)),
                     "success": False}
    ####################################################
    def fill_in_optima_list(self,x,f,grad_norm,eig, success):
        #print("before")
        #print(self.list)
        #print(x)
        #print(f)
        #print(grad_norm)
        clean_indices = np.where(success == True)
        clean_x = x[clean_indices]
        clean_f = f[clean_indices]
        clean_grad_norm = grad_norm[clean_indices]
        clean_eig = eig[clean_indices]
        classifier = []
        for i in range(len(clean_x)):
            if clean_grad_norm[i] > 1e-6: classifier.append("degenerate")
            elif len(np.where(clean_eig[i] > 0.0)[0]) == len(clean_eig[i]): classifier.append("minimum")
            elif len(np.where(clean_eig[i] < 0.0)[0]) == len(clean_eig[i]): classifier.append("maximum")
            elif len(np.where(clean_eig[i] == 0.0)[0])  > 0: classifier.append("zero curvature")
            elif len(np.where(clean_eig[i] < 0.0)[0])  < len(clean_eig[i]): classifier.append("saddle point")
            else: classifier.append("ERROR")

        optima_list =  {"x":       np.vstack([self.list["x"],clean_x]), \
                        "func evals":   np.append(self.list["func evals"],clean_f), \
                        "classifier":   self.list["classifier"] + classifier, \
                        "eigen values": np.vstack([self.list["eigen values"],clean_eig]),\
                        "gradient norm":np.append(self.list["gradient norm"],clean_grad_norm),\
                        "success": self.list["success"]}

        sort_indices = np.argsort(optima_list["func evals"])
        optima_list["x"] = optima_list["x"][sort_indices]
        optima_list["func evals"] = optima_list["func evals"][sort_indices]
        optima_list["classifier"] = [optima_list["classifier"][i] for i in sort_indices]
        optima_list["eigen values"] = optima_list["eigen values"][sort_indices]
        optima_list["gradient norm"] = optima_list["gradient norm"][sort_indices]
        self.list = dict(optima_list)
        return optima_list
    ####################################################
    def get_minima(self,n):
        try:
            index = [i for i,x in enumerate(self.list["classifier"]) if x == "minimum"]
            index = index[0:min(n,len(index))]
            return self.list["x"][index], self.list["func evals"][index]
        except:
            print("no minima available in the optima_list")
            return np.empty((0,self.dim)),np.empty((0))
    ####################################################
    def get_maxima(self,n):
        try:
            index = [i for i,x in enumerate(self.list["classifier"]) if x == "maximum"]
            index = index[0:min(n,len(index))]
            return self.list["x"][index], self.list["func evals"][index]
        except:
            print("no maxima available in the optima_list")
            return np.empty((0,self.dim)),np.empty((0))
    ####################################################
    def get_deflation_points(self,n):
        try:
            index = [i for i, x in enumerate(self.list["classifier"]) if x == "maximum" or x == "minimum" or x == "saddle point"]
            #index = index[0:n]
            return self.list["x"][index], self.list["func evals"][index]
        except:
            print("no deflation points available in the optima_list")
            return np.empty((0,self.dim)),np.empty((0))

