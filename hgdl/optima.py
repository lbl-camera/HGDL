import numpy as np
import time

from loguru import logger
import hgdl.misc as misc
import hgdl.local_methods as local
import hgdl.global_methods as glob
import dask.distributed as distributed
import dask.multiprocessing
from dask.distributed import as_completed

class optima:
    """
    stores all results and adaptations of it
    """
    def __init__(self,dim_x, max_optima):
        """
        input:
        -----
            dim ... the dimensionality of the space
            max_optima ... maximum number of stored optima
        """

        self.dim_x = dim_x
        self.max_optima = max_optima
        self.list = {"x": np.empty((0,self.dim_x)),
                     "f(x)": np.empty((0)),
                     "classifier": [],
                     "Hessian eigvals": np.empty((0,self.dim_x)),
                     "df/dx": np.empty((0,self.dim_x)),
                     "|df/dx|":np.empty((0)),
                     "local success": np.empty((0)),
                     "notes": np.empty((0)),
                     "success": False}
    ####################################################
    def fill_in_optima_list(self,res):
        x,f,fg,eig,local_success,messages = res[0],res[1],res[2],res[3],res[4],res[5]
        if not np.any(local_success) and len(self.list["x"]) == 0: 
            local_success[:] = True
            print("WARNING: No local walker converged.")
        clean_indices = np.where(np.asarray(local_success) == True)[0]
        if len(clean_indices) == 0:
            return {"x": self.list["x"],
                    "f(x)": self.list["f(x)"],
                    "classifier": self.list["classifier"],
                    "Hessian eigvals": self.list["Hessian eigvals"],
                    "df/dx": self.list["df/dx"],
                    "|df/dx|": self.list["|df/dx|"],
                    "success": self.list["success"]}

        clean_x = x[clean_indices]
        clean_f = f[clean_indices]
        clean_fg = fg[clean_indices]
        clean_eig = eig[clean_indices]
        classifier = []
        for i in range(len(clean_x)):
            if any(clean_fg[i] > 1e-5): classifier.append("degenerate")
            elif any(np.isnan(clean_eig[i])): classifier.append("optimum")
            elif all(clean_eig[i] > 0.0): classifier.append("minimum")
            elif all(clean_eig[i] < 0.0): classifier.append("maximum")
            elif any(abs(clean_eig[i]) < 10e-6): classifier.append("zero curvature")
            elif len(np.where(clean_eig[i] < 0.0)[0])  < len(clean_eig[i]): classifier.append("saddle point")
            else: classifier.append("ERROR")

        optima_list =  {"x":       np.vstack([self.list["x"],clean_x]),
                        "f(x)":    np.append(self.list["f(x)"],clean_f),
                        "classifier":   self.list["classifier"] + classifier,
                        "Hessian eigvals": np.vstack([self.list["Hessian eigvals"],clean_eig]),
                        "df/dx":np.vstack([self.list["df/dx"],clean_fg]),
                        "|df/dx|":np.append(self.list["|df/dx|"],np.linalg.norm(clean_fg, axis = 1)),
                        "success": True}

        sort_indices = np.argsort(optima_list["f(x)"])
        optima_list["x"] = optima_list["x"][sort_indices][0:self.max_optima]
        optima_list["f(x)"] = optima_list["f(x)"][sort_indices][0:self.max_optima]
        optima_list["classifier"] = [optima_list["classifier"][i] for i in sort_indices][0:self.max_optima]
        optima_list["Hessian eigvals"] = optima_list["Hessian eigvals"][sort_indices][0:self.max_optima]
        optima_list["df/dx"] = optima_list["df/dx"][sort_indices][0:self.max_optima]
        optima_list["|df/dx|"] = optima_list["|df/dx|"][sort_indices][0:self.max_optima]
        self.list = dict(optima_list)
        return optima_list

    def get_minima(self,n):
        try:
            index = [i for i,x in enumerate(self.list["classifier"]) if x == "minimum"]
            index = index[0:min(n,len(index))]
            return self.list["x"][index], self.list["f(x)"][index]
        except:
            logger.debug("no minima available in the optima_list")
            return np.empty((0,self.dim_x)),np.empty((0))
    ####################################################
    def get_maxima(self,n):
        try:
            index = [i for i,x in enumerate(self.list["classifier"]) if x == "maximum"]
            index = index[0:min(n,len(index))]
            return self.list["x"][index], self.list["f(x)"][index]
        except:
            logger.debug("no maxima available in the optima_list")
            return np.empty((0,self.dim_x)),np.empty((0))
    ####################################################
    def get_deflation_points(self,n):
        try:
            index = [i for i, x in enumerate(self.list["classifier"]) if x == "maximum" or x == "minimum" or x == "zero curvature" or x == "saddle point" or x == "optimum"]
            return self.list["x"][index], self.list["f(x)"][index]
        except:
            logger.debug("no deflation points available in the optima_list")
            return np.empty((0,self.dim_x+self.dim_k)),np.empty((0))

#########################################################
#########################################################
#########################################################
#########################################################
#########################################################
class optima_constr:
    """
    stores all results and adaptations of it
    """
    def __init__(self,dim_x,dim_k, max_optima):
        """
        input:
        -----
            dim ... the dimensionality of the space
            max_optima ... maximum number of stored optima
        """

        self.dim_x = dim_x
        self.dim_k = dim_k
        self.dim = self.dim_x + self.dim_k

        self.max_optima = max_optima
        self.list = {"x": np.empty((0,self.dim_x)),
                     "k": np.empty((0,self.dim_k)),
                     "f(x)": np.empty((0)),
                     "classifier": [],
                     "Hessian eigvals": np.empty((0,self.dim_x)),
                     "dL/dx": np.empty((0,self.dim_x)),
                     "dL/dk": np.empty((self.dim_k)),
                     "|dL/dx|":np.empty((0)),
                     "|dL/dk|":np.empty((0)),
                     "success": False}
    ####################################################
    def fill_in_optima_list(self,res):
        x,f,Lg,eig,local_success = res[0],res[1],res[2],res[3],res[4]
        eig = eig[:,0:self.dim_x]
        if not np.any(local_success) and len(self.list["x"]) == 0: local_success[:] = True
        clean_indices = np.where(np.asarray(local_success) == True)[0]

        if len(clean_indices) == 0:
            return {"x": self.list["x"][:,0:self.dim_x],
                    "k": self.list["k"][:,self.dim_x:self.dim],
                    "f(x)": self.list["f(x)"],
                    "classifier": self.list["classifier"],
                    "Hessian eigvals": self.list["Hessian eigvals"],
                    "dL/dx": self.list["dL/dx"],
                    "dL/dk": self.list["dL/dk"],
                    "|dL/dx|": self.list["|dL/dx|"],
                    "|dL/dk|": self.list["|dL/dk|"],
                    "success": self.list["success"]}

        clean_x = x[clean_indices]
        clean_f = f[clean_indices]
        clean_Lg = Lg[clean_indices]
        clean_eig = eig[clean_indices]
        classifier = []
        for i in range(len(clean_x)):
            if any(clean_Lg[i] > 1e-5): classifier.append("degenerate")
            elif any(np.isnan(clean_eig[i])): classifier.append("optimum")
            elif all(clean_eig[i] > 0.0): classifier.append("minimum")
            elif all(clean_eig[i] < 0.0): classifier.append("maximum")
            elif any(abs(clean_eig[i]) < 10e-6): classifier.append("zero curvature")
            elif len(np.where(clean_eig[i] < 0.0)[0])  < len(clean_eig[i]): classifier.append("saddle point")
            else: classifier.append("ERROR")


        optima_list =  {"x":       np.vstack([self.list["x"],clean_x[:,0:self.dim_x]]),
                        "k":       np.vstack([self.list["k"],clean_x[:,self.dim_x:self.dim]]),
                        "f(x)":    np.append(self.list["f(x)"],clean_f),
                        "classifier":   self.list["classifier"] + classifier,
                        "Hessian eigvals": np.vstack([self.list["Hessian eigvals"],clean_eig]),
                        "dL/dx":np.vstack([self.list["dL/dx"],clean_Lg[:,0:self.dim_x]]),
                        "|dL/dx|":np.append(self.list["|dL/dx|"],np.linalg.norm(clean_Lg[:,0:self.dim_x],axis = 1)),
                        "dL/dk":np.vstack([self.list["dL/dk"],clean_Lg[:,self.dim_x:self.dim]]),
                        "|dL/dk|":np.append(self.list["|dL/dk|"],np.linalg.norm(clean_Lg[:,self.dim_x:self.dim],axis = 1)),
                        "success": True}

        sort_indices = np.argsort(optima_list["f(x)"])
        optima_list["x"] = optima_list["x"][sort_indices][0:self.max_optima]
        optima_list["k"] = optima_list["k"][sort_indices][0:self.max_optima]
        optima_list["f(x)"] = optima_list["f(x)"][sort_indices][0:self.max_optima]
        optima_list["classifier"] = [optima_list["classifier"][i] for i in sort_indices][0:self.max_optima]
        optima_list["Hessian eigvals"] = optima_list["Hessian eigvals"][sort_indices][0:self.max_optima]
        optima_list["dL/dx"] = optima_list["dL/dx"][sort_indices][0:self.max_optima]
        optima_list["dL/dk"] = optima_list["dL/dk"][sort_indices][0:self.max_optima]
        optima_list["|dL/dx|"] = optima_list["|dL/dx|"][sort_indices][0:self.max_optima]
        optima_list["|dL/dk|"] = optima_list["|dL/dk|"][sort_indices][0:self.max_optima]
        self.list = dict(optima_list)
        return optima_list
    ####################################################
    def get_minima(self,n):
        try:
            index = [i for i,x in enumerate(self.list["classifier"]) if x == "minimum"]
            index = index[0:min(n,len(index))]
            return self.list["x"][index], self.list["f(x)"][index]
        except:
            logger.debug("no minima available in the optima_list")
            return np.empty((0,self.dim_x)),np.empty((0))
    ####################################################
    def get_maxima(self,n):
        try:
            index = [i for i,x in enumerate(self.list["classifier"]) if x == "maximum"]
            index = index[0:min(n,len(index))]
            return self.list["x"][index], self.list["f(x)"][index]
        except:
            logger.debug("no maxima available in the optima_list")
            return np.empty((0,self.dim_x)),np.empty((0))
    ####################################################
    def get_deflation_points(self,n):
        try:
            index = [i for i, x in enumerate(self.list["classifier"]) if x == "maximum" or x == "minimum" or x == "zero curvature" or x == "saddle point" or x == "optimum"]
            return np.column_stack([self.list["x"][index],self.list["k"][index]]), self.list["f(x)"][index]
        except:
            logger.debug("no deflation points available in the optima_list")
            return np.empty((0,self.dim_x+self.dim_k)),np.empty((0))

