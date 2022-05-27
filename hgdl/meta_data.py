import numpy as np


class meta_data:
    """
    this class contains all the data necessary to run the workers
    """
    def __init__(self,obj):
        self.workers = obj.workers    ###dictionary of host and walker workers
        self.x0 = obj.x0
        self.func = obj.func
        self.grad = obj.grad
        self.hess = obj.hess
        self.L = obj.L
        self.Lgrad = obj.Lgrad
        self.Lhess = obj.Lhess
        self.bounds = obj.bounds
        self.radius = obj.radius
        self.dim_x = obj.dim_x
        self.dim_k = obj.dim_k
        self.dim = obj.dim_x + self.dim_k

        self.local_max_iter = obj.local_max_iter
        self.number_of_walkers = obj.number_of_walkers
        self.num_epochs = obj.num_epochs
        self.global_optimizer = obj.global_optimizer
        self.local_optimizer = obj.local_optimizer
        self.args = obj.args
        self.constr = obj.constr
        self.tolerance = obj.tolerance
