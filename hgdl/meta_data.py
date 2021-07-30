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
        self.bounds = obj.bounds
        self.radius = obj.radius
        self.dim = obj.dim
        self.local_max_iter = obj.local_max_iter
        self.number_of_walkers = obj.number_of_walkers
        self.num_epochs = obj.num_epochs
        self.global_optimizer = obj.global_optimizer
        self.local_optimizer = obj.local_optimizer
        self.args = obj.args
        self.constr = obj.constr
