import numpy as np



class NonLinearConstraint:
    """
    This class provided HGDL with the capability of constrained function
    optimization. Every constraint should be fined by one class instance.

    Parameters:
    -----------
    nlc : Callable
        The function of the constraint.
    nlc_grad : Callable
        The gradient of the constraint
    nlc_hess : Callable, optional
        The Hessian of the constraint, if available. 
        Is only used for second-order local optimization.
        Default = None.
    ctype : str, optional
        String '<', '>', or '=', defining the constraint as inequality or equalit constraint.
        Default = '='.
    values : float, optional
        Value of the constraint. Default = 0.0.
    """

    def __init__(self,nlc,nlc_grad, nlc_hess = None, ctype = "=", value = 0.0, bounds = np.array([[-10000000,10000000]])):
        self.nlc = nlc
        self.nlc_grad = nlc_grad
        self.nlc_hess = nlc_hess
        self.ctype = ctype
        self.value = value
        self.bounds = bounds
        self.multiplier_index = None
        self.slack_index = None
        if ctype != "=" and len(bounds) == 1: 
            raise Exception("You have specified inequality constraints which need bounds sepcified for the multiplier and the slack varianble.")

    def set_multiplier_index(self,index):
        self.multiplier_index = index
    def set_slack_index(self,index):
        self.slack_index = index

