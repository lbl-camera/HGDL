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

    def __init__(self,nlc,nlc_grad, nlc_hess = None, ctype = "=", value = 0.0, multiplier = 0.0, slack = None, bounds = np.array([[-1e6,1e6]])):
        self.nlc = nlc
        self.nlc_grad = nlc_grad
        self.nlc_hess = nlc_hess
        self.ctype = ctype
        self.value = value
        self.lamb = multiplier
        self.slack = slack
        self.bounds = bounds
