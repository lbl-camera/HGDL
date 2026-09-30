###local optimizer for hgdl
import numpy as np


def deflated_grad(x, *args, grad_func=None, x_defl=[], radius=[]):
    d = deflation_function(x, x_defl, radius)
    return d * grad_func(x, *args)


def deflated_hess(x, *args, grad_func=None, hess_func=None, x_defl=[], radius=[]):
    d, dg = deflation_and_gradient(x, x_defl, radius)
    return (hess_func(x, *args) * d) + np.outer(grad_func(x, *args), dg)


########################################################
########################################################
########################################################
def b(x, x0, r):
    """
    evaluates the bump function
    x ... a point (1d numpy array)
    x0 ... 1d numpy array of location of bump function, a 2d numpy array

    returns the bump function b(x,x0) with radius r
    """
    d = np.sqrt((x - x0).T @ (x - x0))
    a = 1.0 - (d ** 2 / r ** 2)
    if a <= 0:
        return 0.0
    else:
        return np.exp(-1.0 / a) * np.exp(1.0)


###########################################################################
def b_grad(x, x0, r):
    """evaluates the bump function gradient b(x,x0) with radius r
       x ... point
       x0... location of bump
       r ... radius of bump function
    """
    d = np.sqrt((x - x0).T @ (x - x0))
    d2 = (x - x0)
    a = 1.0 - (d ** 2 / r ** 2)
    if a <= 0:
        return np.zeros((len(x)))
    else:
        return (b(x, x0, r) * ((-2.0 * d2) / (a ** 2))) / r ** 2


###########################################################################
# a bump is exactly 1 at its center; it is capped so the deflation factor stays finite
BUMP_CAP = 0.99999


def _bumps(x, x0, r):
    """
    All bumps b(x, x0_i, r_i) and their gradients at once, for K deflation points:
    returns an array (K) and an array (K x D). Same values as b() and b_grad().
    """
    x0 = np.asarray(x0, dtype=float).reshape(len(x0), -1)
    r = np.asarray(r, dtype=float)
    diff = x - x0
    a = 1.0 - np.sum(diff ** 2, axis=1) / r ** 2
    inside = a > 0
    bumps = np.zeros(len(x0))
    grads = np.zeros(x0.shape)
    a_in, r_in = a[inside], r[inside]
    bumps[inside] = np.exp(-1.0 / a_in) * np.exp(1.0)
    grads[inside] = (bumps[inside] * -2.0 / (a_in ** 2 * r_in ** 2))[:, None] * diff[inside]
    return bumps, grads


def deflation_and_gradient(x, x0, r):
    """
    The deflation operator D(x) = prod_i 1/(1 - bump_i(x)) and its gradient
    D(x) * sum_i grad(bump_i) / (1 - bump_i). Each factor is >= 1, so overlapping
    bumps cannot make the operator negative (a sum of bumps can exceed 1 and flip
    its sign). Outside every bump D = 1 and the gradient is 0.
    """
    if len(x0) == 0: return 1.0, np.zeros((len(x)))
    bumps, grads = _bumps(x, x0, r)
    one_minus = 1.0 - np.minimum(bumps, BUMP_CAP)
    d = np.prod(1.0 / one_minus)
    return d, d * np.sum(grads / one_minus[:, None], axis=0)


def deflation_function(x, x0, r):
    """
    input:
        x is one point(1d numpy array)
        x0 is a a 2d array of locations of the bump function
    the return is the deflation operator, the product over all deflation points of
    1.0/(1.0 - bump(x,x0_i)).
    """
    return deflation_and_gradient(x, x0, r)[0]


###########################################################################
def deflation_function_gradient(x, x0, r):
    """
    input:
        x is one point(1d numpy array)
        x0 is a 2d array of locations of the bump function
    the return is the gradient of the deflation operator,
    D(x) * sum_i grad(bump_i) / (1 - bump_i), with D the operator above.
    """
    return deflation_and_gradient(x, x0, r)[1]
