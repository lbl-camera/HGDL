###local optimizer for hgdl
import numpy as np
import hgdl.misc as misc
import asyncio
import time
import matplotlib.pyplot as plt
def b(x,x0,r):
    """
    evaluates the bump function
    x ... a point (1d numpy array)
    x0 ... 1d numpy array of location of bump function
    returns the bump function b(x,x0) with radius r
    """
    d = np.sqrt((x - x0).T @ (x - x0))
    a = 1.0-(d**2/r**2)
    if a <= 0: return 0.0
    else: return np.exp(-1.0/a) * np.exp(1.0)
###########################################################################
def b_grad(x,x0,r):
    """evaluates the bump function gradient b(x,x0) with radius r
       x ... point
       x0... location of bump
       r ... radius of bump function
    """
    d = np.sqrt((x - x0).T @ (x - x0))
    d2= (x - x0)
    a = 1.0 - (d**2/r**2)
    if a <= 0: return np.zeros((len(x)))
    else: return (b(x,x0,r) * ((-2.0*d2)/(a**2))) / r**2
###########################################################################
def deflation_function(x,x0,r):
    """
    input:
        x is one point(1d numpy array)
        x0 is a a 2d array of locations of the bump function
    the return is the deflation operator, e.g. 1.0/(1.0 - bump(x,x0))
    """
    if len(x0) == 0: return 1.0
    s = 0.0
    for i in range(len(x0)):
        s += b(x,x0[i],r)
    return 1.0/(1.0-s)
###########################################################################
def deflation_function_gradient(x,x0,r):
    """
    input:
        x is one point(1d numpy array)
        x0 is a a 2d array of locations of the bump function
    the return is the gradient of the deflation operator, e.g. (1.0/(1.0 - bump(x,x0)))'
    """
    if len(x0) == 0: return np.zeros((len(x)))
    s1 = 0.0
    s2 = np.zeros((len(x)))
    for i in range(len(x0)):
        s1 += b(x,x0[i],r)
        s2 += b_grad(x,x0[i],r)
    return s2/((1.0-s1)**2)


def deflated_grad(x, *args, grad_func = None, x_defl = [], radius = np.inf):
    d = deflation_function(x,x_defl,radius)
    return d*grad_func(x, *args)


def deflated_hess(x,*args, grad_func = None, hess_func = None, x_defl = [], 
                  radius = np.inf):
    d = deflation_function(x,x_defl,radius)
    dg = deflation_function_gradient(x,x_defl,radius)
    return (hess_func(x, *args)*d) + np.outer(grad_func(x, *args),dg)



#def deflated_solve( x, *args, grad_func = None, hess_func = None, x_defl=[],
#                    radius = 0.5, extended_return = False):
#    d = deflation_function(x,x_defl,radius)
#    dg = deflation_function_gradient(x,x_defl,radius)
#    #if extended_return == True: return hess_func(x, *args) + (np.outer(grad_func(x, *args),dg)/d),d,dg
#    return hess_func(x, *args)+ (np.outer(grad_func(x, *args),dg) * d)


