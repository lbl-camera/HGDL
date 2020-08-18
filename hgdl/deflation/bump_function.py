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
    """
    d = np.sqrt((x - x0).T @ (x - x0))
    a = 1.0-(d**2/r**2)
    if a <= 0: return 0.0 
    else: return np.exp(-1.0/a) * np.exp(1.0)
###########################################################################
def b_grad(x,x0,r):
    d = np.sqrt((x - x0).T @ (x - x0))
    d2= (x - x0)
    gr = np.zeros((d2.shape))
    a = 1.0-(d**2/r**2)
    if a <= 0: return np.zeros((len(x)))
    else: return (b(x,x0,r) * ((-2.0*d2)/(a**2))) /r**2
###########################################################################
def deflation_function(x,x0,r):
    """
    input:
        x is one point(1d numpy array)
        x0 is a a 2d array of locations of the bump function
    """
    if len(x0) == 0: s = 0.0; return (1.0/(1.0-s))
    s = 0.0
    for i in range(len(x0)):
        s += b(x,x0[i],r)
    return (1.0/(1.0-s))
###########################################################################
def deflation_function_gradient(x,x0,r):
    """
    input:
        x is one point(1d numpy array)
        x0 is a a 2d array of locations of the bump function
    """
    if len(x0) == 0: return np.zeros((len(x)))
    s1 = 0.0
    s2 = np.zeros((len(x)))
    for i in range(len(x0)):
        s1 += b(x,x0[i],r)
        s2 += b_grad(x,x0[i],r)
    return (1.0/((1.0-s1)**2)) * s2

