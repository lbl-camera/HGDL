import numpy as np

def schwefel(x,arr,brr):
    x = x.astype(float)
    return 418.9829*len(x) - np.sum(x*np.sin(np.sqrt(np.abs(x))))
###########################################################################
def schwefel_gradient(x,*args):
    x = x.astype(float)
    indices = np.where(x==0)
    x[indices] = 0.0001
    return -(np.sin(np.sqrt(np.abs(x))) + (x*np.cos(np.sqrt(np.abs(x)))*(0.5/np.sqrt(np.abs(x))) * (np.sign(x))))
###########################################################################
def schwefel_hessian(x,*args):
    x = x.astype(float)
    e = 1e-4
    hessian = np.zeros((len(x),len(x)))
    for i in range(len(hessian)):
        x_aux1 = np.array(x)
        x_aux1[i] = x[i] + e
        x_aux2 = np.array(x)
        x_aux2[i] = x[i] - e
        a = (((schwefel_gradient(x_aux1,args) - schwefel_gradient(x_aux2,args))/(2.0*e)))
        hessian[i,i] = a[i]
    return hessian
###########################################################################
def non_diff(x):
    p = np.array([2,2])
    if np.linalg.norm(np.subtract(p,x))<1.0: return -1.0
    else: return 0.0
###########################################################################
def non_diff_grad(x):
    return np.zeros((len(x)))
###########################################################################
def non_diff_hess(x):
    return np.zeros((len(x),len(x)))
###########################################################################
