###########################################################################
###########################################################################
##################TEST FUNCTION############################################
###########################################################################
###########################################################################
import numpy as np
import matplotlib.pyplot as plt
import scipy

def schwefel(x,arr,brr):
    x = x.astype(float)
    #print(arr,brr)
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
def plot_schwefel(bounds = [[-500,500],[-500,500]], resolution = 100, points = None, deflation_points = None):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm

    X = np.linspace(bounds[0][0], bounds[0][1], resolution)
    Y = np.linspace(bounds[1][0], bounds[1][1], resolution)
    X, Y = np.meshgrid(X, Y)
    schwefel = np.empty((X.shape))
    gr = np.empty((X.shape))

    for i in range(len(X)):
        for j in range(len(Y)):
            schwefel[i,j] = self.schwefel(np.array([X[i,j],Y[i,j]]))
            if deflation_points is not None:
                gr[i,j] = self.schwefel_gradient(np.array([X[i,j],Y[i,j]]))[0] * self.deflation_function(np.array([[X[i,j],Y[i,j]]]), deflation_points)
                #gr[i,j] = self.deflation_function(np.array([[X[i,j],Y[i,j]]]), deflation_points)


    fig = plt.figure(0)
    a = plt.pcolormesh(X, Y, schwefel, cmap=cm.viridis)
    plt.colorbar(a)
    if points is not None: plt.scatter(points[:,0], points[:,1])

    if len(deflation_points) != 0: plt.scatter(deflation_points[:,0], deflation_points[:,1])
    plt.show()
###########################################################################
