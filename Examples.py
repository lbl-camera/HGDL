# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from HGDL import HGDL

# ## Testing
# ### Rastringin function

# ### Rastringin function - Simple (no args, no params)

A = 10
d = 2
def Rastringin(x):
    return (A*d + np.dot(x,x) - A*np.sum(np.cos(2.*np.pi*x)))

def Rastringin_gradient(x):
    grad = np.empty(len(x))
    for i in range(len(grad)):
        grad[i] = (2.*x[i] + A*np.sin(2.*np.pi*x[i])*2.*np.pi)
    return grad

def Rastringin_hessian(x):
    hess = np.zeros((len(x),len(x)))
    for i in range(len(hess)):
        hess[i,i] = (2 + A*np.cos(2.*np.pi*x[i])*(4.*np.pi*np.pi))
    return hess

X = np.arange(-5,5,1e-1)
Y = np.arange(-5,5,1e-1)
Z = np.empty((len(X),len(Y)))

for i in range(len(X)):
    for j in range((len(Y))):
        Z[i,j] = Rastringin(np.array([X[i],Y[j]]))
plt.contourf(X,Y,Z); plt.colorbar();
plt.show()

k = 2
bounds = np.ones((k,2))*5; bounds[:,0] = bounds[:,1]*(-1);
bounds;

res = HGDL(Rastringin, Rastringin_gradient, bounds).run()
res = res['x']

plt.scatter(res[:,0], res[:,1], color='black');
plt.contour(X,Y,Z); plt.colorbar();
plt.show()

res = HGDL(function=Rastringin, bounds=bounds, hess=Rastringin_hessian, gradient=Rastringin_gradient, method='Newton-CG').run()
res = res['x']

plt.scatter(res[:,0], res[:,1], color='black');
plt.contour(X,Y,Z); plt.colorbar();
plt.show()


# ### Rastringin function - With args (no params)
A = 10
d = 2
def Rastringin(x, origin):
    x = x - origin
    return (A*d + np.dot(x,x) - A*np.sum(np.cos(2.*np.pi*x)))

def Rastringin_gradient(x, origin):
    x = x - origin
    grad = np.empty(len(x))
    for i in range(len(grad)):
        grad[i] = (2.*x[i] + A*np.sin(2.*np.pi*x[i])*2.*np.pi)
    return grad


# In[ ]:


def Rastringin_hessian(x, origin):
    x = x - origin
    hess = np.zeros((len(x),len(x)))
    for i in range(len(hess)):
        hess[i,i] = (2 + A*np.cos(2.*np.pi*x[i])*(4.*np.pi*np.pi))
    return hess


# In[ ]:


X = np.arange(0,10,1e-1)
Y = np.arange(0,10,1e-1)


# In[ ]:


Z = np.empty((len(X),len(Y)))


# In[ ]:


for i in range(len(X)):
    for j in range((len(Y))):
        Z[i,j] = Rastringin(np.array([X[i],Y[j]]), 5*np.ones(2))


# ### Notice that the center is shifted

# In[ ]:


plt.contourf(X,Y,Z); plt.colorbar();


# In[ ]:


k = 2
bounds = np.ones((k,2))*10; bounds[:,0] = 0;
bounds


# ### Notice this little switcheroo, where I put in the origin that I want

# In[ ]:


from functools import partial


# In[ ]:


o = 5*np.ones(2)
NewRastringin = partial(Rastringin, origin = o)
NewRastringin_gradient = partial(Rastringin_gradient, origin = o)
NewRastringin_hessian = partial(Rastringin_hessian, origin = o)


# In[ ]:


res = HGDL(fun=NewRastringin, bounds=bounds, hess=NewRastringin_hessian, jac=NewRastringin_gradient)


# In[ ]:


res['success']


# In[ ]:


res = res['x']


# In[ ]:


plt.scatter(res[:,0], res[:,1], color='black');
plt.contour(X,Y,Z); plt.colorbar();


# In[ ]:


parameters = np.dtype([('numWorkers','i2'), # how many processes to run
                        ('radius_squared','f8'), # how close is close enough for bump function
                        ('maxCount','i4'), # how many iterations of local method
                        ('alpha','f8'), # alpha parameter of bump function
                        ('unfairness','f8'), # how unfair is the global method
                        ('wildness','f8'), # how much randomness is in the global method
                       # Note - set wildness high to be high randomness
                        ('N','i4'), # how many individuals to have
                        ('keepLastX','i2'), # converge when best of last x runs is the same
                        ('maxRuns','i4'), # maximum number of iterations of local method
                        ('returnedThreshold','f8'), # what threshold of % local failed is enough
                        ('verbose',bool), # print out best at each step
                        ('k','i4'), # reserved by algorithm 
                        ('numGenerations','i4'), # reserved by algorithm
                      ])

parameters = np.recarray(1, parameters)

parameters.numWorkers = -1
parameters.maxCount = 3
parameters.alpha = .1
parameters.unfairness = 2.5
parameters.wildness = 1
parameters.N = 3
parameters.keepLastX = 3
parameters.maxRuns = 2
parameters.returnedThreshold=0.7
parameters.verbose = False
parameters = parameters[0]


# In[ ]:


def f(x):
    return 0
def grad(x):
    return np.ones(x.shape[0])
def hess(x):
    return np.zeros((x.shape[0],x.shape[0]))


# In[ ]:


bounds = np.ones((3, 2))


# In[ ]:


bounds[:,0] *= -10.
bounds[:,1] *= 10


# In[ ]:


bounds


# In[ ]:


get_ipython().run_line_magic('run', 'HGDL.ipynb')


# In[ ]:


get_ipython().run_line_magic('load_ext', 'line_profiler')


# In[ ]:


get_ipython().run_line_magic('lprun', '-f HGDL HGDL(fun=f, bounds=bounds, jac=grad, parameters=parameters)')


# In[ ]:


res = HGDL(fun=f, bounds=bounds, jac=grad, parameters=parameters)


# In[ ]:


res['x'].shape


# In[ ]:


res['x'].round(3)


# In[ ]:


res['success']


# In[ ]:





# In[ ]:





# ### And you can see that the modified function works too!

# ### Rastringin function - With args and custom params

# Notice that the rastringin function is staying the same as the last time

# In[ ]:


p = defaultParams()


# In[ ]:


#p is of this type:
np.dtype([('numWorkers','i2'), # how many processes to run
                            ('radius_squared','f8'), # how close is close enough for bump function
                            ('maxCount','i4'), # how many iterations of local method
                            ('alpha','f8'), # alpha parameter of bump function
                            ('unfairness','f8'), # how unfair is the global method
                            ('wildness','f8'), # how much randomness is in the global method
                           # Note - set wildness high to be high randomness
                            ('N','i4'), # how many individuals to have
                            ('keepLastX','i2'), # converge when best of last x runs is the same
                            ('maxRuns','i4'), # maximum number of iterations of local method
                            ('returnedThreshold','f8'), # what threshold of % local failed is enough
                            ('verbose','b'), # print out best at each step
                            ('k','i4'), # reserved by algorithm 
                            ('numGenerations','i4'), # reserved by algorithm
                          ]);


# In[ ]:


p.N


# In[ ]:


res = HGDL(fun=NewRastringin, bounds=bounds, hess=NewRastringin_hessian, jac=NewRastringin_gradient, parameters=p)


# In[ ]:


res['success']


# In[ ]:


res = res['x']


# In[ ]:


plt.scatter(res[:,0], res[:,1], color='black');
plt.contour(X,Y,Z); plt.colorbar();


# ### And you can see that when I used N=3 it was able to explore far less

# ### Schwefel Function

# In[ ]:


def Schwefel(x):
    total = 0
    for i in range(len(x)):
        if np.abs(x[i]) <= 500:
            total += -x[i]*np.sin(np.sqrt(np.abs(x[i])))
        else:
            total += .02*x[i]*x[i]
    return -(-418.9829*(len(x)+1) - total)


# In[ ]:


def Schwefel_gradient(X):
    y = np.empty(len(X))

    Sin = np.sin
    Abs = np.abs
    Cos = np.cos
    Sqrt = np.sqrt

    for i in range(len(X)):
        x = X[i]

        if x<0:
            slope = -1.
        else:
            slope = 1.

        y[i] = -Sin(Sqrt(Abs(x))) - x*Cos(Sqrt(Abs(x)))*slope/(2.*Sqrt(Abs(x)))
    return -y


# In[ ]:


def Schwefel_hessian(x):
    Sin = np.sin
    Abs = np.abs
    Cos = np.cos
    Sqrt = np.sqrt
        
    hess = np.zeros((len(x), len(x)))
    
    for i in range(len(x)):
        xi = x[i]
        if xi<0:
            slope = -1.
        else:
            slope = 1.

        factor = 1./(4.*Abs(xi)**1.5)
        term1 = xi*Sqrt(Abs(xi))*Sin(Sqrt(Abs(xi)))
        term2 = Cos(Sqrt(Abs(xi)))*(xi-2.*Abs(xi)*(2.*slope))
        hess[i,i] = factor*(term1+term2)
    return - hess


# In[ ]:


X = np.arange(-500,500,1e0)
Y = np.arange(-500,500,1e0)


# In[ ]:


Z = np.empty((len(X),len(Y)))


# In[ ]:


for i in range(len(X)):
    for j in range((len(Y))):
        Z[i,j] = Schwefel(np.array([X[i],Y[j]]))


# In[ ]:


plt.contourf(X,Y,Z); plt.colorbar();


# In[ ]:


k = 2
bounds = np.ones((k,2))*500; bounds[:,0] = bounds[:,1]*(-1);
bounds;


# In[ ]:


res = HGDL(fun=Schwefel, bounds=bounds, jac=Schwefel_gradient); 


# In[ ]:


res['success']


# In[ ]:


res['x']


# In[ ]:


plt.scatter(res[:,0], res[:,1], color='black');
plt.contour(X,Y,Z); plt.colorbar();


# In[ ]:


k = 4
bounds = np.ones((k,2))*500; bounds[:,0] = bounds[:,1]*(-1);
bounds;


# In[ ]:


get_ipython().run_line_magic('run', 'HGDL.ipynb')


# This call is taking a long time

# In[ ]:


res = HGDL(fun=Schwefel, bounds=bounds, jac=Schwefel_gradient, hess=Schwefel_hessian); 


# In[ ]:


plt.scatter(res[:,0], res[:,1], color='black');
plt.contour(X,Y,Z); plt.colorbar();


# In[ ]:


res[:7].round(4)


# In[ ]:





# ### Rosen function

# In[ ]:


from scipy.optimize import rosen, rosen_der, rosen_hess


# In[ ]:


X = np.arange(-2.5, 2.5, 1e-2)
Y = np.arange(-2.5, 2.5, 1e-2)


# In[ ]:


Z = np.empty((len(X),len(Y)))


# In[ ]:


for i in range(len(X)):
    for j in range((len(Y))):
        Z[i,j] = rosen(np.array([X[i],Y[j]]))


# In[ ]:


plt.contourf(X,Y,Z); plt.colorbar();


# In[ ]:


k = 2
bounds = np.ones((k,2))*3; bounds[:,0] = bounds[:,1]*(-1);
bounds;


# In[ ]:


res = HGDL(fun=rosen, bounds=bounds, args=(), jac=rosen_der, tol=1e-8, 
                                  hess=rosen_hess, epsilon=1e-8, maxCount=50, alpha=.1, cauchy_wildness=50,
                                  unfairness=2, N=100, keepLastX = 10, numWorkers=-1, method='Newton-CG'); 


# In[ ]:


plt.scatter(res[:,0], res[:,1], color='black');
plt.contour(X,Y,Z); plt.colorbar();


# In[ ]:


res


# In[ ]:


def f(x,scale=1., sub=0):
    return np.sum(x)*scale-sub


# In[ ]:


def wrapper(x0, fun, *args, **kwargs):
    return fun(x0, *args, **kwargs)


# In[ ]:


wrapper(np.ones(2), f, *(2, 1))


# In[ ]:




