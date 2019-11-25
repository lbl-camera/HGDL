#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# ## Testing
# ### Rastringin function

# In[2]:


A = 10
d = 2
def Rastringin(x):
    return (A*d + np.dot(x,x) - A*np.sum(np.cos(2.*np.pi*x)))


# In[3]:


def Rastringin_gradient(x):
    grad = np.empty(len(x))
    for i in range(len(grad)):
        grad[i] = (2.*x[i] + A*np.sin(2.*np.pi*x[i])*2.*np.pi)
    return grad


# In[4]:


def Rastringin_hessian(x):
    hess = np.zeros((len(x),len(x)))
    for i in range(len(hess)):
        hess[i,i] = (2 + A*np.cos(2.*np.pi*x[i])*(4.*np.pi*np.pi))
    return hess


# In[5]:


X = np.arange(-5,5,1e-1)
Y = np.arange(-5,5,1e-1)


# In[6]:


Z = np.empty((len(X),len(Y)))


# In[7]:


for i in range(len(X)):
    for j in range((len(Y))):
        Z[i,j] = Rastringin(np.array([X[i],Y[j]]))


# In[8]:


plt.contourf(X,Y,Z); plt.colorbar();


# In[9]:


k = 2
bounds = np.ones((k,2))*5; bounds[:,0] = bounds[:,1]*(-1);
bounds;


# In[5]:


get_ipython().run_line_magic('run', 'HXDY.ipynb')


# In[11]:


res = HXDY(fun=Rastringin, bounds=bounds, args=(), jac=Rastringin_gradient, tol=1e-2, 
                                  hess=Rastringin_hessian, epsilon=1e-8, maxCount=30, alpha=.1, 
                                  unfairness=5, N=100, keepLastX = 5, numWorkers=-1, method='Newton-CG'); 


# In[12]:


plt.scatter(res[:,0], res[:,1], color='black');
plt.contour(X,Y,Z); plt.colorbar();


# ### Schwefel Function

# In[6]:


def Schwefel(x):
    total = 0
    for i in range(len(x)):
        if np.abs(x[i]) <= 500:
            total += -x[i]*np.sin(np.sqrt(np.abs(x[i])))
        else:
            total += .02*x[i]*x[i]
    return -(-418.9829*(len(x)+1) - total)


# In[7]:


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


# In[8]:


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


# In[9]:


X = np.arange(-500,500,1e0)
Y = np.arange(-500,500,1e0)


# In[10]:


Z = np.empty((len(X),len(Y)))


# In[11]:


for i in range(len(X)):
    for j in range((len(Y))):
        Z[i,j] = Schwefel(np.array([X[i],Y[j]]))


# In[12]:


plt.contourf(X,Y,Z); plt.colorbar();


# In[13]:


k = 2
bounds = np.ones((k,2))*500; bounds[:,0] = bounds[:,1]*(-1);
bounds;


# In[14]:


from time import process_time


# In[15]:


get_ipython().run_line_magic('run', 'HXDY.ipynb')


# In[16]:


def wrapper_boi(params):
    unfairness, cauchy_wildness, tol, maxCount, alpha = params
    TOTAL = 0
    NOW = process_time()
    for i in range(5):
        res = HXDY(fun=Schwefel, bounds=bounds, args=(), jac=Schwefel_gradient, tol=tol, 
                                      hess=Schwefel_hessian, epsilon=1e-2, maxCount=maxCount, alpha=alpha, cauchy_wildness=cauchy_wildness,
                                      unfairness=unfairness, N=100, keepLastX = 10, numWorkers=3, method='Newton-CG');
        TOTAL += res[0,-1]
    LATER = process_time()
    TOTAL += LATER - NOW
    print(params, TOTAL)
    return TOTAL


# In[17]:


from scipy.optimize import minimize


# In[18]:


minimize(wrapper_boi, x0=np.array([2, 50, 1e-2, 50, 1.])


# In[ ]:


unfairness, cauchy_wildness, tol, maxCount, alpha = 2, 50, 1e-2, 50, 1


# In[31]:


res = HXDY(fun=Schwefel, bounds=bounds, args=(), jac=Schwefel_gradient, tol=1e-4, 
                                  hess=Schwefel_hessian, epsilon=1e-8, maxCount=20, alpha=1., 
                                  unfairness=5, N=100, keepLastX = 3, numWorkers=-1, method='L-BFGS-B',
          extraStoppingCriterion=lambda res: True if len(res)>30 else False); 


# In[32]:


res


# In[33]:


res.shape


# In[34]:


plt.scatter(res[:,0], res[:,1], color='black');
plt.contour(X,Y,Z); plt.colorbar();


# In[35]:


k = 4
bounds = np.ones((k,2))*500; bounds[:,0] = bounds[:,1]*(-1);
bounds;


# In[36]:


get_ipython().run_line_magic('run', 'HXDY.ipynb')


# In[37]:


res = HXDY(fun=Schwefel, bounds=bounds, args=(), jac=Schwefel_gradient, tol=10000, 
                                  hess=Schwefel_hessian, epsilon=1e-2, maxCount=50, alpha=.1, cauchy_wildness=50,
                                  unfairness=2, N=100, keepLastX = 10, numWorkers=-1, method='Newton-CG'); 


# In[58]:


process_time()


# In[66]:


bounds


# In[ ]:





# In[38]:


res


# In[ ]:





# In[35]:


get_ipython().run_line_magic('load_ext', 'line_profiler')


# In[36]:


get_ipython().run_line_magic('lprun', "-f walk_individuals -f HXDY HXDY(fun=Schwefel, bounds=bounds, args=(), jac=Schwefel_gradient, tol=10000, hess=Schwefel_hessian, epsilon=1e-2, maxCount=50, alpha=.1, cauchy_wildness=50, unfairness=2, N=100, keepLastX = 10, numWorkers=-1, method='Newton-CG');")


# In[37]:


plt.scatter(res[:,0], res[:,1], color='black');
plt.contour(X,Y,Z); plt.colorbar();


# In[38]:


res[0]


# In[39]:


res[1]


# In[40]:


res.shape


# In[41]:


res


# ### Rosen function

# In[42]:


from scipy.optimize import rosen, rosen_der, rosen_hess


# In[43]:


X = np.arange(-2.5, 2.5, 1e-2)
Y = np.arange(-2.5, 2.5, 1e-2)


# In[44]:


Z = np.empty((len(X),len(Y)))


# In[45]:


for i in range(len(X)):
    for j in range((len(Y))):
        Z[i,j] = rosen(np.array([X[i],Y[j]]))


# In[46]:


plt.contourf(X,Y,Z); plt.colorbar();


# In[47]:


k = 2
bounds = np.ones((k,2))*3; bounds[:,0] = bounds[:,1]*(-1);
bounds;


# In[48]:


res = HXDY(fun=rosen, bounds=bounds, args=(), jac=rosen_der, tol=1e-8, 
                                  hess=rosen_hess, epsilon=1e-8, maxCount=50, alpha=.1, cauchy_wildness=50,
                                  unfairness=2, N=100, keepLastX = 10, numWorkers=-1, method='Newton-CG'); 


# In[49]:


plt.scatter(res[:,0], res[:,1], color='black');
plt.contour(X,Y,Z); plt.colorbar();


# In[50]:


res


# In[ ]:




