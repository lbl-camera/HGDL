#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


data = np.loadtxt('data/qsar_aquatic_toxicity.csv', delimiter=';')
X, y = data[:,:-1], data[:,-1]


# In[3]:


from matplotlib.colors import LogNorm
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


# In[4]:


from sklearn.gaussian_process.kernels import WhiteKernel, Matern
kernel = 1.**2 * Matern(length_scale=1*np.ones(X.shape[1]))        + 1e-3**2 * WhiteKernel(noise_level=1.)


# In[15]:


from sklearn.gaussian_process import GaussianProcessRegressor


# In[12]:


gpr = GaussianProcessRegressor(kernel=kernel,
                              alpha=1e-5,
                              random_state=42)


# In[13]:


from fit import fit


# In[14]:


gpr.fit = fit


# In[8]:


GPs = gpr.fit(gpr, X, y)


# In[ ]:


GPs = GaussianProcessRegressor(kernel=kernel,
                              alpha=1e-5,
                              optimizer='hgdl',
                              random_state=42,
                              ).fit(
                                      X,y,
                              num_individuals=5
                                      )


# In[ ]:




