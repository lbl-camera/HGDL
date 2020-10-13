import numpy as np
import matplotlib.pyplot as plt

x0 = np.zeros(1)
alpha = 1
r = 1
deflation = lambda x: (np.exp(-alpha/(r**2-np.linalg.norm(x-x0)**2) + alpha/r**2) if np.linalg.norm(x-x0)<r else 0)
x = np.arange(-1.5, 1.5, 1e-3)
y = np.array([deflation(z) for z in x])

plt.plot(x, y)
plt.title('bump function (x0=0)')

plt.savefig('bump.png')


