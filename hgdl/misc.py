import numpy as np
import time

def out_of_bounds(x,bounds):
    for i in range(len(x)):
        if x[i] < bounds[i,0] or x[i] > bounds[i,1]:
            return True
    return False
def in_bounds(x, bounds):
    if (bounds[:,1]-x > 0).all() and (bounds[:,0] - x < 0).all():
        return True
    return False
def random_sample(N,k,bounds):
    sample = np.random.random((N, k))
    sample *= bounds[:,1] - bounds[:,0]
    sample += bounds[:,0]
    return sample

def random_population(bounds, n):
    return np.random.uniform(low = bounds[:,0], high = bounds[:,1], size = (n,len(bounds)))

def finish_up_tasks(tasks):
    for f in tasks:
        if f.status == 'cancelled':
            tasks.remove(f)
    while any(f.status == 'pending' for f in tasks):
        #print("finishing up last tasks...")
        time.sleep(0.1)
    return tasks

