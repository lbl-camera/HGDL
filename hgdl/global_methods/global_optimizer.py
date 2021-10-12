##global optimizer for hgdl
import numpy as np
import hgdl.misc as misc


def run_global(x,y,bounds,method,number_of_offspring):
    print("Global optimizer: ", method)
    if method == "genetic": return genetic_step(x,y,bounds,number_of_offspring)
    elif method == "gauss": return gauss_step(x,y,bounds,number_of_offspring)
    elif method =="random": return random_step(x,y,bounds,number_of_offspring)
    elif method is callable: return method(x,y,bounds,number_of_offspring)
    else: raise Exception("no global method specified")

def gauss_step(x, y, bounds,n):
    ####x is x where in bounds
    x_new = []
    y_new = []
    #bounds = np.array(bounds)
    for i in range(len(x)):
        if misc.in_bounds(x[i],bounds): 
            x_new.append(x[i])
            y_new.append(y[i])
    x = np.asarray(x_new)
    y = np.asarray(y_new)
    sorted_indices = np.argsort(y)
    y = y[sorted_indices]
    x = x[sorted_indices]
    x = np.array(x[:-int(len(x)/2)])
    y = np.array(y[:-int(len(x)/2)])
    cov = np.cov(x.T)
    mean= np.mean(x , axis = 0)
    offspring = np.random.multivariate_normal(mean, cov, size = n)
    for i in range(len(offspring)):
        if not misc.in_bounds(offspring[i],bounds): 
            offspring[i] = np.random.uniform(low = bounds[:,0],high = bounds[:,1], size = len(offspring[0]))
    return offspring

def random_step(x, y, bounds,n):
    offspring = np.random.uniform(low = bounds[:,0], high = bounds[:,1], size = (n, len(bounds)))
    return offspring

###########################################################################
def genetic_step(X, y, bounds, numChoose):
    """
    Input:
    X is the individuals - points on a surface
    y is the performance - f(X)
    Notes:
    the children can be outside of the bounds!
    """
    unfairness = 2.5
    wildness = 0.05
    N, k = X.shape
    # normalize the performances to (0,1)
    y -= np.amin(y)
    amax = np.amax(y)
    # if the distribution of performance has no width,
    #   give everyone an equal shot
    if np.isclose(amax,0.):
        p = np.ones(N)*1./N
    else:
        y /= np.amax(y)
        y *= -1.
        y -= np.amin(y)
        y += 1
        p = y/np.sum(y)
    #This chooses from the sample based on the power law,
    #allowing replacement means that the the individuals
    #can have multiple kids
    p = unfairness*np.power(p,unfairness-1)
    p /= np.sum(p)
    if np.isnan(p).any():
        raise Exception("got isnans in GeneticStep")
    moms = np.random.choice(N, size=numChoose, replace=True, p=p)
    dads = np.random.choice(N, size=numChoose, replace=True, p=p)
    # calculate a perturbation to the median
    #   of each individual's parents
    perturbation = np.random.normal(
            loc = 0.,
            scale = wildness*(bounds[:,1]-bounds[:,0]),
            size=(numChoose,k))
    # the children are the median of their parents plus a perturbation
    norm = p[moms]+p[dads]
    weights = (p[moms]/norm, p[dads]/norm)
    weighted_linear_sum = weights[0].reshape(-1,1)*X[moms] + weights[0].reshape(-1,1)*X[dads]
    children = weighted_linear_sum + perturbation
    oob = np.logical_not([misc.in_bounds(x,bounds) for x in children])
    children[oob] = misc.random_sample(np.sum(oob), k, bounds)
    print("=========================")
    print("Children in HGDL genetic alg.:", flush = True)
    print(children)
    print("=========================")
    return children

