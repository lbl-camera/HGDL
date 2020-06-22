import numpy as np
# This is my implementation of a genetic algorithm
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
    #  allowing replacement means that the the individuals
    #  can have multiple kids
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
            scale=wildness*(bounds[:,1]-bounds[:,0]),
            size=(numChoose,k))
    # the children are the median of their parents plus a perturbation
    norm = p[moms]+p[dads]
    weights = (p[moms]/norm, p[dads]/norm)
    weighted_linear_sum = weights[0].reshape(-1,1)*X[moms] + weights[0].reshape(-1,1)*X[dads]
    children = weighted_linear_sum + perturbation
    oob = np.logical_not([in_bounds(x,bounds) for x in children])
    children[oob] = random_sample(np.sum(oob), k, bounds)
    return children
