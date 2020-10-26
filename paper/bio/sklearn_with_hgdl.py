import numpy as np

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

def main():
    # declare that we should use my fitting method
    from fit import fit
    GaussianProcessRegressor.fit = fit
    from data_reader import data_reader
    data = data_reader(42)
    x, y = data.get_training()

    print('HGDL -----------------------------------------------------------')
    from sklearn.gaussian_process.kernels import WhiteKernel, Matern
    kernel = 1.**2 * Matern(length_scale=.01*np.ones(x.shape[1]),nu=2.0)\
            + 1.**2 * WhiteKernel(noise_level=1.)
    GPs = GaussianProcessRegressor(kernel=kernel,
                                  alpha=1e-5,
                                  optimizer='hgdl',
                                  random_state=42,
                                  ).fit(
                                          x,y,
                                  num_individuals=5
                                          )
    for i, gp in enumerate(GPs):
        print('gp - HGDL (',i+1,'): ', gp, '\nkernel:', gp.kernel_)
        print('theta:', gp.kernel_.theta, np.exp(gp.kernel_.theta))
        print('likelihood:', gp.log_marginal_likelihood_value_)

    key = str(np.random.randint(low=0,high=10000))
    with open('data/GPs'+key+'.pkl','wb') as file:
        import pickle
        pickle.dump(GPs, file)

    thetas = np.array([gp.kernel_.theta for gp in GPs])
    print(thetas)
    np.save('data/hgdl_thetas', thetas)

if __name__=="__main__":
    main()


