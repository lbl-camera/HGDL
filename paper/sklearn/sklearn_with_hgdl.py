import numpy as np

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
def main():
    # declare that we should use my fitting method
    from fit import fit
    GaussianProcessRegressor.fit = fit
    # generate data exactly as in the example
    rng = np.random.RandomState(0)
    X = rng.uniform(0, 5, 20)[:, np.newaxis]
    y = 0.5 * np.sin(3 * X[:, 0]) + rng.normal(0, 0.5, X.shape[0])
    print('HGDL -----------------------------------------------------------')
    kernel1 = 1.0 * RBF(length_scale=100.0, length_scale_bounds=(1e-2, 1e3)) \
        + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
    kernel2 = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
        + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e+1))
    GPs = GaussianProcessRegressor(kernel=kernel2, alpha=0.0, optimizer='hgdl').fit(X, y)
    for i, gp in enumerate(GPs):
        print('gp - HGDL (',i+1,'): ', gp, '\nkernel:', gp.kernel_)
        print('theta:', gp.kernel_.theta, np.exp(gp.kernel_.theta))
        print('likelihood:', gp.log_marginal_likelihood_value_)

    thetas = np.array([gp.kernel_.theta for gp in GPs])
    print(thetas)
    np.save('data/hgdl_thetas', thetas)


if __name__ == "__main__":
    main()
