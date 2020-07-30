def main():
    from scipy.optimize import rosen, rosen_der, rosen_hess
    import numpy as np
    from hgdl import HGDL
    from time import sleep, perf_counter
    from sys import exit
    b = np.array([[-2, 2],[-3,3.]])
    import dask.distributed
    client = dask.distributed.Client()
    def newRosen(x):
        print('working on ',x)
        return rosen(x)
    #t = perf_counter()
    #print(HGDL(newRosen, rosen_der, None, b, client, num_individuals=10, max_epochs=2).get_final())
    #print('hgdl took ', perf_counter()-t, 'seconds')
    hgdl = HGDL(rosen, rosen_der, None, b, client, num_individuals=10, max_epochs=2)
    res = hgdl.get_final()
    print(res)

if __name__ == "__main__":
    main()
