def main():
    from scipy.optimize import rosen, rosen_der, rosen_hess
    import numpy as np
    from hgdl.hgdl import HGDL
    from time import sleep, perf_counter
    b = np.array([[-2, 2],[-3,3.]])
    print('this will create an hgdl object, sleep for 3'
            ' seconds, get the best result, sleep for 3 seconds,'
            'then get the final result.\n'
            'working on the epochs should happend even during sleeping\n'
            )
    #hgdl = HGDL(rosen, rosen_der, b, verbose=True)
    a = HGDL(rosen, rosen_der, rosen_hess,[[-2,2],[-2,2]], radius = 0.1, maxEpochs = 10, verbose = False)
    #a.optimize(dask_client = distributed.Client())
    a.optimize(dask_client = True)
    #a.optimize(dask_client = False)
    #res = a.optima_list
    #print(res)


    #print(a.optima_list)
    print("main thread submitted HGDL and will now sleep for 10 seconds")
    sleep(10)
    print("main thread asks for 10 best solutions:")
    print(a.get_latest(10))
    print("main sleeps for another 10 seconds")
    sleep(10)
    print("main thread kills optimization")
    res = a.kill()
    print("hgdl was killed but I am waiting 2s")
    sleep(2)
    print("")
    print("")
    print("")
    print("")
    print(res)

if __name__ == "__main__":
    main()
