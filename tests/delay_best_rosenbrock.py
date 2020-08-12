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
    hgdl = HGDL(rosen, rosen_der, b, verbose=True)
    print('sleeping')
    sleep(3)
    print('waking up and getting best result')
    print(hgdl.get_best())
    print('sleeping')
    sleep(5)
    print('waking up and getting final result')
    res = hgdl.get_final()
    print(res)

if __name__ == "__main__":
    main()
