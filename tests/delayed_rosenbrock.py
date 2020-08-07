def main():
    from scipy.optimize import rosen, rosen_der, rosen_hess
    import numpy as np
    from hgdl.hgdl import HGDL
    from time import sleep, perf_counter
    b = np.array([[-2, 2],[-3,3.]])
    def newRosen(x):
        print('working on',x.round(3))
        return rosen(x)

    hgdl = HGDL(newRosen, rosen_der, b)
    print('sleeping')
    sleep(10)
    print('waking up and getting result')
    res = hgdl.get_final()
    print(res)

if __name__ == "__main__":
    main()
