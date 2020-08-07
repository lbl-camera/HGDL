def main():
    from scipy.optimize import rosen, rosen_der, rosen_hess
    import numpy as np
    from hgdl.hgdl import HGDL
    from time import sleep, perf_counter
    b = np.array([[-2, 2],[-3,3.]])
    hgdl = HGDL(rosen, rosen_der, b, local_method='scipy')
    res = hgdl.get_final()
    print(res)

if __name__ == "__main__":
    main()
