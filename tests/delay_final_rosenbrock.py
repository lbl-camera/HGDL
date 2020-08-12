def main():
    from scipy.optimize import rosen, rosen_der, rosen_hess
    import numpy as np
    from hgdl.hgdl import HGDL
    from time import sleep, perf_counter
    b = np.array([[-2, 2],[-3,3.]])
    print('this will create an hgdl object, sleep for 30'
            ' seconds, then get the final result.\n'
            ' if working correctly, it should print:\n'
            'sleeping, working on epoch, ..., waking up\n'
            'if broken it should print:\n'
            'sleeping, waking up, working on epoch,...\n'
            )
    hgdl = HGDL(rosen, rosen_der, b, verbose=True)
    print('sleeping')
    sleep(30)
    print('waking up and getting result')
    res = hgdl.get_final()
    print(res)

if __name__ == "__main__":
    main()
