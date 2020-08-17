class generic(object):
    def run(self):
        from scipy.optimize import rosen, rosen_der, rosen_hess
        import numpy as np
        from hgdl.hgdl import HGDL
        from time import sleep, perf_counter
        b = np.array([[-2, 2],[-3,3.]])

        hgdl = HGDL(func=rosen, grad=rosen_der, bounds=b, local_method='scipy')
        res = hgdl.get_final()
        return res

def main():
    print('running from class itself', generic.run(None))
    print('running from class instance', generic().run())

if __name__ == "__main__":
    main()
