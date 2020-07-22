def main():
    from scipy.optimize import rosen, rosen_der, rosen_hess
    import numpy as np
    from hgdl import HGDL
    from time import sleep
    from sys import exit
    b = np.array([[-2, 2],[-3,3.]])
    import dask.distributed
    client = dask.distributed.Client()

    hgdl = HGDL(rosen, rosen_der, rosen_hess, b, client, num_individuals=20, max_epochs=10)
    print('got hgdl')
    # local_kwargs={'options':{'disp':True}} 
    #import asyncio
    #for i in range(2):
    #    print(hgdl.get_best())

    res = hgdl.get_final()

#    print(res['minima_x'].shape, res['minima_x'])
#    print(np.linalg.norm(res['minima_x'][0]-res['minima_x'][1])**2)

if __name__ == "__main__":
    main()
