import numpy as np
import time
import hgdl.misc as misc
import dask.distributed
from distributed import Client, get_client, secede, rejoin, protocol
import dask.distributed as distributed
from hgdl.local_methods.dNewton import DNewton

def run_local(d,optima,x0):
    x_init = np.array(x0)
    x_defl,f_defl = optima.get_deflation_points(len(optima.list))
    x,f,grad_norm,eig,success = run_local_optimizer(d,x0,x_defl)
    optima.fill_in_optima_list(x,f,grad_norm,eig,success)
    return optima
    ###########################################################################
def run_local_optimizer(d,x0,x_defl = []):
    """
    this function runs a deflated Newton for
    all the walkers.
    The loop below goes over every walker
    input:
        2d numpy array of initial positions
        2d numpy array of positions of deflations (optional, default = [])
    return:
        optima_locations, func values, gradient norms, eigenvalues, success(bool)
    """

    import hgdl.local_methods.local_optimizer as local
    dim = d.dim
    number_of_walkers = d.number_of_walkers
    number_of_workers = len(d.workers["walkers"])
    local_opt = DNewton
    if len(x0) < number_of_walkers: 
        x0 = np.row_stack([x0,misc.random_population(d.bounds,number_of_walkers - len(x0))])
    try: 
        client = get_client()
        client_available = True
    except:
        client_available = False
    if client_available is True:
        tasks = []
        bf = client.scatter(d)
        for i in range(min(len(x0),number_of_walkers)):
            worker = d.workers["walkers"][(int(i - ((i // number_of_workers)*number_of_workers)))]
            data = {"d":bf,"x0":x0[i],"x_defl":x_defl}
            tasks.append(client.submit(local_opt,data,workers = worker))
        results = client.gather(tasks)
        number_of_walkers = len(tasks)
        x = np.empty((number_of_walkers, dim))
        f = np.empty((number_of_walkers))
        grad_norm = np.empty((number_of_walkers))
        eig = np.empty((number_of_walkers,dim))
        success = np.empty((number_of_walkers))
        #gather results and kick out optima that are too close:
        for i in range(len(tasks)):
            x[i],f[i],grad_norm[i],eig[i],success[i] = results[i]
            #print("x: ",x[i],f[i])
            for j in range(i):
                #exchange for function def too_close():
                if np.linalg.norm(np.subtract(x[i],x[j])) < 2.0 * d.radius and success[j] == True:
                    success[j] = False; break
            for j in range(len(x_defl)):
                if np.linalg.norm(np.subtract(x[i],x_defl[j])) < 2.0 * d.radius\
                and grad_norm[i] < 1e-6:
                    print("CAUTION: Newton converged to deflated position")
                    success[i] = False
                    print(x[i],x_defl[j])
                    print(grad_norm[i])
    elif client_available is False:
        x = np.empty((number_of_walkers, dim))
        f = np.empty((number_of_walkers))
        grad_norm = np.empty((number_of_walkers))
        eig = np.empty((number_of_walkers,dim))
        success = np.empty((number_of_walkers))
        for i in range(d.number_of_walkers):
            data = {"d":d,"x0":x0[i],"x_defl":x_defl}
            x[i],f[i],grad_norm[i],eig[i],success[i] = DNewton(data)
            for j in range(i):
                #exchange for function def too_close():
                if np.linalg.norm(np.subtract(x[i],x[j])) < 2.0 * d.radius and success[j] == True:
                    success[j] = False; break
            for j in range(len(x_defl)):
                if np.linalg.norm(np.subtract(x[i],x_defl[j])) < 2.0 * d.radius \
                and success[i] == True and grad_norm[i] < 1e-6:
                    print("CAUTION: Newton converged to deflated position")
                    success[i] = False
                    print(x[i],x_defl[j])
                    print(grad_norm[i])
    else: raise ValueError("not clear if client is available")
    #print("++++++++++++++++++++++++++++++=")
    #print(x)
    #print(success)
    #print("++++++++++++++++++++++++++++++=")
    return x, f, grad_norm, eig, success



