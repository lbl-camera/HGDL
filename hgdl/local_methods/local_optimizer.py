import numpy as np
import time
import hgdl.misc as misc
import dask.distributed
from distributed import Client, get_client, secede, rejoin, protocol
import dask.distributed as distributed
from hgdl.local_methods.dNewton import DNewton

def run_local(d):
    #break_condition = False
    x_init = np.array(d["x0"])
    optima = d["optima"]
    x_defl,f_defl = optima.get_deflation_points(len(optima.list))
    x,f,grad_norm,eig,success = run_local_optimizer(d,x_defl)
    optima.fill_in_optima_list(x,f,grad_norm,eig,success)
    return optima
    ###########################################################################
#def run_local_optimizer(func,grad,hess,bounds,radius,
#        local_max_iter,local_method,x_init,args,x_defl = []):
def run_local_optimizer(d,x_defl = []):
    func = d["func"]
    grad = d["grad"]
    hess = d["hess"]
    bounds = d["bounds"]
    radius = d["radius"]
    local_max_iter = d["local max iter"]
    local_method = d["local optimizer"]
    x_init = d["x0"]
    args = d["args"]
    print(x_init)
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
    dim = len(x_init[0])
    number_of_walkers = len(x_init)
    local_opt = DNewton
    try: 
        client = get_client()
        client_available = True
    except:
        client_available = False
    if client_available is True:
        tasks = []
        worker_info = client.scheduler_info()["workers"]
        me = distributed.get_worker().address
        del worker_info[me]
        number_of_workers = len(worker_info)
        worker_set = list(worker_info.keys())
        for i in range(number_of_walkers):
            worker_index = (int(i - ((i // number_of_workers)*number_of_workers)))
            worker = worker_set[worker_index]
            #print("worker: ", worker, " at index: ", worker_index)
            data = {"func":func,"grad":grad,"hess":hess,"x0":x_init[i],
                    "x_defl":x_defl,"bounds":bounds,"radius":radius,
                    "local max iter":local_max_iter,"args":args,"result":None}
            big_future = client.scatter(data, workers = worker)
            tasks.append(client.submit(local_opt,big_future,workers = worker))
        del big_future
        tasks = misc.finish_up_tasks(tasks)
        client.gather(tasks)
        number_of_walkers = len(tasks)
        x = np.empty((number_of_walkers, dim))
        f = np.empty((number_of_walkers))
        grad_norm = np.empty((number_of_walkers))
        eig = np.empty((number_of_walkers,dim))
        success = np.empty((number_of_walkers))
        #gather results and kick out optima that are too close:
        for i in range(len(tasks)):
            x[i],f[i],grad_norm[i],eig[i],success[i] = tasks[i].result()
            for j in range(i):
                #exchange for function def too_close():
                if np.linalg.norm(np.subtract(x[i],x[j])) < 2.0 * radius and success[j] == True:
                    success[j] = False; break
            for j in range(len(x_defl)):
                if np.linalg.norm(np.subtract(x[i],x_defl[j])) < 2.0 * radius\
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
        for i in range(number_of_walkers):
            data = {"func":func,"grad":grad,"hess":hess,"x0":x_init[i],
                    "x_defl":x_defl,"bounds":bounds,"radius":radius,
                    "local max iter":local_max_iter,"args":args}
            x[i],f[i],grad_norm[i],eig[i],success[i] = DNewton(data)
            for j in range(i):
                #exchange for function def too_close():
                if np.linalg.norm(np.subtract(x[i],x[j])) < 2.0 * radius and success[j] == True:
                    success[j] = False; break
            for j in range(len(x_defl)):
                if np.linalg.norm(np.subtract(x[i],x_defl[j])) < 2.0 * radius \
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



