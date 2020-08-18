import numpy as np
import time
import hgdl.misc as misc
import dask.distributed
from distributed import Client, get_client, secede, rejoin, protocol
import dask.distributed as distributed
from hgdl.local_methods.dNewton import DNewton

def run_local(func,grad,hess,bounds, radius,
        local_max_iter,global_max_iter,x_init,optima,args,verbose):
    break_condition = False
    x_init = np.array(x_init)
    x_defl,f_defl = optima.get_deflation_points(len(optima.list))
    counter = 0
    while break_condition is False or counter >= global_max_iter:
        counter += 1
        if verbose is True: print("    local replacemet step: ",counter)
        #walk walkers with DNewton
        #print("x_defl")
        #print(x_defl)
        x,f,grad_norm,eig,success = run_dNewton(func, grad,hess,bounds,
                radius,local_max_iter,x_init,args,x_defl)
        if verbose is True: print("    deflated Newton finished in step: ",counter)
        #print("result")
        #print(x)
        #print("optima list before fill in:")
        #print(optima.list)
        optima.fill_in_optima_list(x,f,grad_norm,eig,success)
        #print("optima list after fill in:")
        #print(optima.list)
        x_defl,f_defl = optima.get_deflation_points(len(optima.list))
        #print("therefore the new x_defl is")
        #print(x_defl)
        #print("============")
        #for i in range(len(optima.list["x"])):
        #    for j in range(len(optima.list["x"])):
        #        if i == j: continue
        #        if np.linalg.norm(optima.list["x"][i] - optima.list["x"][j]) < 1e-6:
        #            print(i,j,optima.list["x"][i],optima.list["x"][j])
        #            exit("doublicates in list")

        if len(np.where(success == False)[0]) > len(success)/2.0: break_condition = True
        return optima
    ###########################################################################
def run_dNewton(func,grad,hess,bounds,radius,local_max_iter,x_init,args,x_defl = []):
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
    try: 
        client = get_client()
        client_available = True
    except:
        client_available = False
    if client_available is True:
        tasks = []
        for i in range(number_of_walkers):
            tasks.append(client.submit(DNewton,func, grad,hess,\
            x_init[i],x_defl,bounds,radius,local_max_iter,args))
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
            x[i],f[i],grad_norm[i],eig[i],success[i] = DNewton(func, grad,hess,\
            x_init[i],x_defl,bounds,radius,local_max_iter,args)
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
                    input()
    else: exit("not clear if client is available")
    return x, f, grad_norm, eig, success

