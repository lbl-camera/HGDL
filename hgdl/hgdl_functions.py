import numpy as np
import time
import hgdl.misc as misc
import hgdl.local as local
import hgdl.glob as glob
import dask.distributed
from psutil import cpu_count
from distributed import Client, get_client, secede, rejoin, protocol
import dask.distributed as distributed

def hgdl(transfer_data,init_optima_list, 
        func, grad,hess, bounds,
        maxEpochs,radius, local_max_iter,global_max_iter,
        number_of_walkers, args, verbose):
    if verbose is True: print("    Starting ",maxEpochs," epochs.")
    for i in range(maxEpochs):
        print("Computing epoch ",i," of ",maxEpochs)
        optima_list = run_hgdl_epoch(func,grad,hess,bounds,init_optima_list,
                radius,local_max_iter,global_max_iter,
                number_of_walkers,args,verbose)
        if verbose is True: print("    Epoch ",i," finished")
        a = distributed.protocol.serialize(optima_list)
        transfer_data.set(a)
        init_optima_list = dict(optima_list)
    time.sleep(0.1)
    return optima_list

def run_hgdl_epoch(func,grad,hess,bounds,optima_list,radius,
        local_max_iter,global_max_iter,number_of_walkers,args,verbose):
    """
    an epoch is one local run and one global run,
    where one local run are several convergence runs of all workers from
    the x_init point
    """
    n = len(optima_list["x"])
    nn = min(n,number_of_walkers)
    if verbose is True: print("    global step started")
    #print("global")
    x0 = glob.genetic_step(\
            np.array(optima_list["x"][0:nn,:]),
            np.array(optima_list["func evals"][0:nn]),
            bounds, number_of_walkers,verbose)
    #print("local")
    if verbose is True: print("    global step finished")
    if verbose is True: print("    local step started")
    optima_list = run_local(func,grad,hess,bounds,radius,
            local_max_iter,global_max_iter,
            x0, np.array(optima_list["x"]),optima_list,args,verbose)
    if verbose is True: print("    local step finished")
    return optima_list

def run_local(func,grad,hess,bounds, radius,
        local_max_iter,global_max_iter,x_init, x_defl,optima_list,args,verbose):
    break_condition = False
    x_init = np.array(x_init)
    x_defl = np.array(x_defl)
    counter = 0
    while break_condition is False or counter >= global_max_iter:
        counter += 1
        if verbose is True: print("    local replacemet step: ",counter)
        #walk walkers with DNewton
        x,f,grad_norm,eig,success = run_dNewton(func, grad,hess,bounds,
                radius,local_max_iter,x_init,args,x_defl)
        if verbose is True: print("    Deflated Newton finished in step: ",counter)
        optima_list = fill_in_optima_list(optima_list, 1e-6,x,f,grad_norm,eig,success)
        x_defl = np.array(optima_list["x"])
        if len(np.where(success == False)[0]) > len(success)/2.0: break_condition = True
    return optima_list
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
    dim = len(x_init[0])
    number_of_walkers = len(x_init)
    client = get_client()
    if client is False:
        #this is in case we don't want distributed computing with DASK
        x = np.empty((number_of_walkers, dim))
        f = np.empty((number_of_walkers))
        grad_norm = np.empty((number_of_walkers))
        eig = np.empty((number_of_walkers,dim))
        success = np.empty((number_of_walkers))
        for i in range(number_of_walkers):
            #print("newton for ", i)
            x[i],f[i],grad_norm[i],eig[i],success[i] =\
            local.DNewton(func, grad,hess,\
            x_init[i],x_defl,bounds,1e-6,local_max_iter,args)
        return x, f, grad_norm, eig, success
    else:
        #this is in case there is a DASK client and we want distributed computing
        tasks = []
        for i in range(number_of_walkers):
            tasks.append(client.submit(local.DNewton,func, grad,hess,\
            x_init[i],x_defl,bounds,1e-6,local_max_iter,args))
        tasks = finish_up_tasks(tasks)
        #while any(f.status == 'pending' for f in tasks):
        #    time.sleep(0.1)
        #if any(f.status == 'cancelled' for f in tasks):
        #    print("cancelled tasks")
        #    tasks = []
        #secede()
        client.gather(tasks)
        #rejoin()
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
                if np.linalg.norm(np.subtract(x[i],x[j])) < 2.0 * radius: success[i] = False; break
            for j in range(len(x_defl)):
                if np.linalg.norm(np.subtract(x[i],x_defl[j])) < 1e-5 and success[i] == True:
                    #print("CAUTION: Newton converged to deflated position")
                    success[i] = False
                    #print(x[i],x_defl[j])
                    #input()
        return x, f, grad_norm, eig, success
###########################################################################
def fill_in_optima_list(optima_list,local_tol,x,f,grad_norm,eig, success):
    clean_indices = np.where(success == True)
    clean_x = x[clean_indices]
    clean_f = f[clean_indices]
    clean_grad_norm = grad_norm[clean_indices]
    clean_eig = eig[clean_indices]
    classifier = []
    #print("clean x:")
    #print(clean_x)
    #print("optima list before stacking")
    #print(optima_list["x"])
    for i in range(len(x)):
        if grad_norm[i] > local_tol: classifier.append("degenerate")
        elif len(np.where(eig[i] > 0.0)[0]) == len(eig[i]): classifier.append("minimum")
        elif len(np.where(eig[i] < 0.0)[0]) == len(eig[i]): classifier.append("maximum")
        elif len(np.where(eig[i] == 0.0)[0])  > 0: classifier.append("zero curvature")
        elif len(np.where(eig[i] < 0.0)[0])  < len(eig[i]): classifier.append("saddle point")
        else: classifier.append("ERROR")

    optima_list = {"x":       np.vstack([optima_list["x"],clean_x]), \
                    "func evals":   np.append(optima_list["func evals"],clean_f), \
                    "classifier":   optima_list["classifier"] + classifier, \
                    "eigen values": np.vstack([optima_list["eigen values"],clean_eig]),\
                    "gradient norm":np.append(optima_list["gradient norm"],clean_grad_norm)}
    #print("optima list after stacking")
    #print(optima_list["x"])
    #input()

    sort_indices = np.argsort(optima_list["func evals"])
    optima_list["x"] = optima_list["x"][sort_indices]
    optima_list["func evals"] = optima_list["func evals"][sort_indices]
    optima_list["classifier"] = [optima_list["classifier"][i] for i in sort_indices]
    optima_list["eigen values"] = optima_list["eigen values"][sort_indices]
    optima_list["gradient norm"] = optima_list["gradient norm"][sort_indices]
    return optima_list
###########################################################################
def finish_up_tasks(tasks):
    for f in tasks:
        if f.status == 'cancelled':
            tasks.remove(f)
    while any(f.status == 'pending' for f in tasks):
        #print("finishing up last tasks...")
        time.sleep(0.1)
    return tasks

