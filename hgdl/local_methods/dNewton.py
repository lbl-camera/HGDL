import numpy as np
import hgdl.misc as misc
import hgdl.deflation.bump_function as defl

def DNewton(func, grad, hess, x0,x_defl,bounds,radius,max_iter = 20, args = ()):
    #w = 0 #0.1 #np.random.rand() * 10.0
    #print("DNewton from point:", x0," started, waits: ", w)
    #time.sleep(w)
    e = np.inf
    success = True
    counter = 0
    tol = 1e-6
    x = np.array(x0)
    #print("==================")
    #print(x_defl)
    #print("-------")
    while e > tol:
        counter += 1
        if counter >= max_iter or misc.out_of_bounds(x,bounds):
            #print("DNewton from point:", x0," not converged ",w, " ", counter, misc.out_of_bounds(x,bounds))
            return x,func(x, *args),e,np.linalg.eig(hessian)[0],False
        d = defl.deflation_function(x,x_defl,radius)
        dg = defl.deflation_function_gradient(x,x_defl,radius)
        gradient = grad(x, *args)
        hessian = hess(x, *args)
        e = np.linalg.norm(d*gradient)
        try:
            gamma = np.linalg.solve(hessian + (np.outer(gradient,dg)/d),-gradient)
        except Exception as error: 
            print("solve in dNewton crashed because: ",str(error)," starting least squares")
            gamma,a,b,c = np.linalg.lstsq(hessian + (np.outer(gradient,dg)/d),-gradient)
        x += gamma
        #print("current position: ",x,"epsilon: ",e, gamma, "d*grad: ",d,gradient, "hess ",hessian)
    #print("DNewton from point:", x0," converged to ",x, " with ",e," after waiting: ",w)
    #input()
    return x,func(x, *args),e,np.linalg.eig(hessian)[0], success
