import warnings

import dask.distributed as distributed
import dask.multiprocessing
import numpy as np
from loguru import logger

from . import misc
from .global_methods.global_optimizer import run_global
from .local_methods.local_optimizer import run_local
from .meta_data import meta_data
from .optima import optima


class HGDL:
    """
    This is HGDL, a class for asynchronous HPC-capable optimization. \n
    H ... Hybrid \n
    G ... Global \n
    D ... Deflated \n
    L ... Local \n
    The algorithm places a number of walkers inside the domain 
    (the number is determined by the dask client), all of which perform
    a local optimization in a distributed way in parallel. 
    When the walkers have identified local optima, their 
    positions are communicated back to the host
    who removes the found optima by deflation, 
    and replaces the fittest walkers by a global 
    optimization step. From here the next epoch
    begins with distributed local optimizations of 
    the new walkers. The algorithm results in a 
    sorted list of unique optima (only if optima
    are of the form f'(x) = 0)
    The method `hgdl.optimize` instantly 
    returns a result object that can be 
    queried for a growing, sorted list of
    optima. If a Hessian is provided, 
    those optima are classified as minima, 
    maxima or saddle points.

    Parameters
    ----------
    func : Callable
        The function to be MINIMIZED. A callable that accepts an np.ndarray and 
        optional arguments, and returns a scalar.
    grad : Callable
        The gradient of the function to be MINIMIZED. A callable that accepts an
        np.ndarray and optional arguments, and returns a vector
        (np.ndarray) of shape (D), where D is the dimensionality of the space in
        which the
        optimization takes place.
    bounds : np.ndarray
        The bounds of the domain; an np.ndarray of shape (D x 2), where D is the
        dimensionality of the space in which the
        optimization takes place. Here D is the dimension of the input domain.
    hess : Callable, optional
        The Hessian of the function to be MINIMIZED. A callable that accepts an 
        np.ndarray and optional arguments, and returns a
        np.ndarray of shape (D x D). The default value is no-op.
    num_epochs : int, optional
        The number of epochs the algorithm runs through before being terminated.
        One epoch is the convergence of all local walkers,
        the deflation of the identified optima, and the global replacement of the 
        walkers. Note, the algorithm is running asynchronously, so a high number
        of epochs can be chosen without concerns, it will not affect the run time 
        to obtain the optima. Therefore, the default is
        100000.
    global_optimizer : Callable or str, optional
        The function (identified by a string or a Callable) that replaces the 
        fittest walkers after their local convergence.
        The possible options are `genetic` (default), `random` or a callable that 
        accepts an
        np.ndarray of shape (U x D) of positions, an np.ndarray of shape (U) of 
        function values,
        and np.ndarray of shape (D x 2) of bounds, and an integer specifying the
        number of offspring
        individuals that should be returned. The callable should return the 
        positions of the offspring
        individuals as an np.ndarray of shape (number_of_offspring x D).
    local_optimizer : Callable or str, optional
        The local optimizer that is used. The options are
        `dNewton` (default), `L-BFGS-B`, `BFGS`, `CG`, `Newton-CG`, `SLSQP`.
        The above methods have been tested, but most others should work. Visit 
        the `scipy.optimize.minimize` docs 
        (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
        for specifications and limitations of the local methods. The parameter 
        also accepts a callable of the form func(f,grad,hess,bounds,x0,*args), 
        and returns an object equal to the scipy.optimize.minimize methods.
    number_of_optima : int, optional
        The number of optima that will be stored in the optima list and deflated.
        The default is 1e6.
        After that number is reached, worse-performing optima will not be stored 
        or deflated.
    local_max_iter : int, optional
        The number of iterations before local optimizations are terminated. 
        The default is 1000.
        It can be lowered when second-order local optimizers are used.
    args : tuple, optional
        A tuple of arguments that will be communicated to the function, 
        the gradient, and the Hessian callables.
        Default = ().
    constraints : object, optional
        An optional n-tuple of constraint objects.
        The default is no constraints (). Constraints are defined following 
        scipy.optimize.NonlinearConstraint.

    Attributes
    ----------
    optima : object
        Contains the attribute optima.list in which the optima are stored.
        However, the method 'get_latest()' should be used to access the optima.


    """

    def __init__(self, func, grad, bounds,
                 hess=None, num_epochs=100000,
                 global_optimizer="genetic",
                 local_optimizer="L-BFGS-B",
                 number_of_optima=1000000,
                 local_max_iter=1000,
                 constraints=(),
                 args=()):
        bounds = np.asarray(bounds)
        self.dim = len(bounds)
        self.bounds = bounds
        self.func = func
        self.grad = grad
        if hess:
            self.hess = hess
        else:
            self.hess = self.hess_approx
        if bounds is not None and local_optimizer == "dNewton":
            warnings.warn("Warning: dNewton will not adhere to bounds. It is recommended to formulate your objective function such that it is defined on R^N by simple non-linear transformations.")
        if constraints:
            local_optimizer = "SLSQP"
            warnings.warn("Constraints provided, local optimizer changed to 'SLSQP'")

        self.constraints = constraints
        self.local_max_iter = local_max_iter
        self.num_epochs = num_epochs
        self.global_optimizer = global_optimizer
        self.local_optimizer = local_optimizer
        self.args = args
        self.optima = optima(self.dim, number_of_optima)
        logger.debug("HGDL successfully initiated {}")
        if callable(self.hess): logger.debug("Hessian was provided by the user: {}", self.hess)
        logger.debug("========================")

    ###########################################################################
    ###########################################################################
    ############USER FUNCTIONS#################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    def optimize(self, dask_client=None, x0=None, tolerance=1e-10):
        """
        Function to start the optimization. Note, this function will not 
        return anything. Use the method hgdl.HGDL.get_latest() 
        (non-blocking) or hgdl.HGDL.get_final() (blocking)
        to query results.

        Parameters
        ----------
        dask_client : distributed.client.Client, optional
            The client that will be used for the distibuted local 
            optimizations. The default is a local client.
        x0 : np.ndarray, optional
            An np.ndarray of shape (V x D) of points used as 
            starting positions. If V > number of walkers 
            (specified by the dask client) the array will be truncated.
            If V < number of walkers, random points will be appended.
            The default is None, meaning only random points will be used.
        tolerance : float, optional
            The tolerance used by the local optimizers. The default is 1e-6
        """
        client = self._init_dask_client(dask_client)
        self.tolerance = tolerance
        logger.debug(client)
        self.x0 = self._prepare_starting_positions(x0)
        logger.debug("HGDL starts with: {}", self.x0)
        self.meta_data = meta_data(self)
        self._run_epochs(client)

    ###########################################################################
    def get_client_info(self):
        """
        Function to receive info about the workers.
        """
        return self.workers

    ###########################################################################
    def get_latest(self):
        """
        Function to request the current result.
        No inputs
        """
        try:
            data, frames = self.transfer_data.get()
            self.optima = distributed.protocol.deserialize(data, frames)
            logger.debug("HGDL called get_latest() successfully")
        except Exception as err:
            self.optima = self.optima
            logger.error("HGDL get_latest failed due to {} \n optima list unchanged", str(err))
        optima_list = self.optima.list
        return optima_list

    ###########################################################################
    def get_final(self):
        """
        Function to request the final result.
        CAUTION: This function will block the main thread until
        the end of all epochs is reached.
        No inputs.
        """
        try:
            self.optima = self.main_future.result()
        except Exception as err:
            logger.error("HGDL get_final failed due to {}", str(err))
        optima_list = self.optima.list
        return optima_list

    ###########################################################################
    def cancel_tasks(self):
        """
        Function to cancel all tasks and therefore the execution.
        However, this function does not kill the client.
        """
        logger.debug("HGDL is cancelling all tasks...")
        res = self.get_latest()
        self.break_condition.set(True)
        self.client.cancel(self.main_future)
        logger.debug("Status of HGDL task: ", self.main_future.status)
        logger.debug("This leaves the client alive.")
        return res

    ###########################################################################
    def kill_client(self):
        """
        Function to cancel all tasks and kill the dask client, 
        and therefore the execution.
        If cancel_tasks() is called before, this will throw an error.
        """
        logger.debug("HGDL kill client initialized ...")
        res = self.get_latest()
        try:
            self.break_condition.set(True)
            self.client.cancel(self.main_future)
            self.client.close()
            logger.debug("HGDL kill client successful")
        except Exception as err:
            raise RuntimeError("HGDL kill failed") from err
        return res

    ###########################################################################
    ############USER FUNCTIONS END#############################################
    ###########################################################################
    def _prepare_starting_positions(self, x0):
        if x0 is not None and len(x0[0]) != self.dim:
            raise Exception("Wrong dimensionality of starting positions")
        elif x0 is None:
            x0 = misc.random_population(self.bounds, self.number_of_walkers)
        elif x0.ndim == 1:
            x0 = np.array([x0])

        if len(x0) < self.number_of_walkers:
            x0_aux = np.zeros((self.number_of_walkers, len(x0[0])))
            x0_aux[0:len(x0)] = x0
            x0_aux[len(x0):] = misc.random_population(self.bounds, self.number_of_walkers - len(x0))
            x0 = x0_aux
        elif len(x0) > self.number_of_walkers:
            x0 = x0[0:self.number_of_walkers]
        else:
            x0 = x0
        return x0

    ###########################################################################
    def _init_dask_client(self, dask_client):
        if dask_client is None:
            dask_client = dask.distributed.Client()
            logger.debug("No dask client provided to HGDL. Using the local client")
        else:
            logger.debug("dask client provided to HGDL")
        client = dask_client
        worker_info = list(client.scheduler_info()["workers"].keys())
        if not worker_info: raise Exception("No workers available")
        self.workers = {"host": worker_info[0],
                        "walkers": worker_info[1:]}
        logger.debug(f"Host {self.workers['host']} has {len(self.workers['walkers'])} workers.")
        self.number_of_walkers = len(self.workers["walkers"])
        return client

    ###########################################################################
    def _run_epochs(self, client):
        self.break_condition = distributed.Variable("break_condition", client)
        self.transfer_data = distributed.Variable("transfer_data", client)
        a = distributed.protocol.serialize(self.optima)
        self.transfer_data.set(a)
        self.break_condition.set(False)
        data = {"transfer data": self.transfer_data,
                "break condition": self.break_condition,
                "optima": self.optima, "metadata": self.meta_data}
        bf = client.scatter(data, workers=self.workers["host"])
        self.main_future = client.submit(hgdl, bf, workers=self.workers["host"])
        self.client = client

    ###########################################################################
    def hess_approx(self, x, *args):
        ##implements a first-order approximation
        len_x = len(x)
        hess = np.zeros((len_x, len_x))
        epsilon = 1e-6
        grad_x = self.grad(x, *args)
        for i in range(len_x):
            x_temp = np.array(x)
            x_temp[i] = x_temp[i] + epsilon
            hess[i, i:] = ((self.grad(x_temp, *args) - grad_x) / epsilon)[i:]
        return hess + hess.T - np.diag(np.diag(hess))


###########################################################################
###########################################################################
##################hgdl functions###########################################
###########################################################################
###########################################################################
def hgdl(data):
    metadata = data["metadata"]
    transfer_data = data["transfer data"]
    break_condition = data["break condition"]
    optima = data["optima"]
    logger.debug("HGDL computing epoch 1 of {}", metadata.num_epochs)
    res = run_local(metadata,optima,metadata.x0)
    logger.debug("filling in optima list for the first time.", flush = True)
    optima.fill_in_optima_list(res)
    logger.debug("optima list filled", flush = True)
    a = distributed.protocol.serialize(optima)
    transfer_data.set(a)

    logger.debug("HGDL first local optimization round done.", flush = True)
    for i in range(1, metadata.num_epochs):
        bc = break_condition.get()
        if bc is True:
            logger.debug(f"HGDL Epoch {i} was cancelled")
            break
        logger.debug(f"HGDL computing epoch {i + 1} of ", metadata.num_epochs)
        optima = run_hgdl_epoch(metadata, optima)
        a = distributed.protocol.serialize(optima)
        transfer_data.set(a)
    logger.debug("HGDL finished all epochs!")
    return optima


###########################################################################
def run_hgdl_epoch(metadata, optima):
    optima_list = optima.list
    n = min(len(optima_list),metadata.number_of_walkers)
    ind_pos = [entry["x"] for entry in optima_list]
    ind_fit = [entry["f(x)"] for entry in optima_list]

    global_res = run_global(\
            np.array(ind_pos[:n]),
            np.array(ind_fit[0:n]),
            metadata.bounds[0:metadata.dim], metadata.global_optimizer,n)
    x0 = np.zeros((n,metadata.dim))
    x0[:,0:metadata.dim] = np.array(global_res)
    res = run_local(metadata,optima,x0)
    op = optima.fill_in_optima_list(res)
    return optima
