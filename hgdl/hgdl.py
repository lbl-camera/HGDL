import asyncio
import copy
import threading
import uuid
import warnings

import dask.distributed as distributed
import numpy as np
from loguru import logger

from . import misc
from .global_methods.global_optimizer import run_global
from .local_methods.local_optimizer import collect_results, local_method, stop_run
from .optima import optima
from .problem import Problem, hessian_or_approximation


class HGDL:
    """
    This is HGDL, a class for asynchronous HPC-capable optimization. \n
    H ... Hybrid \n
    G ... Global \n
    D ... Deflated \n
    L ... Local \n
    The algorithm keeps one walker per dask worker thread
    running a local optimization, all in parallel. Whenever a
    walker finishes, the point it found is added to the list
    and removed by deflation, and a new walker is started
    from a global optimization step, so a slow walker never
    holds up the others. The result is a growing list of
    unique points, sorted by function value (only points with
    f'(x) = 0, or with a vanishing gradient of the Lagrangian
    under constraints, can be found).
    The walkers are coordinated by a background thread in the
    calling process, so `optimize()` returns instantly and
    the main thread stays free; the list can be queried at
    any time while it grows.
    What is searched for is set by `mode`: by default
    (`mode="minimization"`) only minima; with
    `mode="stationary_points"` all non-degenerate stationary
    points, which are then classified as minima, maxima or
    saddle points.

    Parameters
    ----------
    func : Callable
        The objective function f. A callable that accepts an np.ndarray of
        shape (D) and optional arguments, and returns a scalar. Its minima
        (`mode="minimization"`) or stationary points
        (`mode="stationary_points"`) are sought. It is sent to the dask
        workers, so it must be serializable by dask (cloudpickle).
    grad : Callable
        The gradient of `func`. A callable that accepts an np.ndarray and
        optional arguments, and returns a vector (np.ndarray) of shape (D),
        where D is the dimensionality of the space in which the optimization
        takes place.
    bounds : np.ndarray
        The bounds of the domain; an np.ndarray of shape (D x 2), where D is the
        dimensionality of the space in which the optimization takes place.
        Starting points and global replacements are drawn inside the bounds.
        scipy optimizers that support bounds respect them; `dNewton` only
        projects its iterates onto them.
    hess : Callable, optional
        The Hessian of `func`. A callable that accepts an np.ndarray and
        optional arguments, and returns a np.ndarray of shape (D x D).
        It is used to accept and classify results and to size their
        deflation radius, and by `dNewton` for its steps. The default is a
        finite-difference approximation computed from `grad`.
    num_epochs : int, optional
        The length of the run: it ends after `num_epochs` times the number of
        walkers (worker threads when `optimize()` is called) local
        optimizations have finished; walkers still running then are
        stopped. There is no barrier between epochs: every walker is replaced
        as soon as it finishes. The algorithm runs asynchronously:
        `get_latest()` returns the points found so far at any time, and
        `cancel_tasks()` stops the run, so a high number of epochs can be
        chosen without concerns; only `get_final()` waits for all of them.
        The default is 100000.
    global_optimizer : Callable or str, optional
        The method that generates the starting points of new walkers, one
        batch (one per walker) at a time, seeded with the best points found so
        far.
        The possible options are `genetic` (default), `random` or a callable that
        accepts an np.ndarray of shape (U x D) of positions, an np.ndarray of
        shape (U) of function values, an np.ndarray of shape (D x 2) of bounds,
        and an integer specifying the number of offspring individuals that
        should be returned. The callable should return the positions of the
        offspring individuals as an np.ndarray of shape (number_of_offspring x D).
    local_optimizer : Callable or str, optional
        The local optimizer that is used. The options are
        `L-BFGS-B` (default), `dNewton`, `BFGS`, `CG`, `Newton-CG`, `SLSQP`.
        The above methods have been tested, but most others should work. Visit
        the `scipy.optimize.minimize` docs
        (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
        for specifications and limitations of the local methods. The parameter
        also accepts a callable of the form func(f,grad,hess,bounds,x0,*args)
        that returns a dict-like result with the keys `x`, `fun` and `jac`,
        like scipy.optimize.minimize.
        `dNewton` is HGDL's own Newton method; how it steps depends on `mode`.
        Whatever the optimizer, a result is only kept if its deflated gradient
        is below 1e-6 in norm and the true Hessian there has the curvature
        `mode` asks for.
    number_of_optima : int, optional
        The maximum number of points returned by `get_latest()` and
        `get_final()`: those with the lowest function values. The default is 1e6.
        Internally every point found is kept and stays deflated, so that it
        cannot be found again, whatever this limit is.
    local_max_iter : int, optional
        The maximum number of iterations of each local optimization; passed to
        scipy optimizers as `options["maxiter"]` and used by `dNewton`.
        The default (None) keeps each scipy method's own default and lets
        `dNewton` run at most 1000 iterations.
        It can be lowered when second-order local optimizers are used.
    constraints : object, optional
        An optional n-tuple of constraint objects.
        The default is no constraints (). Constraints are defined following
        scipy.optimize.minimize (`NonlinearConstraint`, `LinearConstraint` or
        dicts). Providing constraints changes the local optimizer to `SLSQP`
        (with a warning). A result is then accepted as a minimum if it satisfies
        the KKT conditions: it is feasible, the gradient of the Lagrangian is
        below 1e-6, and the Lagrangian is strictly convex along the active
        constraints. Optima on an active constraint are found and deflated
        like interior ones. The bounds are not treated as constraints.
        Constraints cannot be combined with `mode="stationary_points"`.
    args : tuple, optional
        A tuple of arguments that will be communicated to the function,
        the gradient, and the Hessian callables.
        Default = ().
    local_time_limit : float, optional
        Seconds after which a walker stops its local optimization; its result
        is then judged like any other and usually rejected, and a new walker
        takes its place. It is checked between iterations, so a single
        evaluation of `func`, `grad` or `hess` is never interrupted, and it
        does not apply to a callable `local_optimizer`. The default (None) is
        no limit.
    mode : str, optional
        What the walkers search for. `minimization` (default) accepts only
        strict minima (smallest Hessian eigenvalue > 1e-6); any local optimizer
        can be used, and `dNewton` runs as a saddle-free Newton method (Hessian
        eigenvalues replaced by their absolute values) so that it cannot
        converge to maxima or saddle points.
        `stationary_points` accepts every non-degenerate stationary point
        (smallest absolute Hessian eigenvalue > 1e-6), i.e. minima, maxima and
        saddle points, deflates all of them and classifies them. It requires
        `dNewton` as plain Newton method; any other `local_optimizer` is
        replaced by `dNewton` with a warning.

    Attributes
    ----------
    optima : object
        Contains the attribute optima.list in which the points are stored.
        However, the method 'get_latest()' should be used to access them.

    Notes
    -----
    Each accepted point is deflated with a bump function of radius
    1 / (smallest absolute Hessian eigenvalue), capped at 10% of the
    diagonal of the domain. At a stationary point the principal curvatures
    of the graph of f are exactly the Hessian eigenvalues, so this radius is
    the largest principal radius of curvature (the osculating circle in the
    flattest direction). Like any curvature of the graph it depends on the
    scale of f; it is meant as an estimate of the extent of the optimum.

    """

    def __init__(self, func, grad, bounds,
                 hess=None, num_epochs=100000,
                 global_optimizer="genetic",
                 local_optimizer="L-BFGS-B",
                 number_of_optima=1000000,
                 local_max_iter=None,
                 constraints=(),
                 args=(),
                 mode="minimization",
                 local_time_limit=None):
        bounds = np.asarray(bounds)
        self.dim = len(bounds)
        self.bounds = bounds
        self.func = func
        self.grad = grad
        self.hess = hessian_or_approximation(hess, grad)
        if mode not in ("minimization", "stationary_points"):
            raise ValueError(f"mode must be 'minimization' or 'stationary_points', got {mode!r}")
        if mode == "stationary_points":
            if constraints:
                raise ValueError("Constraints require the SLSQP minimizer and cannot be combined "
                                 "with mode='stationary_points'.")
            if local_optimizer != "dNewton":
                warnings.warn(f"mode='stationary_points' requires dNewton; local optimizer changed "
                              f"from {local_optimizer!r} to 'dNewton'")
                local_optimizer = "dNewton"
        if local_optimizer == "dNewton":
            warnings.warn("Warning: dNewton will not adhere to bounds. It is recommended to formulate your objective function such that it is defined on R^N by simple non-linear transformations.")
        if constraints:
            local_optimizer = "SLSQP"
            warnings.warn("Constraints provided, local optimizer changed to 'SLSQP'")

        self.constraints = constraints
        self.local_max_iter = local_max_iter
        self.local_time_limit = local_time_limit
        self.num_epochs = num_epochs
        self.global_optimizer = global_optimizer
        self.local_optimizer = local_optimizer
        self.args = args
        self.mode = mode
        self.optima = optima(self.dim, number_of_optima)

        self.client = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._futures = set()
        self._run_id = None
        self._error = None
        logger.debug("HGDL successfully initiated")
        if hess: logger.debug("Hessian was provided by the user: {}", hess)
        logger.debug("========================")

    ###########################################################################
    ###########################################################################
    ############USER FUNCTIONS#################################################
    ###########################################################################
    ###########################################################################
    ###########################################################################
    def optimize(self, dask_client=None, x0=None, tolerance=1e-10):
        """
        Function to start the optimization. This function returns
        immediately; the walkers are coordinated by a background thread of
        the calling process and run on the dask workers.
        Use the method hgdl.HGDL.get_latest() (non-blocking) or
        hgdl.HGDL.get_final() (blocking) to query results.

        Parameters
        ----------
        dask_client : distributed.client.Client, optional
            The client that will be used for the distributed local
            optimizations. Every worker thread runs one walker at a time.
            The default is a local client.
        x0 : np.ndarray, optional
            An np.ndarray of shape (V x D) of points used as
            starting positions. If V > number of walkers
            (specified by the dask client) the array will be truncated.
            If V < number of walkers, random points will be appended.
            The default is None, meaning only random points will be used.
        tolerance : float, optional
            The convergence tolerance of the local optimizers (scipy's `tol`;
            the step and gradient tolerance of `dNewton`). The default is 1e-10.
        """
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("HGDL is already running; call cancel_tasks() first.")
        self.client = self._init_dask_client(dask_client)
        self.tolerance = tolerance
        logger.debug(self.client)
        self.x0 = self._prepare_starting_positions(x0)
        logger.debug("HGDL starts with: {}", self.x0)

        # unique, so that stopping this run does not stop other runs sharing the cluster
        self._run_id = uuid.uuid4().hex
        problem = Problem(self.func, self.grad, self.hess, self.bounds, self.args,
                          self.local_optimizer, self.local_max_iter, self.tolerance,
                          self.constraints, self.mode, self.local_time_limit, self._run_id)
        # sent to every worker once; walker tasks only carry a reference to it
        self._problem = self.client.scatter(problem, broadcast=True, hash=False)

        self._stop.clear()
        self._error = None
        self._thread = threading.Thread(target=self._run_walkers, name="hgdl-coordinator", daemon=True)
        self._thread.start()

    ###########################################################################
    def get_client_info(self):
        """
        Function to receive info about the workers.
        """
        return self.workers

    ###########################################################################
    def get_latest(self):
        """
        Function to request the current result. Non-blocking.
        No inputs.

        Returns
        -------
        optima list : list of dicts, sorted by f(x) ascending
        Each entry is a dict with the keys `x` (the point), `f(x)`,
        `classifier` (`minimum`, `maximum`, `saddle point`, `zero curvature`
        or `degenerate`), `Hessian eigvals`, `df/dx`, `|df/dx|` and `radius`
        (of its deflation).
        With constraints, `df/dx` is the gradient of the Lagrangian and
        `Hessian eigvals` are the eigenvalues of the Lagrangian's Hessian along
        the active constraints (fewer than the dimension when constraints are
        active, none when they fix the point).
        """
        with self._lock:
            return copy.deepcopy(self.optima.list[:self.optima.max_optima])

    ###########################################################################
    def get_final(self):
        """
        Function to request the final result.
        CAUTION: This function will block the main thread until
        the run has finished (see `num_epochs`).
        If the optimization failed (for instance because the objective
        function raised), that exception is raised here.
        No inputs.

        Returns
        -------
        optima list : list of dicts, sorted by f(x) ascending; see get_latest()
        """
        if self._thread is not None:
            self._thread.join()
        if self._error is not None:
            raise self._error
        return self.get_latest()

    ###########################################################################
    def cancel_tasks(self):
        """
        Function to cancel all tasks and therefore the execution.
        No new walker is started, walkers that have not started yet are
        cancelled, and running walkers stop at their next iteration; a
        running evaluation of the objective is not interrupted. The client
        stays alive.

        Returns
        -------
        optima list : the latest list of dicts, sorted by f(x)
        """
        logger.debug("HGDL is cancelling all tasks...")
        res = self.get_latest()
        self._stop.set()
        self._stop_walkers()
        logger.debug("This leaves the client alive.")
        return res

    ###########################################################################
    def kill_client(self):
        """
        Function to cancel all tasks and close the dask client,
        and therefore the execution.

        Returns
        -------
        optima list : the latest list of dicts, sorted by f(x)
        """
        logger.debug("HGDL kill client initialized ...")
        res = self.cancel_tasks()
        try:
            self.client.close()
            logger.debug("HGDL kill client successful")
        except Exception as err:
            raise RuntimeError("HGDL kill failed") from err
        return res

    ###########################################################################
    ############USER FUNCTIONS END#############################################
    ###########################################################################
    def _prepare_starting_positions(self, x0):
        if x0 is None:
            x0 = misc.random_population(self.bounds, self.number_of_walkers)
        x0 = np.array(x0, dtype=float, ndmin=2)
        if x0.shape[1] != self.dim:
            raise Exception("Wrong dimensionality of starting positions")

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
            dask_client = distributed.Client()
            logger.debug("No dask client provided to HGDL. Using the local client")
        else:
            logger.debug("dask client provided to HGDL")
        self._count_walkers(dask_client)
        if not self.workers["walkers"]: raise Exception("No workers available")
        logger.debug(f"HGDL uses {self.number_of_walkers} walkers on {len(self.workers['walkers'])} workers.")
        return dask_client

    ###########################################################################
    def _count_walkers(self, client):
        # one walker per worker thread; re-read during the run so adaptive clusters can grow or shrink
        workers = client.scheduler_info()["workers"]
        self.workers = {"walkers": list(workers)}
        self.number_of_walkers = sum(w.get("nthreads", 1) for w in workers.values())
        return self.number_of_walkers

    ###########################################################################
    def _run_walkers(self):
        """
        The coordinator, running in the background thread. It keeps one walker per
        worker thread running; whenever one finishes, its result is judged against
        the optima found so far and a new walker takes its place. There is no barrier,
        so a slow or diverging walker only occupies its own worker thread. The run
        ends after num_epochs x (number of initial walkers) walkers have finished.
        """
        n_first = len(self.x0)
        budget = self.num_epochs * n_first
        starts = list(self.x0)
        first_failures = []
        completed = 0
        target = n_first
        pool = distributed.as_completed(loop=self.client.loop)
        try:
            while not self._stop.is_set() and len(self._futures) < target and budget > 0:
                self._submit_walker(pool, starts)
            for future in pool:
                if self._stop.is_set():
                    break
                result = future.result()
                with self._lock:
                    self._futures.discard(future)
                completed += 1
                self._accept(result, completed, n_first, first_failures)
                if completed >= budget:
                    break
                if completed % n_first == 0:
                    # about once per former epoch: adaptive clusters can grow or shrink
                    target = max(self._count_walkers(self.client), 1)
                while not self._stop.is_set() and len(self._futures) < target:
                    self._submit_walker(pool, starts)
            logger.debug("HGDL finished after {} walkers", completed)
        except (Exception, asyncio.CancelledError) as err:
            if self._stop.is_set():
                logger.debug("HGDL stopped: {}", repr(err))
            else:
                logger.exception(err)
                self._error = err
        finally:
            self._stop_walkers()

    def _submit_walker(self, pool, starts):
        if not starts:
            starts.extend(self._starting_positions(max(self.number_of_walkers, 1)))
        x_defl, _, radii = self.optima.get_deflation_points(len(self.optima.list))
        future = self.client.submit(local_method, starts.pop(0), problem=self._problem,
                                    x_defl=x_defl, radius=radii, pure=False)
        with self._lock:
            self._futures.add(future)
        pool.add(future)

    def _accept(self, result, completed, n_first, first_failures):
        # judged against the current optima, which may have grown while the walker ran
        x_defl, _, radii = self.optima.get_deflation_points(len(self.optima.list))
        res = collect_results([result], self.dim, x_defl, radii)
        with self._lock:
            if res[-1][0]:
                self.optima.fill_in_optima_list(res)
            elif completed <= n_first:
                first_failures.append(result)
            if completed == n_first and not self.optima.list and first_failures:
                # none of the first walkers converged: keep their results anyway (as the
                # first epoch did), so that get_final() has an answer
                self.optima.fill_in_optima_list(collect_results(first_failures, self.dim))

    def _stop_walkers(self):
        # cancels walkers that have not started and tells running ones to stop: one message
        # per worker, after which walkers check a local flag (no requests to the scheduler)
        with self._lock:
            futures, self._futures = list(self._futures), set()
        try:
            if futures:
                self.client.cancel(futures)
            if self._run_id is not None:
                self.client.run(stop_run, self._run_id, on_error="ignore")
        except Exception as err:  # e.g. the client was closed
            logger.debug("HGDL could not stop the walkers: {}", repr(err))

    ###########################################################################
    def _starting_positions(self, number_of_walkers):
        # replace the walkers via the global method, seeded with the best optima so far
        optima_list = self.optima.list
        n = min(len(optima_list), number_of_walkers)
        if n > 0:
            x = run_global(np.array([entry["x"] for entry in optima_list[:n]]),
                           np.array([entry["f(x)"] for entry in optima_list[:n]]),
                           self.bounds, self.global_optimizer, n)
        else:
            x = np.empty((0, self.dim))
        if len(x) < number_of_walkers:
            x = np.vstack([x, misc.random_population(self.bounds, number_of_walkers - len(x))])
        return x
