# Changelog

## Unreleased

### Changed architecture
- No more epoch barrier: one walker per worker thread is kept running, and each is replaced by
  a new one (from the global method) as soon as it finishes, so a slow or diverging walker no
  longer holds up all others. A run ends after `num_epochs` times the number of walkers local
  optimizations have finished; walkers still running then are stopped, so `get_final()` does
  not wait for a straggler. With several workers, runs are no longer exactly reproducible with
  a seeded RNG, because walkers finish in varying order.

### New
- `local_time_limit` (seconds, default None): a walker stops its local optimization after this
  time; its result is judged as usual (and usually rejected).
- `cancel_tasks()` and `kill_client()` now stop running walkers at their next iteration, instead
  of letting them finish in the background (one message to every worker; walkers check a local
  flag). A single evaluation of the objective is still not interrupted.
- `info` (default False): if True, a line with the progress, the number of points found and the
  current best point is printed whenever a new point is accepted, and when the run ends.

### Fixed
- With constraints, optima on an active constraint were never accepted or deflated: the
  acceptance test required the gradient of `f` to vanish, which it does not there. Constrained
  results are now judged by the KKT conditions: feasibility, a vanishing gradient of the
  Lagrangian (multipliers computed from the active constraints, with the correct signs for
  inequalities), and positive curvature of the Lagrangian along the active constraints, which
  also sets the deflation radius (10% of the domain diagonal when the active constraints fix
  the point). The bounds are not treated as constraints.
- `local_optimizer="trust-constr"` raised in every walker (its result keeps the gradient under
  `grad`, not `jac`).
- SLSQP stops when `f` stops changing, which left the gradient around 1e-5 and failed the 1e-6
  acceptance test, so even interior minima of constrained runs were often rejected. Constrained
  results are now refined by a few Newton steps on the KKT system before the test.

### Behavior changes
- With constraints, `df/dx` in the results is the gradient of the Lagrangian and `Hessian eigvals`
  are the eigenvalues of its Hessian along the active constraints (fewer than the dimension when
  constraints are active).

### Docs
- The example notebooks install hgdl 2.4.0; the Schwefel example passed only one of its two
  constraints.

## 2.4.0

### Changed architecture
- HGDL no longer runs its epoch loop as a task on a dask worker. `optimize()` starts a background
  thread in the calling process and returns immediately, as before. **Every worker thread now runs
  a walker**; no worker is reserved as a host, and a single-worker client works.
- Results are kept in the calling process instead of cluster-wide `distributed.Variable`s, so
  several HGDL runs can share one client, and other clients on a shared scheduler can no longer
  read or overwrite them.
- The objective, gradient and Hessian are sent to each worker once per run instead of with every task.

### New
- `mode="minimization"` (default) or `mode="stationary_points"`. In stationary-point mode all
  non-degenerate stationary points are found, deflated and classified as minimum, maximum or
  saddle point; it uses `dNewton` (other local optimizers are replaced with a warning) and
  cannot be combined with constraints (`ValueError`).

### Behavior changes
- `get_final()` raises the error that stopped the run (e.g. an exception in the objective)
  instead of returning an empty list.
- `get_client_info()` returns `{"walkers": [...]}`; there is no `host` entry any more.
- In the default mode, `dNewton` takes saddle-free Newton steps, so it only converges to minima.
- `number_of_optima` only limits how many points `get_latest()`/`get_final()` return. All points
  found are kept and stay deflated, so they are not found again.
- `local_max_iter` defaults to `None`: scipy methods keep their own iteration limit, `dNewton`
  stops after 1000 iterations. An explicit value is now passed to scipy as `maxiter`.
- `Hessian eigvals` in the results are the eigenvalues of the true Hessian, not of the deflated one.
- The deflation radius (1 / smallest absolute Hessian eigenvalue) is capped at 10% of the domain diagonal.

### Fixed
- Overlapping deflation regions could make the deflation operator negative and attract walkers
  to known optima; the operator is now the product of the individual deflations.
- `dNewton` accepted saddle points and maxima (listed as "zero curvature", never deflated).
- A callable `local_optimizer` or `global_optimizer` raised an error.
- `x0` given as a 1-D array or a list raised an error.
- Classification ignored large negative gradient components.
- `dNewton`'s NaN check never triggered.
- Runs without a Hessian failed on in-process clusters (`Client(processes=False)`).
- `hgdl.support_functions.schwefel_gradient` modified its input.

### Performance
- Deflation is vectorized: 20–90x faster with 100–10,000 known points.

### Packaging, CI and docs
- Dependency requirements are now ranges identical to fvgp and gpCAM (`scipy>=1.13`,
  `numpy>=2.1`, `dask>=2024.1`, `distributed>=2024.1`) instead of `~=` pins, which had decided
  the scientific stack for every fvgp and gpCAM install. Python 3.10–3.14.
- CI tests Python 3.10–3.14 plus the minimum supported versions.
- The documentation uses the same Sphinx setup as fvGP and gpCAM.
- New deterministic test suite with known-answer optimization tests.
