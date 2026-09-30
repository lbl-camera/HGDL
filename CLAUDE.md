# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

HGDL (Hybrid Global Deflated Local) is a Python library for asynchronous, HPC-distributed
constrained optimization. It returns a *growing, sorted list of unique optima* of a
differentiable function rather than a single answer, by combining distributed local
optimization, bump-function deflation of already-found optima, and a global (genetic)
replacement step. Published to PyPI as `hgdl`; docs at hgdl.readthedocs.io.

## Commands

```bash
pip install -e .[tests]          # editable install (REQUIRED — see "Versioning" below)
pytest tests                     # full suite
pytest tests/test_schwefel.py    # single test file
pytest tests/test_schwefel.py::test_schwefel -s   # single test; -s to see the prints
pytest tests --cov=./ --cov-report=xml            # what CI runs
hatch build                      # build sdist + wheel
pip install -e .[docs] && cd docs && make html    # build Sphinx docs
```

CI (`.github/workflows/HGDL-CI.yml`) runs pytest on Python 3.10–3.14 plus a minimum-versions job, and publishes to
PyPI on pushes of `X.Y.Z` tags.

### Versioning

The version is derived from git tags by `hatch-vcs` and written to `hgdl/_version.py` at
build/install time. `hgdl/__init__.py` raises a `RuntimeError` if that file is missing, so
running from a plain source checkout without `pip install -e .` will fail immediately.

### Logging

`hgdl/__init__.py` calls `logger.disable('hgdl')`. All internal instrumentation is loguru
`logger.debug` calls that are invisible until you run `logger.enable("hgdl")`. Do this when
debugging worker behavior — most of the algorithm's state is only observable that way.

## Architecture

### Coordinator thread + walker tasks (the key structural fact)

`optimize()` is **non-blocking**: it scatters a `Problem` to every worker once
(`client.scatter(..., broadcast=True)`), starts a daemon thread (`hgdl-coordinator`) *in the
calling process* running `HGDL._run_epochs`, and returns. The main thread stays free.

- **Every worker thread is a walker**: `number_of_walkers` is the sum of `nthreads` over all
  workers, re-read at each epoch so adaptive clusters can grow or shrink. No worker is reserved
  as a host, and a single-worker client works. The walker count is not a user parameter,
  which is why `_prepare_starting_positions` silently truncates or pads `x0` to match.
- Walkers are plain dask tasks (`client.map(local_method, x0s, problem=<future>, ...)`), not
  pinned to workers and not actors; dask places them and reschedules them if a worker dies.
- State is plain Python in the calling process: the coordinator mutates `self.optima` under
  `self._lock`; `get_latest()` returns a deep copy. Nothing is named globally in the cluster,
  so several HGDL runs can share one client.
- Stopping is a `threading.Event` (`self._stop`): `cancel_tasks()` sets it and cancels the
  current epoch's futures; walkers already running finish in the background, because dask
  cannot interrupt a running task. `kill_client()` does the same and then closes the client
  (fvgp's `kill_client` relies on that).
- Errors in the coordinator (e.g. the objective raised) are stored in `self._error` and
  re-raised by `get_final()`, which joins the thread. Exceptions after `_stop` is set are the
  expected result of cancelling and are only logged.

### Epoch loop

`HGDL._run_epochs` runs `num_epochs` epochs; the first uses the prepared `x0`, the rest get
their starting points from `_starting_positions`. Each epoch:

1. `run_global` ([global_methods/global_optimizer.py](hgdl/global_methods/global_optimizer.py)) —
   `genetic_step` (weighted crossover of the current optima + perturbation, out-of-bounds
   children resampled) or `random_step` — replaces the walkers, seeded with the best optima;
   padded with random points up to the walker count.
2. One `local_method` task per walker ([local_methods/local_optimizer.py](hgdl/local_methods/local_optimizer.py))
   runs a deflated local optimization; the coordinator `gather`s them (the epoch barrier).
3. `collect_results` rejects walkers that converged onto each other or into a deflated
   region, then `optima.fill_in_optima_list` classifies and merges the results.

### Deflation

This is the mechanism that makes optima *unique*. Once an optimum is found, its location and
a radius are registered; subsequent local optimizations see a *deflated* gradient and Hessian
(`functools.partial` wrappers from [local_methods/bump_function.py](hgdl/local_methods/bump_function.py))
that blow up near known optima, so walkers cannot reconverge there. The operator is a
*product* `∏ 1/(1 − bump_i)` — every factor is ≥ 1, so overlapping bumps can't flip its sign
(a sum could). The radius is `1/min|eigenvalue of the Hessian|` — at a stationary point that is exactly the
largest principal radius of curvature of the graph of `f` (osculating circle in the flattest
direction). It scales with `f` by design (the graph lives in (x, f) space); scale- or
shift-invariant alternatives were evaluated and deliberately not adopted. It is capped at
`MAX_RADIUS_FRACTION` (0.1) of the domain diagonal so one poorly conditioned optimum can't
deflate the whole domain.
Only points classified `minimum`, `maximum`, or `saddle point` become deflation points
(`optima.get_deflation_points`).

### Optima bookkeeping

[hgdl/optima.py](hgdl/optima.py) holds `self.list`, a list of dicts with keys
`x`, `f(x)`, `classifier`, `Hessian eigvals`, `df/dx`, `|df/dx|`, `radius`, kept sorted
ascending by `f(x)`. It is **never truncated**: it is also the set of deflation points, and a
dropped point would stop being deflated and be found again. `number_of_optima` only limits what
`get_latest()`/`get_final()` return. The classifier is derived from the
gradient norm and the sign pattern of the Hessian eigenvalues (`degenerate`, `zero curvature`,
`minimum`, `maximum`, `saddle point`). Note that `fill_in_optima_list` force-accepts the first
round's results if nothing converged and the list is still empty.

### Local optimizers

`local_method` dispatches three ways: the built-in `dNewton`
([local_methods/dNewton.py](hgdl/local_methods/dNewton.py), a damped Newton with
`lstsq` fallback for singular Hessians), any `scipy.optimize.minimize` method name, or a
user callable `f(func, grad, hess, bounds, x0, *args)` returning a scipy-like result dict.
`mode` (constructor argument) decides what is searched for. `"minimization"` (default): any
local optimizer; `dNewton` takes saddle-free Newton steps (`saddle_free_step`: symmetric part of
the deflated Hessian with |eigenvalues|, floored), so it only converges to minima. Do not replace
this with `0.5 * H @ H.T`: that is a descent direction too, but not scale invariant and it
diverges for curvature < 1. `"stationary_points"`: forces plain-Newton `dNewton` (with a warning)
and rejects constraints with a `ValueError`, since those force the SLSQP minimizer.

All branches share one acceptance test after the optimizer returns: a result counts as
`local_success` when the deflated gradient is finite with `|grad| < 1e-6` **and** the *true*
(undeflated, symmetrized) Hessian has `min λ > 1e-6` (minimization) or `min |λ| > 1e-6`
(stationary points). Those true eigenvalues
also give the radius and the classification. `local_max_iter=None` (default) keeps each scipy
method's own `maxiter` and caps dNewton at `DNEWTON_DEFAULT_MAX_ITER` (1000); an explicit value
is passed to both.

Behavioral coupling to be aware of when editing `HGDL.__init__`: `dNewton` ignores `bounds`
(it only projects onto them inside the iteration) and warns; passing `constraints` silently
overrides `local_optimizer` to `"SLSQP"`. If `hess` is omitted, `problem.approximate_hessian`
(a forward-difference Hessian built from `grad`) is used via `functools.partial`, so it pickles
without the `HGDL` object.

### Known bugs

When a bug is found, pin it as an `xfail(strict=True)` test in `tests/test_hgdl.py` whose reason
string names it; there are none open right now. Workers started without a nanny
(`dask-mpi`, `--no-nanny`, `distributed.utils_test.cluster`) keep multi-threaded BLAS, and
concurrent walkers then oversubscribe the CPU; that is deliberately left to the cluster setup.

### Problem

[hgdl/problem.py](hgdl/problem.py) holds `Problem`, the snapshot of everything a walker
needs (`func`, `grad`, `hess`, `bounds`, `args`, `local_optimizer`, `local_max_iter`,
`tolerance`, `constraints`), built in `optimize()` because `tolerance` is only known there.
Anything a walker must see has to be added here. It is serialized with dask's cloudpickle,
so lambdas and closures work, but nothing in it may reference a dask client, future, lock or
thread — in particular never a bound method of `HGDL`, which now holds all of those.

## Conventions

- hgdl is upstream of fvgp and gpCAM. Its `numpy`/`scipy`/`dask`/`distributed` requirements
  and Python range (3.10–3.14) must stay identical to theirs: open `>=` ranges, never `~=`
  pins, because a pin here dictates the whole stack for every fvgp/gpCAM install. The
  `minimum` CI job installs the lower bounds exactly; raise a bound only together with
  fvgp and gpCAM.
- The API uses plain `np.ndarray`s throughout, no dataclasses or type hints.
- Public `HGDL` methods carry full numpydoc docstrings that Sphinx `autoclass` renders
  directly into the published API docs — update them when changing signatures.
- All tests live in [tests/test_hgdl.py](tests/test_hgdl.py): deterministic unit tests plus
  known-answer end-to-end runs on a four-well function, using an in-process cluster
  (`processes=False`) and a seeded global RNG so runs are reproducible and coverage sees
  worker code. Known bugs are pinned with `xfail(strict=True)` and a reason code; when a fix
  makes one XPASS, remove its marker.
- Standalone scripts outside the repo may import a stale non-editable `hgdl` from
  site-packages; pytest uses the repo copy because `tests/` is a package.
