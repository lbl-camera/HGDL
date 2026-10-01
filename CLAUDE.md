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
pytest tests                     # full suite (~30 s; all tests are in tests/test_hgdl.py)
pytest tests -k stationary       # tests matching a name
pytest tests/test_hgdl.py::test_finds_all_four_minima   # single test
pytest tests --cov=hgdl --cov-report=term-missing       # coverage (CI uses --cov=./ --cov-report=xml)
hatch build                      # build sdist + wheel
pip install -e .[docs] && cd docs && make html    # build Sphinx docs
```

Docs use the same Sphinx setup as fvGP/gpCAM (pydata theme, myst-nb with execution off). The
notebooks in `examples/` are the only tracked copies; `docs/source/conf.py` (and a pre-build step
in `.readthedocs.yml`) copies them into the gitignored `docs/source/examples/` at build time.

### CI and release

`.github/workflows/HGDL-CI.yml` runs the suite on Python 3.10–3.14 plus a `minimum` job that
installs the lower dependency bounds exactly, and uploads coverage to Codecov.

- Releases are made by pushing an `X.Y.Z` tag on `master`: the `deploy` job builds with
  `hatch build`, publishes to PyPI via **trusted publishing** (OIDC, no stored token) and creates
  the GitHub Release with `gh release create`. Any tag push releases, so never tag a branch.
  The same tag triggers `context7-refresh.yml`.
- Every action is pinned to a full commit SHA with the version in a comment; Dependabot
  (`.github/dependabot.yml`) opens monthly PRs to bump them. Review those one by one: major
  bumps can change inputs — codecov-action ≥ v5 passes `env_vars` to its CLI verbatim, so it
  must be `OS,PYTHON` without spaces.
- Workflows default to `contents: read`; only `deploy` gets `id-token: write` and
  `contents: write`.

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
calling process* running `HGDL._run_walkers`, and returns. The main thread stays free.

- **Every worker thread is a walker**: `number_of_walkers` is the sum of `nthreads` over all
  workers, re-read every `number_of_walkers` completions so adaptive clusters can grow or shrink. No worker is reserved
  as a host, and a single-worker client works. The walker count is not a user parameter,
  which is why `_prepare_starting_positions` silently truncates or pads `x0` to match.
- Walkers are plain dask tasks (`client.submit(local_method, x0, problem=<future>, ...)`), not
  pinned to workers and not actors; dask places them and reschedules them if a worker dies.
- State is plain Python in the calling process: the coordinator mutates `self.optima` under
  `self._lock`; `get_latest()` returns a deep copy. Nothing is named globally in the cluster,
  so several HGDL runs can share one client.
- Stopping: `self._stop` (a `threading.Event`) stops the coordinator; `_stop_walkers()` cancels
  the pending futures and calls `client.run(stop_run, run_id)` once on every worker, which
  adds the run's uuid to the worker-local `_STOPPED_RUNS`; running walkers check that set
  between iterations (`_Watch` in local_optimizer.py) and give up. One message per worker,
  never a request per walker: a scheduler poll per walker would not scale to thousands of
  walkers. Dask cannot interrupt a running task, so a single long evaluation of the objective
  still runs to its end. `cancel_tasks()` and the end of the run both call it.
  `kill_client()` does the same and then closes the client (fvgp's `kill_client` relies on it).
- `local_time_limit` (seconds, default None) uses the same `_Watch`: scipy methods get a
  `callback` that raises `StopIteration` (SLSQP and TNC let it escape instead of returning, so
  `local_method` catches it and uses the last iterate), dNewton a `should_stop` hook. A user
  callable `local_optimizer` cannot be stopped.
- Errors in the coordinator (e.g. the objective raised) are stored in `self._error` and
  re-raised by `get_final()`, which joins the thread. Exceptions after `_stop` is set are the
  expected result of cancelling and are only logged.

### Walker pool (no epoch barrier)

`HGDL._run_walkers` keeps one walker per worker thread in flight and consumes them with
`distributed.as_completed`, so a slow or diverging walker only occupies its own thread. The run
ends after `num_epochs × len(x0)` walkers have *completed*; whatever is still running then is
cancelled/stopped, so `get_final()` never waits for a straggler. For each completed walker:

1. `_accept` judges its result with `collect_results` against the *current* deflation points
   (the list may have grown while it ran), rejecting convergence into a deflated region, then
   `optima.fill_in_optima_list` classifies and merges it.
2. `_submit_walker` starts a replacement ([local_methods/local_optimizer.py](hgdl/local_methods/local_optimizer.py))
   with the current deflation points. Starting points come from a queue: first the prepared
   `x0`, then batches of `_starting_positions(number_of_walkers)` — `run_global`
   ([global_methods/global_optimizer.py](hgdl/global_methods/global_optimizer.py)), i.e.
   `genetic_step` (weighted crossover of the current optima + perturbation, out-of-bounds
   children resampled) or `random_step`, seeded with the best optima and padded with random
   points — so the global method still sees whole populations.

The force-accept rule of `fill_in_optima_list` (see below) is applied once, after the first
`len(x0)` completions, to the failed results among them, and only if nothing was accepted;
gpCAM relies on it to always get `get_final()[0]` (e.g. when every optimum is on a bound).
With more than one worker, completion order varies, so runs are not exactly reproducible even
with a seeded RNG; single-worker runs are.

### Deflation

This is the mechanism that makes optima *unique*. Once an optimum is found, its location and
a radius are registered; subsequent local optimizations see a *deflated* gradient and Hessian
(`functools.partial` wrappers from [local_methods/bump_function.py](hgdl/local_methods/bump_function.py))
that blow up near known optima, so walkers cannot reconverge there. This follows Noack & Funke,
"Hybrid genetic deflated Newton method for global optimisation", JCAM 325 (2017) (HGDN, the
former name): the bump `b()` is the paper's Eq. (7) with `α = r²` (radial instead of
per-coordinate support), the deflated gradient is Eq. (8). The operator is a
*product* `∏ 1/(1 − bump_i)` — every factor is ≥ 1, so overlapping bumps can't flip its sign
(a sum could). The radius is `1/min|eigenvalue of the Hessian|` — at a stationary point that is exactly the
largest principal radius of curvature of the graph of `f` (osculating circle in the flattest
direction). It scales with `f` by design (the graph lives in (x, f) space); scale- or
shift-invariant alternatives were evaluated and deliberately not adopted. It is capped at
`MAX_RADIUS_FRACTION` (0.1) of the domain diagonal so one poorly conditioned optimum can't
deflate the whole domain.
`deflation_and_gradient` evaluates all bumps at once with numpy (the scalar `b()`/`b_grad()` are
kept as the reference the tests compare against); `local_method` stacks the deflation points
into arrays once per walker.
Only points classified `minimum`, `maximum`, or `saddle point` become deflation points
(`optima.get_deflation_points`).

Known, deliberate differences from the paper (not bugs): `genetic_step` averages two parents
with fitness weights plus a tiny perturbation instead of the paper's gene crossover and
mutation, and the paper's Algorithm 4 inner loop (reset walkers and repeat the deflated search
until nothing new is found) is not implemented — each walker does one local search and is
replaced.

### Optima bookkeeping

[hgdl/optima.py](hgdl/optima.py) holds `self.list`, a list of dicts with keys
`x`, `f(x)`, `classifier`, `Hessian eigvals`, `df/dx`, `|df/dx|`, `radius`, kept sorted
ascending by `f(x)`. It is **never truncated**: it is also the set of deflation points, and a
dropped point would stop being deflated and be found again. `number_of_optima` only limits what
`get_latest()`/`get_final()` return. The classifier is derived from the
gradient norm and the sign pattern of the Hessian eigenvalues (`degenerate`, `zero curvature`,
`minimum`, `maximum`, `saddle point`). Note that `fill_in_optima_list` force-accepts all the
results it is given if none converged and the list is still empty (used once, for the first
walkers; see "Walker pool").

### Local optimizers

`local_method` dispatches three ways: the built-in `dNewton`
([local_methods/dNewton.py](hgdl/local_methods/dNewton.py), a Newton method projected onto the
bounds), any `scipy.optimize.minimize` method name, or a user callable
`f(func, grad, hess, bounds, x0, *args)` returning a scipy-like result with `x`, `fun`, `jac`.
`mode` (constructor argument) decides what is searched for. `"minimization"` (default): any
local optimizer; `dNewton` takes saddle-free Newton steps (`saddle_free_step`: symmetric part of
the deflated Hessian with |eigenvalues|, floored), so it only converges to minima. Do not replace
this with `0.5 * H @ H.T`: that is a descent direction too, but not scale invariant and it
diverges for curvature < 1. `"stationary_points"`: forces plain-Newton `dNewton` (with a warning;
singular Hessians fall back to `lstsq`) and rejects constraints with a `ValueError`, since those
force the SLSQP minimizer.

All branches share one acceptance test after the optimizer returns: a result counts as
`local_success` when the deflated gradient is finite with `|grad| < 1e-6` **and** the *true*
(undeflated, symmetrized) Hessian has `min λ > 1e-6` (minimization) or `min |λ| > 1e-6`
(stationary points). Those true eigenvalues
also give the radius and the classification.

With constraints (always SLSQP, minimization only) the test is the KKT conditions instead
([local_methods/kkt.py](hgdl/local_methods/kkt.py)): the result is first refined by a few
Newton steps on the KKT system (`newton_polish`; SLSQP stops on the change of `f`, leaving
`|grad| ≈ 1e-5`), then `kkt_conditions` computes the multipliers of the active constraints by
bounded least squares on the true `∇f` (signs enforce which side of an inequality is active),
and the result is accepted if it is feasible, `|∇f − Jᵀw| < 1e-6` and the Lagrangian's Hessian
projected onto the null space of the strongly active constraints has `min λ > 1e-6`. `g` and
`eig` returned by `local_method` are then that Lagrangian gradient and those projected
eigenvalues, so `eig` can be shorter than `dim`, which is why `collect_results` keeps `eig` as
a list. If the active constraints fix the point (no tangent space) the radius is the cap. The
bounds are deliberately *not* treated as constraints (users are told to use wide bounds;
dNewton ignores them), so an optimum on a bound is still rejected. Missing constraint
Jacobians/Hessians are central differences; the Hessian without a Jacobian uses second
differences of values, since differencing a difference quotient loses half the digits.

`local_max_iter=None` (default) keeps each scipy
method's own `maxiter` and caps dNewton at `DNEWTON_DEFAULT_MAX_ITER` (1000); an explicit value
is passed to both.

Behavioral coupling to be aware of when editing `HGDL.__init__`: `dNewton` ignores `bounds`
(it only projects onto them inside the iteration) and warns; passing `constraints` silently
overrides `local_optimizer` to `"SLSQP"` (with a warning). If `hess` is omitted, `problem.approximate_hessian`
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
`tolerance`, `constraints`, `mode`, plus `constraint_blocks`, the constraints normalized by
`kkt.normalize_constraints`), built in `optimize()` because `tolerance` is only known there.
Anything a walker must see has to be added here. It is serialized with dask's cloudpickle,
so lambdas and closures work, but nothing in it may reference a dask client, future, lock or
thread — in particular never a bound method of `HGDL`, which now holds all of those.

## Conventions

- hgdl is upstream of fvgp and gpCAM. Its `numpy`/`scipy`/`dask`/`distributed` requirements
  and Python range (3.10–3.14) must stay identical to theirs: open `>=` ranges, never `~=`
  pins, because a pin here dictates the whole stack for every fvgp/gpCAM install. The
  `minimum` CI job installs the lower bounds exactly; raise a bound only together with
  fvgp and gpCAM.
- What fvgp and gpCAM rely on (keep it stable, or change them in step): `from hgdl.hgdl import
  HGDL`; the constructor keywords `hess`, `local_optimizer`, `global_optimizer`, `num_epochs`,
  `constraints`; `optimize(dask_client=, x0=<(1, D) array>, tolerance=)`; `get_final()[0]["x"]`
  as the answer (list sorted by `f(x)`); `get_latest()`, `cancel_tasks()`, and `kill_client()`
  closing the client. gpCAM calls HGDL without a Hessian. Before a release, run gpCAM's suite
  and fvGP's `-k "hgdl or test_train_basic"` tests against the working tree (the fvGP ones take
  ~10 min on many-core machines because of the no-nanny BLAS issue under "Known bugs").
- User-visible changes (API, defaults, results, architecture) go into [CHANGELOG.md](CHANGELOG.md)
  under the topmost `(unreleased)` version heading.
- CI runs no linter or formatter; the `[tool.black]` section in `pyproject.toml` is not enforced.
- The API uses plain `np.ndarray`s throughout, no dataclasses or type hints.
- Public `HGDL` methods carry full numpydoc docstrings that Sphinx `autoclass` renders
  directly into the published API docs — update them when changing signatures.
- All tests live in [tests/test_hgdl.py](tests/test_hgdl.py): deterministic unit tests plus
  known-answer end-to-end runs on a four-well function, using an in-process cluster
  (`processes=False`, so coverage sees worker code) and a seeded global RNG. With several
  workers the walker completion order still varies, so end-to-end tests must assert sets of
  optima, never exact sequences. Known bugs are pinned with `xfail(strict=True)` and a reason code; when a fix
  makes one XPASS, remove its marker. The known-answer tests' epoch counts were chosen from
  multi-seed scans (e.g. stationary mode: 15 epochs suffice in 20/20 seeds, the test uses 20);
  don't lower them without re-running such a scan.
- Standalone scripts outside the repo may import a stale non-editable `hgdl` from
  site-packages; pytest uses the repo copy because `tests/` is a package.
