"""
Deterministic unit tests and known-answer integration tests for HGDL.

Tests marked ``xfail(strict=True)`` pin down known bugs. They are expected to
fail today; once a bug is fixed the test XPASSes, strict mode turns that into a
failure, and the marker must be removed. The reason string names the bug.

Integration tests use an in-process dask cluster (``processes=False``) so that
coverage is recorded for code running on the workers and startup is fast; one
end-to-end test also runs on a multi-process cluster, which is what fvgp and
gpCAM users get from ``Client()``.
"""
import time
import warnings

import numpy as np
import pytest
from distributed import Client, get_task_stream
from scipy.optimize import NonlinearConstraint, minimize

from hgdl import misc
from hgdl.global_methods.global_optimizer import genetic_step, random_step, run_global
from hgdl.hgdl import HGDL
from hgdl.local_methods import bump_function as bf
from hgdl.local_methods.dNewton import DNewton, saddle_free_step
from hgdl.local_methods.local_optimizer import MAX_RADIUS_FRACTION, collect_results, local_method
from hgdl.optima import optima
from hgdl.problem import Problem, approximate_hessian
from hgdl.support_functions import (non_diff, non_diff_grad, non_diff_hess,
                                    schwefel, schwefel_gradient)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


###########################################################################
# Test problems (module level so they are picklable)
###########################################################################
# Four-well function: minima at (+-1, +-1), saddles at (0, +-1) and (+-1, 0),
# maximum at (0, 0).
FOUR_MINIMA = np.array([[1., 1.], [1., -1.], [-1., 1.], [-1., -1.]])
FOUR_SADDLES = np.array([[0., 1.], [0., -1.], [1., 0.], [-1., 0.]])
MAXIMUM = np.array([0., 0.])


def fourwell(x, *args):
    return (x[0] ** 2 - 1) ** 2 + (x[1] ** 2 - 1) ** 2


def fourwell_grad(x, *args):
    return np.array([4 * x[0] * (x[0] ** 2 - 1), 4 * x[1] * (x[1] ** 2 - 1)])


def fourwell_hess(x, *args):
    return np.diag([12 * x[0] ** 2 - 4, 12 * x[1] ** 2 - 4])


QUAD_C = np.array([0.3, -0.2])


def quad(x, *args):
    return np.sum((x - QUAD_C) ** 2)


def quad_grad(x, *args):
    return 2 * (x - QUAD_C)


def quad_hess(x, *args):
    return 2 * np.eye(len(x))


SHALLOW = 1e-5


def shallow(x, *args):
    return SHALLOW * np.sum((x - QUAD_C) ** 2)


def shallow_grad(x, *args):
    return 2 * SHALLOW * (x - QUAD_C)


def shallow_hess(x, *args):
    return 2 * SHALLOW * np.eye(len(x))


def x0_at_least_half(x):
    return x[0]


def bfgs_local_optimizer(func, grad, hess, bounds, x0, *args):
    return minimize(func, x0, jac=grad, method="BFGS", args=args)


def my_global_optimizer(x, y, bounds, n):
    return np.zeros((n, len(bounds)))


def exploding(x, *args):
    raise ValueError("objective blew up")


def slow_quad(x, *args):
    time.sleep(0.05)
    return quad(x)


BOUNDS = np.array([[-2., 2.], [-2., 2.]])
UNIT = np.array([[-1., 1.], [-1., 1.]])


###########################################################################
# Helpers and fixtures
###########################################################################
def make_hgdl(func=quad, grad=quad_grad, hess=quad_hess, bounds=UNIT, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return HGDL(func, grad, bounds, hess=hess, **kwargs)


def make_problem(tolerance=1e-10, **kwargs):
    """The Problem a walker receives, built the way HGDL.optimize() builds it."""
    h = make_hgdl(**kwargs)
    return Problem(h.func, h.grad, h.hess, h.bounds, h.args, h.local_optimizer,
                   h.local_max_iter, tolerance, h.constraints, h.mode)


def walk(problem, x0, x_defl=(), radius=()):
    return local_method(np.array(x0, dtype=float), problem, x_defl, radius)


@pytest.fixture(autouse=True)
def seed():
    # Every random draw (x0, padding, genetic step) happens on the global numpy RNG
    # in this process (the coordinator thread), so seeding makes whole runs reproducible.
    np.random.seed(12345)


def make_client(n_workers=5):
    return Client(n_workers=n_workers, threads_per_worker=1, processes=False, dashboard_address=None)


@pytest.fixture
def client():
    c = make_client()
    yield c
    c.close()


def contains_point(points, target, atol=1e-4):
    return any(np.allclose(p, target, atol=atol) for p in points)


def wait_for_results(h, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        res = h.get_latest()
        if len(res) > 0:
            return res
        time.sleep(0.1)
    raise TimeoutError("HGDL produced no results")


###########################################################################
# misc
###########################################################################
def test_out_of_bounds():
    assert misc.out_of_bounds(np.array([0., 2.]), UNIT)
    assert misc.out_of_bounds(np.array([-2., 0.]), UNIT)
    assert not misc.out_of_bounds(np.array([0., 0.]), UNIT)


def test_in_bounds_is_strict():
    assert misc.in_bounds(np.array([0., 0.]), UNIT)
    assert not misc.in_bounds(np.array([1., 0.]), UNIT)
    assert not misc.in_bounds(np.array([0., -3.]), UNIT)


@pytest.mark.parametrize("sampler", [lambda b, n: misc.random_sample(n, len(b), b), misc.random_population])
def test_random_samplers_respect_bounds(sampler):
    bounds = np.array([[0., 1.], [10., 20.], [-5., -4.]])
    s = sampler(bounds, 500)
    assert s.shape == (500, 3)
    assert np.all(s >= bounds[:, 0]) and np.all(s <= bounds[:, 1])


def test_project_onto_bounds():
    x = np.array([-3., 0.5, 3.])
    b = np.array([[-1., 1.]] * 3)
    np.testing.assert_array_equal(misc.project_onto_bounds(x, b), [-1., 0.5, 1.])
    np.testing.assert_array_equal(x, [-3., 0.5, 3.])  # input untouched


class _FakeFuture:
    def __init__(self, statuses):
        self._statuses = list(statuses)

    @property
    def status(self):
        return self._statuses.pop(0) if len(self._statuses) > 1 else self._statuses[0]


def test_finish_up_tasks():
    slow = _FakeFuture(["finished", "pending", "finished"])
    cancelled = _FakeFuture(["cancelled"])
    done = _FakeFuture(["finished"])
    assert misc.finish_up_tasks([slow, cancelled, done]) == [slow, done]


###########################################################################
# support_functions
###########################################################################
def test_schwefel_global_minimum():
    assert abs(schwefel(np.array([420.9687, 420.9687]))) < 1e-3


def test_schwefel_gradient_matches_finite_differences():
    x = np.array([100., -250.])
    eps = 1e-6
    fd = np.array([(schwefel(x + eps * e) - schwefel(x - eps * e)) / (2 * eps) for e in np.eye(2)])
    np.testing.assert_allclose(schwefel_gradient(x.copy()), fd, rtol=1e-5)


def test_schwefel_gradient_at_zero_is_finite():
    assert np.all(np.isfinite(schwefel_gradient(np.array([0., 1.]))))


def test_schwefel_gradient_does_not_mutate_input():
    x = np.array([0., 1.])
    schwefel_gradient(x)
    np.testing.assert_array_equal(x, [0., 1.])


def test_non_diff_functions():
    assert non_diff(np.array([2., 2.])) == -1.0
    assert non_diff(np.array([0., 0.])) == 0.0
    np.testing.assert_array_equal(non_diff_grad(np.zeros(3)), np.zeros(3))
    np.testing.assert_array_equal(non_diff_hess(np.zeros(3)), np.zeros((3, 3)))


###########################################################################
# bump_function
###########################################################################
def test_bump_values():
    x0 = np.array([0., 0.])
    assert bf.b(x0, x0, 1.0) == 1.0
    assert bf.b(np.array([2., 0.]), x0, 1.0) == 0.0
    assert 0.0 < bf.b(np.array([0.5, 0.]), x0, 1.0) < 1.0


def test_bump_gradient_outside_is_zero():
    np.testing.assert_array_equal(bf.b_grad(np.array([2., 0.]), np.zeros(2), 1.0), np.zeros(2))


def test_bump_gradient_matches_finite_differences():
    x, x0, r, eps = np.array([0.3, -0.4]), np.array([0.1, 0.1]), 1.5, 1e-7
    fd = np.array([(bf.b(x + eps * e, x0, r) - bf.b(x - eps * e, x0, r)) / (2 * eps) for e in np.eye(2)])
    np.testing.assert_allclose(bf.b_grad(x, x0, r), fd, rtol=1e-5)


def test_deflation_without_points_is_identity():
    x = np.array([0.2, 0.2])
    assert bf.deflation_function(x, [], []) == 1.0
    np.testing.assert_array_equal(bf.deflation_function_gradient(x, [], []), np.zeros(2))
    np.testing.assert_array_equal(bf.deflated_grad(x, grad_func=quad_grad), quad_grad(x))
    np.testing.assert_array_equal(bf.deflated_hess(x, grad_func=quad_grad, hess_func=quad_hess), quad_hess(x))


def test_deflation_at_deflation_point_is_large_but_finite():
    x = np.array([0.5, 0.5])
    assert bf.deflation_function(x, [x], [1.0]) == pytest.approx(1e5)
    assert np.all(np.isfinite(bf.deflation_function_gradient(x, [x], [1.0])))


def test_deflation_gradient_matches_finite_differences():
    x, xd, r, eps = np.array([0.3, -0.4]), [np.array([0.1, 0.1])], [1.5], 1e-7
    fd = np.array([(bf.deflation_function(x + eps * e, xd, r) - bf.deflation_function(x - eps * e, xd, r))
                   / (2 * eps) for e in np.eye(2)])
    np.testing.assert_allclose(bf.deflation_function_gradient(x, xd, r), fd, rtol=1e-5)


def test_deflated_grad_and_hess_scale_with_deflation():
    x, xd, r = np.array([0.3, -0.4]), [np.array([0.1, 0.1])], [1.5]
    d = bf.deflation_function(x, xd, r)
    assert d > 1.0
    g = bf.deflated_grad(x, grad_func=quad_grad, x_defl=xd, radius=r)
    np.testing.assert_allclose(g, d * quad_grad(x))
    h = bf.deflated_hess(x, grad_func=quad_grad, hess_func=quad_hess, x_defl=xd, radius=r)
    expected = d * quad_hess(x) + np.outer(quad_grad(x), bf.deflation_function_gradient(x, xd, r))
    np.testing.assert_allclose(h, expected)


def _reference_deflation(x, x0, r):
    # the operator written point by point with the scalar b()/b_grad(), as before vectorization
    d, s = 1.0, np.zeros(len(x))
    for p, rr in zip(x0, r):
        capped = min(bf.b(x, p, rr), bf.BUMP_CAP)
        d /= 1.0 - capped
        s += bf.b_grad(x, p, rr) / (1.0 - capped)
    return d, d * s


def test_vectorized_deflation_matches_point_by_point():
    rng = np.random.default_rng(3)
    points = list(rng.uniform(-1, 1, (25, 2)))
    radii = list(rng.uniform(0.05, 0.6, 25))  # plenty of overlaps
    for x in list(rng.uniform(-1.2, 1.2, (300, 2))) + [points[0], points[1] + 1e-9]:
        d, g = bf.deflation_and_gradient(x, points, radii)
        d_ref, g_ref = _reference_deflation(x, points, radii)
        assert d == pytest.approx(d_ref, rel=1e-12)
        np.testing.assert_allclose(g, g_ref, rtol=1e-10, atol=1e-12)


def test_overlapping_deflation_stays_positive():
    x = np.array([0., 0.])
    points = [np.array([0.1, 0.]), np.array([-0.1, 0.])]
    assert bf.deflation_function(x, points, [1., 1.]) >= 1.0


###########################################################################
# dNewton
###########################################################################
def test_dnewton_converges_on_quadratic():
    x, f, g, eig, success = DNewton(quad, quad_grad, quad_hess, UNIT, np.array([0.9, 0.9]), 100, 1e-10)
    assert success
    np.testing.assert_allclose(x, QUAD_C, atol=1e-10)
    np.testing.assert_allclose(np.sort(eig), [2., 2.])


def test_dnewton_singular_hessian_uses_lstsq():
    grad = lambda x: np.array([2 * x[0], 0.])
    hess = lambda x: np.array([[2., 0.], [0., 0.]])
    x, f, g, eig, success = DNewton(lambda x: x[0] ** 2, grad, hess, UNIT, np.array([0.5, 0.5]), 100, 1e-10)
    assert success
    assert abs(x[0]) < 1e-10


def test_dnewton_infinite_step_aborts():
    grad = lambda x: np.array([-np.inf, 0.])
    res = DNewton(lambda x: 0., grad, lambda x: np.eye(2), UNIT, np.zeros(2), 100, 1e-10)
    assert res[-1] is False


def test_dnewton_max_iter_aborts():
    res = DNewton(fourwell, fourwell_grad, fourwell_hess, BOUNDS, np.array([1.7, 1.9]), 1, 0.0)
    assert res[-1] is False


def test_dnewton_nan_gradient_reports_failure():
    grad = lambda x: np.array([np.nan, 0.])
    res = DNewton(lambda x: 0., grad, lambda x: np.eye(2), UNIT, np.zeros(2), 100, 1e-10)
    assert res[-1] is False


def scaled_fourwell(x, *args):
    return 0.01 * fourwell(x)


def scaled_fourwell_grad(x, *args):
    return 0.01 * fourwell_grad(x)


def scaled_fourwell_hess(x, *args):
    return 0.01 * fourwell_hess(x)


@pytest.mark.parametrize("start, expected", [([0.05, 0.9], [0., 1.]),    # near a saddle
                                             ([0.1, -0.1], [0., 0.])])   # near the maximum
def test_plain_dnewton_converges_to_nearby_stationary_point(start, expected):
    x, *_ = DNewton(fourwell, fourwell_grad, fourwell_hess, BOUNDS, np.array(start), 100, 1e-10)
    np.testing.assert_allclose(x, expected, atol=1e-8)


@pytest.mark.parametrize("start, expected", [([0.05, 0.9], [1., 1.]),
                                             ([0.1, -0.1], [1., -1.])])
def test_saddle_free_dnewton_escapes_saddles_and_maxima(start, expected):
    x, f, g, eig, success = DNewton(fourwell, fourwell_grad, fourwell_hess, BOUNDS, np.array(start),
                                    100, 1e-10, saddle_free=True)
    assert success
    np.testing.assert_allclose(x, expected, atol=1e-8)


def test_saddle_free_dnewton_is_scale_invariant():
    # 0.5*H@H.T would diverge here (curvature 0.08 < 1); saddle-free Newton is exact Newton at minima
    x, *_ = DNewton(scaled_fourwell, scaled_fourwell_grad, scaled_fourwell_hess, BOUNDS,
                    np.array([0.9, 0.9]), 100, 1e-10, saddle_free=True)
    np.testing.assert_allclose(x, [1., 1.], atol=1e-8)


def test_saddle_free_step_is_newton_at_minima_and_always_descends():
    H, g = np.diag([2., 3.]), np.array([1., -1.])
    np.testing.assert_allclose(saddle_free_step(H, g), np.linalg.solve(H, -g))
    H_indef = np.diag([2., -3.])
    assert g @ saddle_free_step(H_indef, g) < 0
    assert np.all(np.isfinite(saddle_free_step(np.zeros((2, 2)), g)))  # singular: floored


def test_saddle_free_dnewton_nan_hessian_aborts():
    res = DNewton(lambda x: 0., lambda x: np.ones(2), lambda x: np.full((2, 2), np.nan), UNIT,
                  np.zeros(2), 100, 1e-10, saddle_free=True)
    assert res[-1] is False


###########################################################################
# local_method (called directly, no dask involved)
###########################################################################
@pytest.mark.parametrize("method", ["dNewton", "L-BFGS-B", "BFGS"])
def test_local_method_success(method):
    # on BOUNDS the radius cap (0.1 * diagonal = 0.57) is above 1/lambda = 0.5
    x, f, g, eig, r, success = walk(make_problem(local_optimizer=method, bounds=BOUNDS), [0.9, 0.9])
    assert success
    np.testing.assert_allclose(x, QUAD_C, atol=1e-5)
    assert f == pytest.approx(0., abs=1e-9)
    assert r == pytest.approx(0.5, rel=1e-3)
    np.testing.assert_allclose(eig, [2., 2.])


@pytest.mark.parametrize("method", ["dNewton", "L-BFGS-B"])
def test_local_method_rejects_maximum(method):
    # (0, 0) is a stationary maximum of the four-well function
    problem = make_problem(func=fourwell, grad=fourwell_grad, hess=fourwell_hess, bounds=BOUNDS,
                           local_optimizer=method)
    x, f, g, eig, r, success = walk(problem, [0., 0.])
    assert not success
    assert r == 0.0
    np.testing.assert_array_equal(eig, [0.])


def test_local_method_unknown_method_raises():
    problem = make_problem()
    problem.local_optimizer = 123
    with pytest.raises(Exception, match="no local method"):
        walk(problem, [0.9, 0.9])


def test_local_method_callable_optimizer():
    x, f, g, eig, r, success = walk(make_problem(local_optimizer=bfgs_local_optimizer), [0.9, 0.9])
    assert success
    np.testing.assert_allclose(x, QUAD_C, atol=1e-5)


def test_local_method_reports_true_hessian_eigenvalues():
    problem = make_problem(local_optimizer="dNewton")
    x, f, g, eig, r, success = walk(problem, [0.9, 0.9], [QUAD_C + 0.3], [1.0])
    assert success
    np.testing.assert_allclose(np.sort(eig), [2., 2.], rtol=1e-6)


CALLS = {"n": 0}


def counted_fourwell_grad(x, *args):
    CALLS["n"] += 1
    return fourwell_grad(x)


def test_default_local_max_iter_keeps_scipy_defaults(monkeypatch):
    import hgdl.local_methods.local_optimizer as lo
    seen = {}

    def fake_minimize(*args, **kwargs):
        seen.update(kwargs["options"])
        return {"x": QUAD_C, "fun": 0., "jac": np.zeros(2)}

    monkeypatch.setattr(lo, "minimize", fake_minimize)
    walk(make_problem(local_optimizer="SLSQP"), [0.9, 0.9])
    assert "maxiter" not in seen


def test_default_local_max_iter_for_dnewton(monkeypatch):
    import hgdl.local_methods.local_optimizer as lo
    seen = {}

    def fake_dnewton(func, grad, hess, bounds, x0, max_iter, tol, *args, saddle_free=False):
        seen["max_iter"] = max_iter
        return QUAD_C, 0., np.zeros(2), None, True

    monkeypatch.setattr(lo, "DNewton", fake_dnewton)
    walk(make_problem(local_optimizer="dNewton"), [0.9, 0.9])
    assert seen["max_iter"] == lo.DNEWTON_DEFAULT_MAX_ITER
    walk(make_problem(local_optimizer="dNewton", local_max_iter=7), [0.9, 0.9])
    assert seen["max_iter"] == 7


def test_local_max_iter_limits_scipy_optimizers():
    problem = make_problem(func=fourwell, grad=counted_fourwell_grad, hess=fourwell_hess, bounds=BOUNDS,
                           local_optimizer="L-BFGS-B", local_max_iter=1)
    CALLS["n"] = 0
    walk(problem, [1.9, -1.7])
    assert CALLS["n"] <= 5  # one iteration plus its line search, and the final eigen check


def test_local_method_radius_is_bounded_by_domain():
    # 1/lambda = 5e4 here; the radius is capped at a fraction of the domain diagonal
    problem = make_problem(func=shallow, grad=shallow_grad, hess=shallow_hess, local_optimizer="dNewton")
    x, f, g, eig, r, success = walk(problem, [0.9, 0.9])
    assert success
    assert r == pytest.approx(MAX_RADIUS_FRACTION * np.linalg.norm(UNIT[:, 1] - UNIT[:, 0]))


def test_local_method_minimization_dnewton_escapes_saddle():
    problem = make_problem(func=fourwell, grad=fourwell_grad, hess=fourwell_hess, bounds=BOUNDS,
                           local_optimizer="dNewton")
    x, f, g, eig, r, success = walk(problem, [0.05, 0.9])
    assert success
    np.testing.assert_allclose(x, [1., 1.], atol=1e-6)


@pytest.mark.parametrize("start, point, eigenvalues", [([0.05, 0.9], [0., 1.], [-4., 8.]),
                                                       ([0.1, -0.1], [0., 0.], [-4., -4.])])
def test_local_method_stationary_mode_accepts_saddles_and_maxima(start, point, eigenvalues):
    problem = make_problem(func=fourwell, grad=fourwell_grad, hess=fourwell_hess, bounds=BOUNDS,
                           local_optimizer="dNewton", mode="stationary_points")
    x, f, g, eig, r, success = walk(problem, start)
    assert success
    np.testing.assert_allclose(x, point, atol=1e-8)
    np.testing.assert_allclose(eig, eigenvalues)
    assert r == pytest.approx(0.25)  # 1 / min|eigenvalue|


###########################################################################
# collect_results (post-processing of one epoch, runs in the coordinator)
###########################################################################
def walker_result(x, r=0.5, success=True, g=(0., 0.)):
    return np.array(x, float), 0., np.array(g, float), np.array([2., 2.]), r, success


def test_collect_results_stacks_walkers():
    x, f, g, eig, r, success = collect_results(
        [walker_result([0.1, 0.1]), walker_result([0.9, 0.9], r=0.0, success=False)], 2)
    assert x.shape == g.shape == eig.shape == (2, 2)
    assert list(success) == [True, False]


def test_collect_results_dedupes_walkers_that_converged_together():
    res = collect_results([walker_result([0.3, -0.2])] * 3, 2)
    # every walker converged to the same minimum; only the first is kept
    assert list(res[-1]) == [True, False, False]


def test_collect_results_flags_convergence_inside_deflation_radius():
    res = collect_results([walker_result([0.3, -0.2])], 2, x_defl=[np.array([5., 5.])], radii=[10.])
    assert not any(res[-1])


def test_collect_results_of_empty_epoch():
    x, f, g, eig, r, success = collect_results([], 2)
    assert x.shape == (0, 2) and len(success) == 0


###########################################################################
# optima
###########################################################################
def make_res(x, f, g, eig, r, success):
    return (np.array(x, float), np.array(f, float), np.array(g, float),
            np.array(eig, float), np.array(r, float), np.array(success, bool))


def test_optima_classifier_branches():
    o = optima(2, 100)
    o.fill_in_optima_list(make_res(
        x=[[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]],
        f=[0, 1, 2, 3, 4, 5],
        g=[[1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
        eig=[[1, 1], [0, 1], [1, 2], [-1, -2], [1, -1], [np.nan, np.nan]],
        r=[1, 1, 1, 1, 1, 1],
        success=[True] * 6))
    assert [e["classifier"] for e in o.list] == [
        "degenerate", "zero curvature", "minimum", "maximum", "saddle point", "ERROR"]
    assert set(o.list[0]) == {"x", "f(x)", "classifier", "Hessian eigvals", "df/dx", "|df/dx|", "radius"}


def test_optima_classifier_uses_gradient_magnitude():
    o = optima(2, 100)
    o.fill_in_optima_list(make_res([[0, 0]], [0], [[-1, 0]], [[1, 2]], [1], [True]))
    assert o.list[0]["classifier"] == "degenerate"


def test_optima_keeps_and_deflates_every_point_sorted():
    # max_optima=3, but all four points stay: the list is also the set of deflation points
    o = optima(1, 3)
    o.fill_in_optima_list(make_res([[0], [1]], [5, 1], [[0], [0]], [[1], [1]], [1, 1], [True, True]))
    o.fill_in_optima_list(make_res([[2], [3]], [3, 0], [[0], [0]], [[1], [1]], [1, 1], [True, True]))
    assert [e["f(x)"] for e in o.list] == [0, 1, 3, 5]
    assert len(o.get_deflation_points(len(o.list))[0]) == 4


def test_get_latest_returns_at_most_number_of_optima():
    h = make_hgdl(number_of_optima=2)
    h.optima.fill_in_optima_list(make_res([[0, 0], [1, 1], [2, 2]], [3, 1, 2], [[0, 0]] * 3,
                                          [[1, 1]] * 3, [1, 1, 1], [True] * 3))
    assert [e["f(x)"] for e in h.get_latest()] == [1, 2]
    assert len(h.optima.list) == 3  # all still kept and deflated


def test_optima_force_accepts_first_round():
    o = optima(1, 10)
    o.fill_in_optima_list(make_res([[0], [1]], [1, 2], [[5], [5]], [[0], [0]], [0, 0], [False, False]))
    assert len(o.list) == 2


def test_optima_ignores_failed_rounds_once_filled():
    o = optima(1, 10)
    o.fill_in_optima_list(make_res([[0]], [1], [[0]], [[1]], [1], [True]))
    before = list(o.list)
    assert o.fill_in_optima_list(make_res([[1]], [0], [[0]], [[1]], [1], [False])) == before


def test_optima_getters():
    o = optima(2, 100)
    o.fill_in_optima_list(make_res(
        x=[[1, 1], [2, 2], [3, 3], [4, 4]], f=[1, 2, 3, 4],
        g=[[0, 0]] * 4, eig=[[1, 2], [-1, -2], [1, -1], [0, 1]], r=[1, 2, 3, 4],
        success=[True] * 4))
    assert [e["f(x)"] for e in o.get_minima(10)] == [1]
    assert [e["f(x)"] for e in o.get_maxima(10)] == [2]
    assert o.get_minima(0) == []
    x, f, r = o.get_deflation_points(10)
    assert f == [1, 2, 3] and r == [1, 2, 3]  # zero-curvature entry is not deflated
    assert len(x) == 3


def test_optima_getters_tolerate_malformed_list():
    o = optima(2, 100)
    o.list = [{}]
    assert o.get_minima(1) is None
    assert o.get_maxima(1) is None
    assert o.get_deflation_points(1) == ([], [], [])


###########################################################################
# global optimizer
###########################################################################
def test_genetic_step_children_in_bounds():
    X = misc.random_population(BOUNDS, 10)
    y = np.array([fourwell(x) for x in X])
    children = run_global(X, y, BOUNDS, "genetic", 25)
    assert children.shape == (25, 2)
    assert all(misc.in_bounds(c, BOUNDS) for c in children)


def test_genetic_step_equal_fitness():
    X = misc.random_population(BOUNDS, 5)
    children = genetic_step(X, np.ones(5), BOUNDS, 7)
    assert children.shape == (7, 2)


def test_genetic_step_resamples_out_of_bounds_children():
    X = np.ones((5, 2))  # every parent on the corner: most children land outside
    children = genetic_step(X, np.arange(5.), UNIT, 50)
    assert all(misc.in_bounds(c, UNIT) for c in children)


def test_genetic_step_nan_fitness_raises():
    X = misc.random_population(BOUNDS, 3)
    with pytest.raises(Exception, match="isnans"):
        genetic_step(X, np.array([1., np.nan, 2.]), BOUNDS, 3)


def test_random_step():
    children = run_global(np.zeros((1, 2)), np.zeros(1), BOUNDS, "random", 40)
    assert children.shape == (40, 2)
    assert np.all(children >= BOUNDS[:, 0]) and np.all(children <= BOUNDS[:, 1])
    assert random_step(None, None, BOUNDS, 0).shape == (0, 2)


def test_run_global_unknown_method_raises():
    with pytest.raises(Exception, match="no global method"):
        run_global(np.zeros((1, 2)), np.zeros(1), BOUNDS, "annealing", 1)


def test_run_global_callable():
    out = run_global(np.zeros((1, 2)), np.zeros(1), BOUNDS, my_global_optimizer, 3)
    np.testing.assert_array_equal(out, np.zeros((3, 2)))


###########################################################################
# HGDL construction and helpers (no client)
###########################################################################
def test_hess_defaults_to_finite_difference_approximation():
    import pickle
    h = HGDL(fourwell, fourwell_grad, BOUNDS)
    assert h.hess.func is approximate_hessian
    for x in [np.array([0.3, -1.2]), np.array([1., 1.])]:
        np.testing.assert_allclose(h.hess(x), fourwell_hess(x), atol=1e-4)
    # picklable on its own, without dragging the HGDL object along (was C16)
    restored = pickle.loads(pickle.dumps(h.hess))
    np.testing.assert_allclose(restored(np.array([1., 1.])), fourwell_hess(np.array([1., 1.])), atol=1e-4)


def test_user_hessian_is_used_as_is():
    assert HGDL(fourwell, fourwell_grad, BOUNDS, hess=fourwell_hess).hess is fourwell_hess


def test_default_local_optimizer_is_lbfgsb():
    assert HGDL(quad, quad_grad, UNIT).local_optimizer == "L-BFGS-B"


def test_dnewton_warns_about_bounds():
    with pytest.warns(UserWarning, match="dNewton"):
        HGDL(quad, quad_grad, UNIT, local_optimizer="dNewton")


def test_constraints_force_slsqp():
    nlc = NonlinearConstraint(x0_at_least_half, 0.5, np.inf)
    with pytest.warns(UserWarning, match="SLSQP"):
        h = HGDL(quad, quad_grad, UNIT, local_optimizer="BFGS", constraints=(nlc,))
    assert h.local_optimizer == "SLSQP"
    assert h.constraints == (nlc,)


def test_mode_defaults_to_minimization():
    assert make_hgdl().mode == "minimization"


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="mode must be"):
        HGDL(quad, quad_grad, UNIT, mode="maxima")


def test_stationary_mode_forces_dnewton():
    with pytest.warns(UserWarning, match="requires dNewton"):
        h = HGDL(quad, quad_grad, UNIT, local_optimizer="BFGS", mode="stationary_points")
    assert h.local_optimizer == "dNewton"


def test_stationary_mode_keeps_dnewton_without_switch_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        h = HGDL(quad, quad_grad, UNIT, local_optimizer="dNewton", mode="stationary_points")
    assert h.local_optimizer == "dNewton"
    assert not any("requires dNewton" in str(w.message) for w in caught)


def test_stationary_mode_rejects_constraints():
    nlc = NonlinearConstraint(x0_at_least_half, 0.5, np.inf)
    with pytest.raises(ValueError, match="cannot be combined"):
        HGDL(quad, quad_grad, UNIT, constraints=(nlc,), mode="stationary_points")


def prepared(x0, n=4):
    h = make_hgdl()
    h.number_of_walkers = n
    return h._prepare_starting_positions(x0)


def test_prepare_starting_positions_random():
    x0 = prepared(None)
    assert x0.shape == (4, 2)
    assert np.all(np.abs(x0) <= 1)


def test_prepare_starting_positions_pads():
    x0 = prepared(np.array([[0.1, 0.2]]))
    assert x0.shape == (4, 2)
    np.testing.assert_array_equal(x0[0], [0.1, 0.2])


def test_prepare_starting_positions_truncates():
    given = misc.random_population(UNIT, 6)
    np.testing.assert_array_equal(prepared(given), given[:4])


def test_prepare_starting_positions_exact():
    given = misc.random_population(UNIT, 4)
    np.testing.assert_array_equal(prepared(given), given)


def test_prepare_starting_positions_wrong_dim_raises():
    with pytest.raises(Exception, match="dimensionality"):
        prepared(np.zeros((3, 5)))


def test_prepare_starting_positions_accepts_1d():
    x0 = prepared(np.array([0.1, 0.2]))
    np.testing.assert_array_equal(x0[0], [0.1, 0.2])


def test_prepare_starting_positions_accepts_list():
    x0 = prepared([[0.1, 0.2]])
    np.testing.assert_array_equal(x0[0], [0.1, 0.2])


def test_get_latest_and_final_before_optimize_return_empty():
    h = make_hgdl()
    assert h.get_latest() == []
    assert h.get_final() == []


def test_cancel_before_optimize_is_harmless():
    assert make_hgdl().cancel_tasks() == []


def test_kill_client_before_optimize_raises():
    with pytest.raises(RuntimeError, match="kill failed"):
        make_hgdl().kill_client()


def test_starting_positions_without_optima_are_random():
    h = make_hgdl()
    x = h._starting_positions(3)
    assert x.shape == (3, 2)
    assert np.all(np.abs(x) <= 1)


def test_import_without_version_file_raises(monkeypatch):
    import importlib
    import sys
    for name in [m for m in sys.modules if m == "hgdl" or m.startswith("hgdl.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "hgdl._version", None)
    with pytest.raises(RuntimeError, match="pip install -e"):
        importlib.import_module("hgdl")


def test_problem_snapshot():
    problem = make_problem(args=(1, 2), local_optimizer="BFGS")
    assert problem.args == (1, 2)
    assert problem.local_optimizer == "BFGS"
    assert problem.dim == 2 and problem.local_max_iter is None
    assert problem.constraints == () and problem.tolerance == 1e-10


###########################################################################
# dask client handling
###########################################################################
def test_every_worker_is_a_walker(client):
    h = make_hgdl()
    assert h._init_dask_client(client) is client
    info = h.get_client_info()
    assert set(info) == {"walkers"}
    assert h.number_of_walkers == 5 == len(info["walkers"])


def test_one_walker_per_worker_thread():
    c = Client(n_workers=2, threads_per_worker=3, processes=False, dashboard_address=None)
    try:
        h = make_hgdl()
        h._init_dask_client(c)
        assert h.number_of_walkers == 6
    finally:
        c.close()


def test_init_dask_client_default(client, monkeypatch):
    monkeypatch.setattr("dask.distributed.Client", lambda: client)
    h = make_hgdl()
    assert h._init_dask_client(None) is client


def test_init_dask_client_without_workers_raises():
    class NoWorkers:
        def scheduler_info(self):
            return {"workers": {}}

    with pytest.raises(Exception, match="No workers"):
        make_hgdl()._init_dask_client(NoWorkers())


def test_single_worker_client_works():
    c = make_client(n_workers=1)
    try:
        h = make_hgdl(num_epochs=3)
        h.optimize(dask_client=c)
        final = h.get_final()
        assert h.number_of_walkers == 1
        assert np.allclose(final[0]["x"], QUAD_C, atol=1e-5)
    finally:
        c.close()


###########################################################################
# asynchronous execution
###########################################################################
def test_optimize_returns_immediately_and_runs_in_background(client):
    h = make_hgdl(slow_quad, quad_grad, quad_hess, num_epochs=10 ** 6)
    t0 = time.time()
    h.optimize(dask_client=client)
    assert time.time() - t0 < 5
    assert h._thread.is_alive()
    wait_for_results(h)
    with pytest.raises(RuntimeError, match="already running"):
        h.optimize(dask_client=client)
    h.cancel_tasks()
    h._thread.join(timeout=60)
    assert not h._thread.is_alive()


def test_walkers_run_on_every_worker(client):
    h = make_hgdl(slow_quad, quad_grad, quad_hess, num_epochs=3)
    with get_task_stream(client) as ts:
        h.optimize(dask_client=client)
        h.get_final()
    walker_tasks = [t for t in ts.data if "local_method" in t["key"]]
    assert len(walker_tasks) == 3 * 5
    assert {t["worker"] for t in walker_tasks} == set(client.scheduler_info()["workers"])


def test_objective_errors_are_raised_by_get_final(client):
    h = make_hgdl(exploding, quad_grad, quad_hess, num_epochs=5)
    h.optimize(dask_client=client)
    with pytest.raises(ValueError, match="objective blew up"):
        h.get_final()
    assert h.get_latest() == []


def test_two_runs_on_one_client_do_not_interfere(client):
    a = make_hgdl(num_epochs=10 ** 6)
    b = make_hgdl(fourwell, fourwell_grad, fourwell_hess, BOUNDS, num_epochs=10 ** 6)
    a.optimize(dask_client=client)
    b.optimize(dask_client=client)
    wait_for_results(a)
    wait_for_results(b)
    a.cancel_tasks()
    a.get_final()
    assert b._thread.is_alive()  # cancelling a did not stop b
    b.cancel_tasks()
    assert np.allclose(a.get_final()[0]["x"], QUAD_C, atol=1e-5)
    # each run only saw its own objective
    minima_b = [e["x"] for e in b.get_final() if e["classifier"] == "minimum"]
    assert minima_b and all(contains_point(FOUR_MINIMA, m) for m in minima_b)


def test_coordinator_stops_before_an_epoch_when_cancelled(client):
    h = make_hgdl(num_epochs=10)
    h.client = h._init_dask_client(client)
    h.tolerance = 1e-10
    h.x0 = h._prepare_starting_positions(None)
    h._problem = client.scatter(make_problem(), broadcast=True, hash=False)
    h._stop.set()
    h._run_epochs()
    assert h.get_latest() == [] and h._error is None


###########################################################################
# end-to-end runs with known answers
###########################################################################
@pytest.mark.parametrize("method", [
    "dNewton",
    "L-BFGS-B"])
def test_finds_all_four_minima(client, method):
    h = make_hgdl(fourwell, fourwell_grad, fourwell_hess, BOUNDS,
                  local_optimizer=method, num_epochs=20)
    h.optimize(dask_client=client)
    final = h.get_final()
    # every critical point of the four-well function has non-zero curvature
    assert not [e for e in final if e["classifier"] == "zero curvature"]
    minima = [e["x"] for e in final if e["classifier"] == "minimum"]
    for target in FOUR_MINIMA:
        assert contains_point(minima, target), f"missing minimum {target}"
    # deflation keeps the minima unique
    assert len(minima) == 4
    assert all(e["f(x)"] == pytest.approx(0., abs=1e-8) for e in final if e["classifier"] == "minimum")
    fs = [e["f(x)"] for e in final]
    assert fs == sorted(fs)
    assert [e["f(x)"] for e in h.get_latest()] == fs


def test_stationary_mode_finds_and_classifies_all_nine_stationary_points(client):
    # 15 epochs sufficed in 20/20 seeds (10 epochs: 18/20); 20 leaves a margin
    h = make_hgdl(fourwell, fourwell_grad, fourwell_hess, BOUNDS,
                  local_optimizer="dNewton", mode="stationary_points", num_epochs=20)
    h.optimize(dask_client=client)
    final = h.get_final()
    by_class = {c: [e["x"] for e in final if e["classifier"] == c] for c in ("minimum", "saddle point", "maximum")}
    for points, expected in [(by_class["minimum"], FOUR_MINIMA), (by_class["saddle point"], FOUR_SADDLES),
                             (by_class["maximum"], [MAXIMUM])]:
        assert len(points) == len(expected)
        for target in expected:
            assert contains_point(points, target), f"missing {target}"
    assert not [e for e in final if e["classifier"] in ("zero curvature", "ERROR")]


def test_finds_minima_with_approximate_hessian_and_user_x0(client):
    h = make_hgdl(fourwell, fourwell_grad, None, BOUNDS, local_optimizer="L-BFGS-B",
                  global_optimizer="random", num_epochs=20)
    h.optimize(dask_client=client, x0=np.array([[0.9, 0.9]]))
    minima = [e["x"] for e in h.get_final() if e["classifier"] == "minimum"]
    assert contains_point(minima, [1., 1.], atol=1e-3)


def test_multiprocess_cluster_like_gpcam():
    # gpCAM's call pattern: no Hessian, x0 of shape (1, D), a default multi-process
    # Client, then get_final(). Functions are pickled to real worker processes.
    c = Client(n_workers=3, threads_per_worker=1, processes=True, dashboard_address=None)
    try:
        h = make_hgdl(fourwell, fourwell_grad, None, BOUNDS, local_optimizer="L-BFGS-B", num_epochs=10)
        h.optimize(dask_client=c, x0=np.array([0.9, 0.9]).reshape(1, -1))
        minima = [e["x"] for e in h.get_final() if e["classifier"] == "minimum"]
        assert contains_point(minima, [1., 1.], atol=1e-3)
    finally:
        c.close()


def test_constrained_run_respects_constraint(client):
    nlc = NonlinearConstraint(x0_at_least_half, 0.5, np.inf)
    h = make_hgdl(fourwell, fourwell_grad, fourwell_hess, BOUNDS,
                  num_epochs=10, constraints=(nlc,))
    h.optimize(dask_client=client)
    minima = [e["x"] for e in h.get_final() if e["classifier"] == "minimum"]
    assert contains_point(minima, [1., 1.]) and contains_point(minima, [1., -1.])
    assert all(m[0] >= 0.5 - 1e-6 for m in minima)


def test_cancel_tasks_returns_latest_and_stops(client):
    h = make_hgdl(fourwell, fourwell_grad, fourwell_hess, BOUNDS, num_epochs=10 ** 6)
    h.optimize(dask_client=client)
    wait_for_results(h)
    res = h.cancel_tasks()
    assert len(res) > 0
    final = h.get_final()  # returns once the coordinator has stopped; no error
    assert len(final) >= len(res)
    assert client.status == "running"


def test_kill_client_returns_latest_and_closes_client():
    c = make_client()
    h = make_hgdl(fourwell, fourwell_grad, fourwell_hess, BOUNDS, num_epochs=10 ** 6)
    h.optimize(dask_client=c)
    wait_for_results(h)
    assert len(h.kill_client()) > 0
    assert c.status == "closed"
    h.get_final()  # coordinator ends without raising
    assert len(h.kill_client()) > 0  # idempotent
