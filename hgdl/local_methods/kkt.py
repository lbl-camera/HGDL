"""
First- and second-order optimality (KKT) conditions at a point returned by a
constrained local optimizer.

On an active constraint grad f is not zero; at an optimum it is balanced by the
gradients of the active constraints, grad f = J_A^T w, with the Lagrange multipliers
w >= 0 for an active lower bound (lb <= c(x)), w <= 0 for an active upper bound and w
free for an equality. Whether the point is a minimum, and its deflation radius, are
decided by the curvature of the Lagrangian L = f - w^T c on the tangent space of the
strongly active constraints (those with a nonzero multiplier).

The multipliers are computed here from grad f and the constraint Jacobian instead of
being taken from the optimizer, so the test is the same for every local optimizer.
"""
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import LinearConstraint, NonlinearConstraint, lsq_linear

# a constraint is active within this distance of its bound, relative to max(1, |bound|);
# the same margin is allowed for infeasibility
ACTIVE_TOL = 1e-6
# an active constraint with a smaller multiplier does not restrict the tangent space
MULTIPLIER_TOL = 1e-8
# steps of the central differences for Jacobians and Hessians the user did not provide:
# first differences of values or of a Jacobian, and second differences of values
FD_STEP = 1e-6
FD2_STEP = 1e-4


def _dense(m, n):
    """A dense (.. x n) array from a numpy array, sparse matrix or LinearOperator."""
    if hasattr(m, "toarray"):
        return m.toarray()
    if not isinstance(m, np.ndarray) and hasattr(m, "dot"):
        return m.dot(np.eye(n))
    return np.asarray(m, dtype=float)


def _central_differences(func, x):
    """Jacobian (len(func(x)) x len(x)) of a vector-valued function."""
    columns = []
    for i in range(len(x)):
        h = FD_STEP * max(1., abs(x[i]))
        xp, xm = np.array(x, dtype=float), np.array(x, dtype=float)
        xp[i] += h
        xm[i] -= h
        columns.append((np.atleast_1d(func(xp)) - np.atleast_1d(func(xm))) / (2. * h))
    return np.column_stack(columns)


def _second_differences(func, x):
    """Hessian of a scalar function from its values only."""
    n = len(x)
    h = FD2_STEP * np.maximum(1., np.abs(x))
    hess = np.empty((n, n))
    for i in range(n):
        for j in range(i, n):
            total = 0.
            for si, sj, sign in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)):
                y = np.array(x, dtype=float)
                y[i] += si * h[i]
                y[j] += sj * h[j]
                total += sign * func(y)
            hess[i, j] = hess[j, i] = total / (4. * h[i] * h[j])
    return hess


class _NonlinearBlock:
    """lb <= fun(x, *args) <= ub, with optional user Jacobian and Hessian."""

    def __init__(self, fun, lb, ub, jac=None, hess=None, args=()):
        self.fun, self.lb, self.ub = fun, lb, ub
        self.jac = jac if callable(jac) else None
        self.hess = hess if callable(hess) else None
        self.args = args

    def values(self, x):
        return np.atleast_1d(np.asarray(self.fun(x, *self.args), dtype=float))

    def jacobian(self, x):
        if self.jac is None:
            return _central_differences(self.values, x)
        return _dense(self.jac(x, *self.args), len(x)).reshape(-1, len(x))

    def weighted_hessian(self, x, w):
        """sum_i w_i * Hessian of constraint i"""
        if self.hess is not None:
            return _dense(self.hess(x, w), len(x))
        if self.jac is not None:
            h = _central_differences(lambda y: self.jacobian(y).T @ w, x)
            return 0.5 * (h + h.T)
        # differencing a difference quotient would lose half the digits
        return _second_differences(lambda y: w @ self.values(y), x)


class _LinearBlock:
    """lb <= A x <= ub"""

    def __init__(self, A, lb, ub):
        self.A = np.atleast_2d(_dense(A, np.shape(A)[-1]))
        self.lb, self.ub = lb, ub

    def values(self, x):
        return self.A @ x

    def jacobian(self, x):
        return self.A

    def weighted_hessian(self, x, w):
        return np.zeros((len(x), len(x)))


def normalize_constraints(constraints):
    """
    The constraints in any form scipy.optimize.minimize accepts (NonlinearConstraint,
    LinearConstraint, dicts with 'type', 'fun' and optional 'jac', 'args'; one or a
    sequence) as a list of blocks lb <= c(x) <= ub.
    """
    if isinstance(constraints, (dict, NonlinearConstraint, LinearConstraint)):
        constraints = [constraints]
    blocks = []
    for c in constraints:
        if isinstance(c, NonlinearConstraint):
            blocks.append(_NonlinearBlock(c.fun, c.lb, c.ub, jac=c.jac, hess=c.hess))
        elif isinstance(c, LinearConstraint):
            blocks.append(_LinearBlock(c.A, c.lb, c.ub))
        elif isinstance(c, dict):
            if c["type"] == "eq":
                lb, ub = 0., 0.
            elif c["type"] == "ineq":
                lb, ub = 0., np.inf
            else:
                raise ValueError(f"unknown constraint type {c['type']!r}")
            blocks.append(_NonlinearBlock(c["fun"], lb, ub, jac=c.get("jac"), args=c.get("args", ())))
        else:
            raise TypeError(f"unsupported constraint {c!r}")
    return blocks


def _state(x, grad, blocks):
    """
    Constraint values and Jacobian at x, the active set and the multipliers that best
    balance grad f (bounded least squares, signs as in the module docstring).
    """
    n = len(x)
    values = [b.values(x) for b in blocks]
    sizes = [len(v) for v in values]
    c = np.concatenate(values)
    J = np.vstack([b.jacobian(x) for b in blocks]).reshape(len(c), n)
    lb = np.concatenate([np.broadcast_to(np.asarray(b.lb, dtype=float), (s,)) for b, s in zip(blocks, sizes)])
    ub = np.concatenate([np.broadcast_to(np.asarray(b.ub, dtype=float), (s,)) for b, s in zip(blocks, sizes)])

    with np.errstate(invalid="ignore"):
        tol_lb = ACTIVE_TOL * np.maximum(1., np.abs(lb))
        tol_ub = ACTIVE_TOL * np.maximum(1., np.abs(ub))
        feasible = bool(np.all(np.isfinite(c)) and np.all(c >= lb - tol_lb) and np.all(c <= ub + tol_ub))
        equality = lb == ub
        at_lb = ~equality & np.isfinite(lb) & (c - lb <= tol_lb)
        at_ub = ~equality & np.isfinite(ub) & (ub - c <= tol_ub)
    active = equality | at_lb | at_ub

    w = np.zeros(len(c))
    if np.any(active):
        lo = np.where(at_lb, 0., -np.inf)[active]
        hi = np.where(at_ub, 0., np.inf)[active]
        w[active] = lsq_linear(J[active].T, grad, bounds=(lo, hi), method="bvls").x
    strong = equality | (active & (np.abs(w) > MULTIPLIER_TOL))
    # the bound each strongly active constraint sits on
    target = np.where(at_ub, ub, lb)
    return c, J, w, strong, target, sizes, feasible


def _lagrangian_hessian(x, hess, blocks, w, sizes):
    h = 0.5 * (hess + hess.T)
    for b, wb in zip(blocks, np.split(w, np.cumsum(sizes)[:-1])):
        if np.any(wb != 0.):
            h = h - b.weighted_hessian(x, wb)
    return h


def _residual(grad, c, J, w, strong, target):
    return np.linalg.norm(np.concatenate([grad - J.T @ w, (c - target)[strong]]))


def newton_polish(x, grad_func, hess_func, blocks, args=(), max_steps=5):
    """
    A few Newton steps on the KKT system of the strongly active constraints,
    grad f - J_S^T w = 0, c_S(x) = bound. SLSQP stops on the change of f, which near
    an optimum leaves |grad L| around the square root of its tolerance; this takes it
    to machine precision. It is only tried close to a KKT point (|grad L| < 1e-3), and
    a step is kept only if it reduces the KKT residual and stays feasible.
    """
    x = np.array(x, dtype=float)
    grad = np.asarray(grad_func(x, *args), dtype=float)
    c, J, w, strong, target, sizes, feasible = _state(x, grad, blocks)
    residual = _residual(grad, c, J, w, strong, target)
    if not feasible or not np.isfinite(residual) or np.linalg.norm(grad - J.T @ w) >= 1e-3:
        return x
    for _ in range(max_steps):
        if residual < 1e-12:
            break
        n, k = len(x), int(np.sum(strong))
        h = _lagrangian_hessian(x, np.asarray(hess_func(x, *args), dtype=float), blocks, w, sizes)
        Js = J[strong]
        K = np.block([[h, -Js.T], [Js, np.zeros((k, k))]])
        F = np.concatenate([grad - J.T @ w, (c - target)[strong]])
        step = np.linalg.lstsq(K, -F, rcond=None)[0]
        x_new = x + step[:n]
        grad_new = np.asarray(grad_func(x_new, *args), dtype=float)
        state = _state(x_new, grad_new, blocks)
        residual_new = _residual(grad_new, *state[:5])
        if not state[-1] or not residual_new < residual:
            break
        x, grad, residual = x_new, grad_new, residual_new
        c, J, w, strong, target, sizes, feasible = state
    return x


def kkt_conditions(x, grad, hess, blocks):
    """
    The KKT conditions of min f subject to the constraint blocks, at x.

    input:
        x: the point
        grad, hess: grad f and the Hessian of f at x (true, not deflated)
        blocks: from normalize_constraints
    return:
        lagrangian_gradient: grad f - J_A^T w, zero at a KKT point
        eigenvalues: of the Lagrangian Hessian on the tangent space of the strongly
            active constraints; empty if they fix x (as many as dimensions)
        feasible: bool
    """
    c, J, w, strong, target, sizes, feasible = _state(x, grad, blocks)
    lagrangian_gradient = grad - J.T @ w
    Z = null_space(J[strong]) if np.any(strong) else np.eye(len(x))
    if Z.shape[1] == 0:
        return lagrangian_gradient, np.empty(0), feasible
    reduced = Z.T @ _lagrangian_hessian(x, hess, blocks, w, sizes) @ Z
    return lagrangian_gradient, np.linalg.eigvalsh(0.5 * (reduced + reduced.T)), feasible
