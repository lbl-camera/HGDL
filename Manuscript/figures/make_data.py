"""
Data for the figures of hgdl_methodology.tex, computed with hgdl's own deflation code.
Run from the repository root:  python Manuscript/figures/make_data.py
"""
from pathlib import Path

import numpy as np

from hgdl.local_methods import bump_function as bf

out = Path(__file__).parent

# (a) two overlapping bumps and the deflation operator D = prod 1/(1 - b_i) (log scale)
x = np.linspace(-1.5, 2.5, 2001)
centers, radii = np.array([[0.], [0.8]]), np.array([1.0, 0.7])
b1 = np.array([bf.b(np.array([t]), centers[0], radii[0]) for t in x])
b2 = np.array([bf.b(np.array([t]), centers[1], radii[1]) for t in x])
D = np.array([bf.deflation_function(np.array([t]), centers, radii) for t in x])
np.savetxt(out / "bumps.dat", np.column_stack([x, b1, b2, b1 + b2, D]),
           header="x b1 b2 bsum D", comments="")

# (b) f'(x) of f = (x^2 - 1)^2 and the deflated derivative D f' with the root x = 1 deflated
x = np.linspace(-1.6, 1.6, 3201)
fp = 4 * x * (x ** 2 - 1)
xh, r = np.array([[1.0]]), np.array([0.5])
Dfp = np.array([bf.deflated_grad(np.array([t]), grad_func=lambda y: 4 * y * (y ** 2 - 1),
                                 x_defl=xh, radius=r)[0] for t in x])
Dfp = np.clip(Dfp, -60, 60)
Dfp[np.argmin(np.abs(x - 1.0))] = np.nan  # break the curve where it jumps from -inf to +inf
np.savetxt(out / "deflated_derivative.dat", np.column_stack([x, fp, Dfp]),
           header="x fp Dfp", comments="")
