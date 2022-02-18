---
banner: _static/landing.png
banner_height: "100vh"
---

```{toctree}
---
hidden: true
maxdepth: 2
caption: API
---
api/HGDL.md
```

```{toctree}
---
hidden: true
maxdepth: 2
caption: Examples
---
examples/rosenbrock.md
```

# The HGDL Package: Hybrid HPC Distributed Optimization

+++

The problem HGDL is attempting to solve is:

+++

$
\Large argmin_x f(x)
$

+++

For HGDL this happens by using local optimization, global optimization and deflation to find many unique optima.
Deflation has the capability to remove identified optima from the function to avoid the high costs of reidentification.
This is done by applying a deflation operator, based on the, so-called, bump function.

+++

$
\Large b(x-x_0, r, \alpha) = c e^{\frac{-\alpha}{r^2-\sum_{i=0}^{d} (x_i-x_{0_i})^2}}
$

+++

$
\Large \text{deflation} = \frac{1}{1-b}
$

+++

 * x is the probe point
 * $x_0$ is a single, technical minima
 * minima is a list of all technical minima
 * r is the radius of deflation
 * alpha is a parameter describing the shape of the bump function.

HGDL is customized for distributed HPC computing; all workers can be distributed across many nodes.
All local optimizations will then happen in parallel.

## See Also

* [gpCAM](https://gpcam.readthedocs.io)
* [fvGP](https://fvgp.readthedocs.io)
 
