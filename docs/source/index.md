---
banner: _static/landing.png
banner_height: "40vh"
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

## HGDL

Welcome to the documentation of the HGDL API.
HGDL is an optimization algorithms specialized to find not only one but a diverse set of optima,
alleviating challenges of non-uniqueness that is common in modern applications of inversion problems
and training machine learning models.


HGDL is customized for distributed HP computing; all workers can be distributed across as many nodes.
All local optimizations will then be executed in parallel.
As solutions are found, they are deflated which effectively removes those optima from the function,
so that they cannot be identified again. For more information have a look at the links below.

## See Also

* [Paper](https://ieeexplore.ieee.org/abstract/document/9652812)
* [HGDN](https://www.sciencedirect.com/science/article/pii/S037704271730225X)
* [gpCAM](https://gpcam.readthedocs.io)
* [fvGP](https://fvgp.readthedocs.io)
