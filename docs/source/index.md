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
api/logging.md
```

```{toctree}
---
hidden: true
maxdepth: 2
caption: Examples
---
examples/rosenbrock_constrained.ipynb
examples/schwefel_constrained.ipynb
```

# HGDL

+++

## HGDL

Welcome to the documentation of the HGDL API.
HGDL is an optimization algorithm specialized in finding not only one but a diverse set of optima,
alleviating challenges of non-uniqueness that are common in modern applications such as inversion problems
and training of machine learning models.


HGDL is customized for distributed HP computing; all workers can be distributed across as many nodes or cores.
All local optimizations will then be executed in parallel.
As solutions are found, they are deflated which effectively removes those optima from the function,
so that they cannot be reidentified by subsequent local searches. For more information please have a look at the content below.

## See Also

* [Recent Paper](https://ieeexplore.ieee.org/abstract/document/9652812)
* [HGDN](https://www.sciencedirect.com/science/article/pii/S037704271730225X)
* [gpCAM](https://gpcam.readthedocs.io)
* [fvGP](https://fvgp.readthedocs.io)
