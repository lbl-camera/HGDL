```{toctree}
:hidden:
:maxdepth: 1

api/overview.md
examples/index.md
```

# HGDL — Hybrid Global Deflated Local Optimization

HGDL is an optimization algorithm specialized in finding not only one but a diverse set of optima,
alleviating challenges of non-uniqueness that are common in modern applications such as inversion problems
and training of machine learning models.
It provides the distributed, asynchronous, constrained optimization behind
[fvGP](https://fvgp.readthedocs.io) and [gpCAM](https://gpcam.readthedocs.io).

HGDL is customized for distributed HPC; all workers can be distributed across as many nodes or cores,
and all local optimizations are executed in parallel.
As solutions are found, they are deflated, which effectively removes those optima from the function
so that they cannot be reidentified by subsequent local searches.
The result is a growing, sorted list of unique optima that can be queried while the optimization runs.

## See Also

* [Recent Paper](https://ieeexplore.ieee.org/abstract/document/9652812)
* [HGDN](https://www.sciencedirect.com/science/article/pii/S037704271730225X)
* [fvGP](https://fvgp.readthedocs.io)
* [gpCAM](https://gpcam.readthedocs.io)
