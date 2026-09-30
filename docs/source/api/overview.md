# API Reference

```{toctree}
:hidden:
:maxdepth: 1

HGDL.md
logging.md
```

HGDL is an API for asynchronous, HPC-distributed constrained optimization that returns
a growing, sorted list of unique optima rather than a single answer.

The [HGDL](HGDL.md) class presents the complete functionality of the package:
construct it with a function, its gradient and the bounds, call `optimize()` with a
[dask](https://docs.dask.org) client, and query the optima with `get_latest()` while
the optimization runs or `get_final()` once it is done.

HGDL is the optimizer used by [fvGP](https://fvgp.readthedocs.io) for training and by
[gpCAM](https://gpcam.readthedocs.io) for acquisition-function optimization.

See [Logging](logging.md) for how to observe what the workers are doing.
