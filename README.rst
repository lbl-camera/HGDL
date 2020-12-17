=====
HGDL (Hybrid Global Deflated Local) Optimizer
=====

.. image:: https://img.shields.io/travis//hgdl.svg
        :target: https://travis-ci.org//hgdl

.. image:: https://img.shields.io/pypi/v/hgdl.svg
        :target: https://pypi.python.org/pypi/hgdl

* Free software: GPL license
* Documentation: (COMING SOON!) https://.github.io/hgdl.


What is this?
-------------
Algorithm:
~~~~~~~~~~
The HGDL in HGDL Optimizer stands for Hybrid Global Deflated Local Optimizer. This means:
 * Hybrid - that the algorithm uses both global and local optimizers together
 * Global - a class of optimization methods that does not use the function derivative
 * Deflated Local:
  - Local - a class of optimization methods that use the function derivative 
  - Deflated means that the gradient is modified so that previously found minima are avoided

Any global or local method can be plugged into HGDL, and several general use ones are provided, making HGDL very flexible.

How to Use:
--------------

Install:
~~~~~~~

Clone the git repo:

``git clone https://bitbucket.org/berkeleylab/hgdl.git``

There are 2 ways to install. Pip and Conda.
Pip:

``cd hgdl``

``pip install .``

Conda:
``conda create –name hgdlEnv``

``conda activate hgdlEnv``

``cd hgdl``

``conda install -y -c conda-forge wheel numpy scipy matplotlib dask pytorch distributed numba dask-mpi distributed scikit-learn jupyter jupyterlab future mpi4py``

``pip install .``

Running on a laptop:
~~~~~~~~~~~~~~~~~~~
In your code, do this:

from dask.distributed import Client
hgdl_result = HGDL(..., client=Client(), ...)

This will create a dask distributed client on your laptop that HGDL will use for parallelism

Running on a HPC:
~~~~~~~~~~~~~~~~~
This works on the LBL NERSC computer and UTK ISAAC HPC cluster, so it should work on your cluster, so long as you have mpirun as a command.

Command line:
``mpirun --np NUM_NODES dask-mpi --no-nanny --scheduler-file scheduler.json &``

In code:
``from dask.distributed import Client`` 
``client = Client(scheduler_file='scheduler.json')``
``hgdl_result = HGDL(..., client=client, ...)``

Copyright
=========

--------------

HGDL Copyright (c) 2020, The Regents of the University of California,
through Lawrence Berkeley National Laboratory (subject to receipt of any
required approvals from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this
software, please contact Berkeley Lab’s Intellectual Property Office at
IPO@lbl.gov.

NOTICE. This Software was developed under funding from the U.S.
Department of Energy and the U.S. Government consequently retains
certain rights. As such, the U.S. Government has been granted for itself
and others acting on its behalf a paid-up, nonexclusive, irrevocable,
worldwide license in the Software to reproduce, distribute copies to the
public, prepare derivative works, and perform publicly and display
publicly, and to permit others to do so.

-

