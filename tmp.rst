HGDL
====

This is the README file for the Hybrid Global Deflated Local (HGDL)
optimization algorithm

What is this?
-------------

This is a minimization procedure written in python that uses a \* Hybrid
- both global and local minimizers \* Global - a range of global
optimizers, such as genetic alg. \* Deflated - found minima are avoided
by the local minimizer \* Local - local minimization method, such as
Newton’s method minimization scheme to find minima in a function. The
algorithm uses global and local optimization, together with deflation to
find the global minimum along with many local optima. Of course, as
always, there is no guarantee the global minimum will be found.

How to install
--------------

Pip:
~~~~

cd path/to/hgdl/ pip install .

Conda
~~~~~

conda create –name hgdlEnv conda activate hgdlEnv git clone
git@bitbucket.org:berkeleylab/hgdl.git cd hgdl conda install -y -c
conda-forge –file conda_reqs.txt pip install .

How to use
----------

see the examples folder

Running the Paper problems
--------------------------

There are some extra requirements that I didn’t put in bc they are only
required for the paper pip install scipy sklearn mpi4py

Running on NERSC
----------------

pip install dask_mpi mpi4py dask-scheduler –scheduler-file
scheduler.json srun dask-worker –scheduler-file scheduler.json
–no-scheduler

.. code:: math

   \Large c = e^{\frac{\alpha}{r^2}}
   \Large b(x-x_0, r, \alpha) = c e^{\frac{-\alpha}{r^2-\sum_{i=0}^{d} (x_i-x_{0_i})^2}}
   \Large \frac{\partial b} {\partial x_i} = 
   b(x-x_o, r, \alpha) \frac{-2\alpha(x_i - x_{0_i})}{(r^2 - \sum_{i=0}^{d} (x_i-x_{0_i})^2)^2}
   \Large \text{deflation} = \frac{1}{1-b}
   \Large \frac{\partial}{\partial x_i} \text{deflation} = \frac{2}{(1-b)^2} \frac{\partial}{\partial x_i} b
   \Large \frac{\frac{\partial}{\partial x_i} \text{deflation}}{\text{deflation}} = \frac{2}{1-b} \frac{\partial}{\partial x_i} b = 2 \times \text{deflation} \times \frac{\partial}{\partial x_i} b

-  x is the probe point
-  :math:`x_0` is a single, technical minima
-  minima is a list of all technical minima
-  r is the radius of deflation
-  alpha is a parameter describing the shape of the bump function.

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

--------------
