# HGDL
This is the README file for the Hybrid Global Deflated Local (HGDL) optimization algorithm

## What is this?
This is a minimization procedure written in python that uses a 
* Hybrid - both global and local minimizers
* Global - a range of global optimizers, such as genetic alg.
* Deflated - found minima are avoided by the local minimizer
* Local - local minimization method, such as Newton's method
minimization scheme to find minima in a function.
The algorithm uses global and local optimization, together with deflation to find the global
minimum along with many local optima. Of course, as always, there is no guarantee the global minimum will be found. 

## How to install
### Pip:
cd path/to/hgdl/
pip install .

### Conda
conda create --name hgdlEnv
conda activate hgdlEnv
git clone git@bitbucket.org:berkeleylab/hgdl.git
cd hgdl
conda install -y -c conda-forge --file conda\_reqs.txt
pip install .

## How to use
see the examples folder

## Running the Paper problems 
There are some extra requirements that I didn't put in bc they are only required for the paper
pip install scipy sklearn mpi4py

## Running on NERSC
pip install dask\_mpi mpi4py
dask-scheduler --scheduler-file scheduler.json
srun dask-worker --scheduler-file scheduler.json --no-scheduler 

```math
\Large c = e^{\frac{\alpha}{r^2}}
\Large b(x-x_0, r, \alpha) = c e^{\frac{-\alpha}{r^2-\sum_{i=0}^{d} (x_i-x_{0_i})^2}}
\Large \frac{\partial b} {\partial x_i} = 
b(x-x_o, r, \alpha) \frac{-2\alpha(x_i - x_{0_i})}{(r^2 - \sum_{i=0}^{d} (x_i-x_{0_i})^2)^2}
\Large \text{deflation} = \frac{1}{1-b}
\Large \frac{\partial}{\partial x_i} \text{deflation} = \frac{2}{(1-b)^2} \frac{\partial}{\partial x_i} b
\Large \frac{\frac{\partial}{\partial x_i} \text{deflation}}{\text{deflation}} = \frac{2}{1-b} \frac{\partial}{\partial x_i} b = 2 \times \text{deflation} \times \frac{\partial}{\partial x_i} b
```

 * x is the probe point
 * $x_0$ is a single, technical minima
 * minima is a list of all technical minima
 * r is the radius of deflation
 * alpha is a parameter describing the shape of the bump function.
 
