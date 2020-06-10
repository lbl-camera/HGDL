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

