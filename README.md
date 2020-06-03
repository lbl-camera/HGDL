# MBSB-HGDL
Wow, what a messy title. I promise it's way simpler than that.

## What is this?
This is a minimization procedure written in python that uses a 
* Hybrid - both global and local minimizers
* Global - a range of global optimizers, such as genetic alg.
* Deflated - found minima are avoided by the local minimizer
* Local - local minimization method, such as Newton's method
minimization scheme to find minima in a function.
This tries to find several minima, so if you have only one, lucky you - you don't want this.
This uses a deflated minimization method, so if you don't have the gradient, you're up a creek, since finite differencing is not great.

