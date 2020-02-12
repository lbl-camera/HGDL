# MBSB-HGDL
Wow, what a messy title. I promise it's way simpler than that.

## To Do List:
    * unit tests - how to do them?
    * example of using HXDY with a custom function
    * example of using HXYDY with args    
    
## What is this?
This is a minimization procedure written in python that uses a 
* Hybrid - both global and local minimizers
* Global - a range of global optimizers, such as genetic alg.
* Deflated - found minima are avoided by the local minimizer
* Local - local minimization method, such as Newton's method
minimization scheme to find minima in a function.
This tries to find several minima, so if you have only one, lucky you - you don't want this.
This uses a deflated minimization method, so if you don't have the gradient, you're up a creek, since finite differencing is not great.

## How Do I use this?
You can import from the python file directly or run the jupyter notebook to import the functions.

## Examples:
There are a thousand and one different use cases for this. If you are thinking, "Oh jeez, I'd like to use this, but I want to be able to have custom functionality X", then I would recommend playing around in the Examples.ipynb. I show how to put in custom stopping conditions, how to make the bounds, and what good conditions are. I try to provide reasonable defaults, but obviously, all that can be overidden for different cases

## Hooks:
added this before every commit: "jupyter nbconvert --to python *.ipynb" to keep .py up to date

