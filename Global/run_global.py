from .genetic import genetic_step
from .gaussian import gaussian_step

def run_global(hgdl):
    if hgdl.global_method == 'genetic':
        return genetic_step(hgdl)
    elif hgdl.global_method == 'gaussian':
        return gaussian_step(hgdl)
    else:
        print("global method not understood")
