from .genetic import genetic_step
from .gaussian import gaussian_step

def run_global(hgdl):
    x, y = hgdl.results.get_all()
    if hgdl.global_method == 'genetic':
        return genetic_step(hgdl, x, y)
    elif hgdl.global_method == 'gaussian':
        return gaussian_step(hgdl, x, y)
    else:
        print("global method not understood")
