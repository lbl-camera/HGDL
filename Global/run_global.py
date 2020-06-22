from .genetic import genetic_step
def run_global(hgdl):
    if hgdl.global_method == 'genetic':
        return genetic_step(hgdl)
    else:
        print("global method not understood")
