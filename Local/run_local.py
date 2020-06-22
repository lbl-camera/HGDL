from .newton import newton

def run_local(hgdl):
    if hgdl.local_method == 'my_newton':
        return newton(hgdl)
    else:
        print("local method not understood")
