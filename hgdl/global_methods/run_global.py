from .genetic import genetic_step
from .gaussian import gaussian_step

def run_global(info):
    x, y = info.results.get_all()
    # check for nan vals, i think they're sneaking in
    print('isnan before check?', np.isnan(x).any(), np.isnan(y).any())
    mask = np.logical_or(np.isnan(x).any(axis=1), np.isnan(y))
    x, y = x[~mask], y[~mask]
    # check again
    print('isnan after check?', np.isnan(x).any(), np.isnan(y).any())

    if info.global_method == 'genetic':
        return genetic_step(info, x, y)
    elif info.global_method == 'gaussian':
        return gaussian_step(info, x, y)
    # remember you can use *info.global_args 
    #   and **info.global_kwargs
    elif callable(info.global_method):
        return info.global_method(
                x, y, *info.global_args, **info.global_kwargs)
    elif info.global_method == None:
        return []
    # elif info.global_method == 'my_custom_name':
    #   return my_global_method(x, y, *info.global_args, **info.global_kwargs)
    else:
        print("global method not understood")
