import numpy as np
from dask.distributed import Client, as_completed

def my_code(func, x, client):
    # x is not big, but the func calls a big array 
    # is there a way to scatter func? 
    futures = client.map(func, x)
    # just does some stupid work 
    max_val = -1
    for res in as_completed(futures):
        if res.result() > max_val:
            max_val = res.result()
    return max_val

def user_code():
    big_data = np.random.random((10**3,10**3))
    def big_func(x):
        if np.sum(big_data)<0: print('this is just to call the big array')
        return x**2
    client = Client()
    print('dashboard:',client.dashboard_link)
    best = my_code(big_func, range(101), client)
    print('best:',best)
    client.close()

def main():
    user_code()

if __name__ == "__main__":
    main()
