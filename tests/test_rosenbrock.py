import numpy as np
from hgdl.hgdl import HGDL as hgdl
from scipy.optimize import rosen, rosen_der, rosen_hess


def test_rosebrock():
    from time import sleep
    print('this will create an hgdl object, sleep for 3'
          ' seconds, get the best result, sleep for 3 seconds,'
          'then get the final result.\n'
          'working on the epochs should happend even during sleeping\n'
          )
    from scipy.optimize import NonlinearConstraint
    def g1(x): return (np.linalg.norm(x) ** 2 / 10.0) - 2.0

    nlc = NonlinearConstraint(g1, -np.inf, 0)
    bounds = np.array([[-2, 2], [-2, 2]])
    a = hgdl(rosen, rosen_der, bounds, hess=rosen_hess,
             global_optimizer="random",
             local_optimizer="dNewton",
             number_of_optima=30000,
             args=(), num_epochs=1000,
             constraints=(nlc,)
             )
    x0 = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1], size=(20, 2))
    a.optimize(x0=x0)

    print("main thread submitted HGDL and will now sleep for 5 seconds")
    sleep(5)
    print("main thread asks for solutions:")
    print(a.get_latest())
    print("main sleeps for another 3 seconds")
    sleep(3)
    print("main thread kills optimization")
    res = a.kill_client()
    # res = a.cancel_tasks()
    print("hgdl was killed but I am waiting 2s")
    sleep(2)
    print("")
    print("")
    print("")
    print("")
    print(res)


if __name__ == "__main__":
    test_rosebrock()
