import jax
import jax.numpy as np

def schwefel(x):
    print('working on ',x) 
    return 418.9829*len(x) - np.sum(x*np.sin(np.sqrt(np.abs(x))))


def main():
    schwefel_func = jax.jit(schwefel)
    grad = jax.jit(jax.grad(schwefel_func))
    hess = jax.jit(jax.jacfwd(jax.jacrev(schwefel_func)))
    import numpy as onp
    b = onp.array([[-500.,500.],[-500.,500]])
    x0 = 420.96
    print(hess(x0*np.ones(2)))
    print(grad(x0*np.ones(2)))
    print("schwefel minimum @ ",x0 * np.ones(2),"  ", schwefel(x0*np.ones(2)))

    from hgdl import HGDL
    hgdl = HGDL(schwefel_func, grad, hess, b, max_epochs=10, num_workers=20, bestX = 50, max_local = 10, r = 30)
    from time import sleep
    for i in range(10):
        print(hgdl.get_best())
        sleep(10)
    print(hgdl.get_final())
    '''

    x, y = np.arange(-420,420.,7), np.arange(-420,420.,7)
    xx, yy = np.meshgrid(x,y)

    z = np.array([[schwefel(np.array([a,b])) for a,b in zip(c,d)] for c,d in zip(xx,yy)])
    plt.contourf(x,y,z)
    plt.show()

    grad = jax.jit(jax.grad(schwefel))
    z = np.array([[grad(np.array([a,b])) for a,b in zip(c,d)] for c,d in zip(xx,yy)])
    plt.contourf(x,y,z[:,:,0])
    plt.show()
    plt.contourf(x,y,z[:,:,1])

    plt.show()
    '''


if __name__ == '__main__':
    main()
