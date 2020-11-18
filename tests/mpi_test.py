from dask_mpi import initialize
from dask.distributed import Client
initialize()
c = Client()
print(c)
