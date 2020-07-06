import dask.distributed

def p():
    print('hi')
p()
cluster = dask.distributed.LocalCluster(dashboard_address=0)
client = dask.distributed.Client(cluster)
dask.config.set({'scheduler.work-stealing': False})
p()

