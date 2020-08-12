import numpy as np

class Results(object):
    def __init__(self, info):
        self.func = info.func
        self.bestX = info.bestX
        self.N = info.num_individuals
        self.minima_x = np.empty((0, info.k), np.float64)
        self.minima_y = np.empty(0, np.float64)
        self.global_x = np.empty((0, info.k), np.float64)
        self.global_y = np.empty(0, np.float64)
    def update_minima(self, new_minima):
        minima_y = np.array([self.func(x) for x in new_minima])
        self.minima_x = np.append(self.minima_x, new_minima, 0)
        self.minima_y = np.append(self.minima_y, minima_y)
    def update_global(self, new_global):
        global_y = np.array([self.func(x) for x in new_global])
        self.global_x = np.append(self.global_x, new_global, 0)
        self.global_y = np.append(self.global_y, global_y)
    def get_all(self):
        x = np.append(self.minima_x, self.global_x, 0)
        y = np.append(self.minima_y, self.global_y)
        return x, y

    def best(self, n=None):
        if n is None: n = self.bestX
        result = {}
        # get best
        x, y = self.get_all()
        c = np.argmin(y)
        result['best'] = (x[c], y[c])
        # sorted minima
        c = np.argsort(self.minima_y)
        result['minima'] = (self.minima_x[c][:n], self.minima_y[c][:n])
        # sorted globals 
        c = np.argsort(self.global_y)
        result['global'] = (self.global_x[c][:n], self.global_y[c][:n])

        return result

