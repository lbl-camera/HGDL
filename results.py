import numpy as np

class Results(object):
    def __init__(self, hgdl):
        self.minima_x = np.empty((0, hgdl.k), np.float64)
        self.minima_y = np.empty(0, np.float64)
        self.genetic_x = np.empty((0, hgdl.k), np.float64)
        self.genetic_y = np.empty(0, np.float64)
        self.func = hgdl.func
        self.bestX = hgdl.bestX
    def update_minima(self, new_minima):
        minima_y = np.array([self.func(x) for x in new_minima])
        self.minima_x = np.append(self.minima_x, new_minima, 0)
        self.minima_y = np.append(self.minima_y, minima_y)
    def update_genetic(self, new_genetic):
        genetic_y = np.array([self.func(x) for x in new_genetic])
        self.genetic_x = np.append(self.genetic_x, new_genetic, 0)
        self.genetic_y = np.append(self.genetic_y, genetic_y)
    def get_all(self):
        x = np.append(self.minima_x, self.genetic_x, 0)
        y = np.append(self.minima_y, self.genetic_y)
        return x, y
    def sort(self):
        x = np.append(self.minima_x, self.genetic_x, 0)
        y = np.append(self.minima_y, self.genetic_y)
        c = np.argmin(y)
        return x[c], y[c]
    def roll_up(self):
        x, y = self.sort()
        c1, c2 = np.argsort(self.minima_y), np.argsort(self.genetic_y)
        self.minima_x, self.minima_y = self.minima_x[c1], self.minima_y[c1]
        self.genetic_x, self.genetic_y = self.genetic_x[c2], self.genetic_y[c2]
        return {
            'best_x':x,'best_y':y,
            'minima_x':self.minima_x[:self.bestX],
            'minima_y':self.minima_y[:self.bestX],
            'genetic_x':self.genetic_x[:self.bestX],
            'genetic_y':self.genetic_y[:self.bestX]
        }

