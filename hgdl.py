# coding: utf-8
#  imports

import numpy as np
#import numba as nb
#from math import ceil
#from functools import partial
#from multiprocessing import Pool

from Local import run_local
from utility import in_bounds, random_sample

class HGDL(object):
    def __init__(func, grad, hess, bounds, r=.3, alpha=.1, maxEpochs=5, num_individuals=5, maxLocal=5, numWorkers=None, bestX=5):
        """
        HGDL
            * Hybrid - uses both local and global optimization
            * G - uses global optimizer
            * D - uses deflation
            * L - uses local extremum localMethod
        Mandatory Parameters:
            * func - should return a scalar given a numpy array x
            -- note: use functools.partial if you have optional params
            * grad - gradient vector at x
            * hess - hessian array at x
            * bounds - numpy array of bounds in same format as scipy.optimize
        Optional Parameters:
            * r (0.3) - the radius of the deflation operator
            * alpha (0.1) - the alpha term of the bump function
            * maxEpochs (5) - the maximum number of epochs
            * numIndividuals (15) - the number of individuals to run
            * maxLocal (5) - the maximum number of local runs to do
            * numWorkers (logical cpu cores -1) - how many processes to use
            * bestX (5) - return the best X x's
        Returns:
            a dict of the form
            either {"success":False} if len(x) is 0
            or {"success":True, "x",x, "y",y} with the bestX x's and their y's
        """
        self.k = len(bounds)
        self.r = r
        self.alpha = alpha
        self.max_epochs = max_epochs
        self.max_local = max_local
        self.num_individuals = num_individuals
        if numWorkers is None:
            from psutil import cpu_count
            num_workers = cpu_count(logical=False)-1
        self.num_workers = num_workers
        self.results = Results(x0)
        self.bestX = bestX
    def epoch_step(self):
        genetic_x = random_sample(numIndividuals, k, bounds)
        genetic_y = np.array([func(x) for x in genetic_x])
        results_edge = np.empty((0,k))
        results_minima = np.empty((0,k))
        if numWorkers is None: numWorkers = max(cpu_count(logical=False)-1,1)
        for i in range(maxEpochs):
            newStarts = GeneticStep(genetic_x, genetic_y, bounds, numIndividuals)
            newFuncVals = np.array([func(x) for x in newStarts])
            genetic_x = np.append(genetic_x, newStarts, 0)
            genetic_y = np.append(genetic_y, newFuncVals)
            c = np.argsort(genetic_y)
            genetic_x, genetic_y = genetic_x, genetic_y
            results_edge, results_minima = deflated_local(
                    genetic_x[:numIndividuals], results_edge,
                    results_minima, grad, hess, bounds,
                    r, alpha, maxLocal, numWorkers)
            print("at epoch ",i+1,", found top 10:")
            print("edges",results_edge[:10].round(2),"\nminima",results_minima[:10].round(2))
            print("genetic",genetic_x[:10].round(2))

        func_vals_edge = np.array([func(x) for x in results_edge])
        func_vals_minima = np.array([func(x) for x in results_minima])
        c_edge, c_minima = np.argsort(func_vals_edge), np.argsort(func_vals_minima)
        results_minima, func_vals_minima = results_minima[c_minima], func_vals_minima[c_minima]
        results_edge, func_vals_edge = results_edge[c_edge], func_vals_edge[c_edge]
        if len(results_minima) < bestX:
            print("well there buckaroo, i couldn't find all ya asked for, my guy")
        return {
                "minima":results_minima, "minima_y":func_vals_minima,
                "edge":results_edge, "edge_y": func_vals_edge,
                "genetic":genetic_x, "genetic_y":genetic_y}


