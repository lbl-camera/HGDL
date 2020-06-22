
def alreadyFound(newMinima, oldMinima, radius_squared):
    """
    check if the new minimum is within range of the old ones
    """
    c = oldMinima - newMinima
    return (np.sum(c*c,1)<radius_squared).any()

def deflated_local(starts, results_edge, results_minima, gradient, hessian, bounds, r,alpha, maxLocal, numWorkers):
    with Pool(numWorkers) as workers:
        for j in range(maxLocal):
            percent_none = 0.
#            tmp_results = workers.imap_unordered(
#                    partial(newton,minima=results_minima,
#                        gradient=gradient,hessian=hessian,bounds=bounds,r=r,alpha=alpha),
#                    starts)
            func = partial(newton,minima=results_minima,
                        gradient=gradient,hessian=hessian,bounds=bounds,r=r,alpha=alpha)
#            for x in tmp_results:
            for i in range(len(starts)):
                x = func(starts[i])
                if not x["success"]:
                    percent_none += 1./starts.shape[0]
                else:
                    if alreadyFound(x["x"], results_minima, r**2):
                        percent_none += 1./starts.shape[0]
                    else:
                        if x["edge"]:
                            results_edge = np.append(results_edge, x["x"].reshape(1,-1), 0)
                        else:
                            results_minima = np.append(results_minima, x["x"].reshape(1,-1), 0)
                if percent_none > 0.2:
                    return results_edge, results_minima
    return results_edge, results_minima


