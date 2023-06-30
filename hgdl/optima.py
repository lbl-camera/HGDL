import numpy as np

from loguru import logger


class optima:
    """
    stores all results and adaptations of it
    """

    def __init__(self, dim_x, max_optima):
        """
        input:
        -----
            dim ... the dimensionality of the space
            max_optima ... maximum number of stored optima
        """

        self.dim_x = dim_x
        self.max_optima = max_optima
        self.list = []

    ####################################################

    def make_optima_list_entry(self, x, f, classifier, eigs, grad, grad_norm, r):
        list_entry = {"x": x,
                      "f(x)": f,
                      "classifier": classifier,
                      "Hessian eigvals": eigs,
                      "df/dx": grad,
                      "|df/dx|": grad_norm,
                      "radius": r}

        return list_entry

    def fill_in_optima_list(self, res):
        x, f, g, eig, r, local_success = res[0], res[1], res[2], res[3], res[4], res[5]
        if not np.any(local_success) and len(self.list) == 0: local_success[:] = True
        clean_indices = np.where(np.asarray(local_success) == True)[0]
        if len(clean_indices) == 0: return self.list

        clean_x = x[clean_indices]
        clean_f = f[clean_indices]
        clean_g = g[clean_indices]
        clean_eig = eig[clean_indices]
        clean_radii = r[clean_indices]
        classifier = []
        ##making the classifier
        for i in range(len(clean_x)):
            if any(clean_g[i] > 1e-3):
                classifier.append("degenerate")
            elif any(abs(clean_eig[i]) < 10e-6):
                classifier.append("zero curvature")
            elif all(clean_eig[i] > 0.0):
                classifier.append("minimum")
            elif all(clean_eig[i] < 0.0):
                classifier.append("maximum")
            elif abs(np.max(np.sign(clean_eig[i])) - np.min(np.sign(clean_eig[i]))) == 2:
                classifier.append("saddle point")
            else:
                classifier.append("ERROR")

        new_optima_list = []
        for i in range(len(clean_x)):
            new_optima_list.append(
                self.make_optima_list_entry(clean_x[i], clean_f[i], classifier[i], clean_eig[i], clean_g[i],
                                            np.linalg.norm(clean_g[i]), clean_radii[i]))
        optima_list = self.list + new_optima_list

        def find_f(d): return d["f(x)"]

        optima_list.sort(key=find_f)
        self.list = optima_list[0:self.max_optima]
        return self.list

    def get_minima(self, n):
        try:
            minima_list = [entry for entry in self.list if entry["classifier"] == "minimum"]
            minima_list = minima_list[0:min(n, len(minima_list))]
            return minima_list
        except:
            logger.debug("no minima available in the optima_list")
            return None

    ####################################################
    def get_maxima(self, n):
        try:
            maxima_list = [entry for entry in self.list if entry["classifier"] == "maximum"]
            maxima_list = maxima_list[0:min(n, len(maxima_list))]
            return maxima_list
        except:
            logger.debug("no maxima available in the optima_list")
            return None

    ####################################################
    def get_deflation_points(self, n):
        try:
            defl_list = [entry for entry in self.list if
                         entry["classifier"] == "maximum" or entry["classifier"] == "minimum" or entry[
                             "classifier"] == "saddle point"]
            defl_x = [entry["x"] for entry in defl_list]
            defl_f = [entry["f(x)"] for entry in defl_list]
            defl_r = [entry["radius"] for entry in defl_list]

            return defl_x, defl_f, defl_r
        except Exception as e:
            logger.debug("no deflation points available in the optima_list")
            return [], [], []

#########################################################
#########################################################
