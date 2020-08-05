import numpy as np
import matplotlib.pyplot as plt

class data_reader(object):
    def __init__(self, seed):
        """
        https://archive.ics.uci.edu/ml/datasets/QSAR+aquatic+toxicity

        1) TPSA(Tot)
        2) SAacc
        3) H-050
        4) MLOGP
        5) RDCHI
        6) GATS1p
        7) nN
        8) C-040
        9) quantitative response, LC50 [-LOG(mol/L)]
        """

        rng = np.random.default_rng(seed)
        data = np.loadtxt('data/qsar_aquatic_toxicity.csv', delimiter=';')
        rng.shuffle(data)
        mark = int(len(data)*.8)
        training, test = data[:mark], data[mark:]

        X, y = training[:,:-1], training[:,-1]
        self.training_mean_x = np.mean(X, 0)
        X -= self.training_mean_x
        self.training_std_x = np.std(X, 0)
        X /= self.training_std_x
        self.training_mean_y = np.mean(y)
        y -= self.training_mean_y
        self.training_std_y = np.std(y)
        y /= self.training_std_y

        X_test, y_test = test[:,:-1], test[:,-1]
        X_test -= self.training_mean_x
        X_test /= self.training_std_x
        y_test -= self.training_mean_y
        y_test /= self.training_std_y

        self.X_train, self.y_train = X, y
        self.X_test, self.y_test = X_test, y_test

    def get_training(self):
        return self.X_train, self.y_train
    def get_test(self):
        return self.X_test, self.y_test

    def coefficient_of_determination(self, y_true, y_pred):
        """
        use wikipedia formalism
        """
        y_hat = np.mean(y_true)
        ss_res = np.sum(np.power(y_true-y_pred,2))
        ss_tot = np.sum(np.power(y_true-y_hat,2))
        return 1.-ss_res/ss_tot
    def info(self,x,y,gp):
        y_pred, y_cov = gp.predict(x, return_cov=True)
        y_std = np.sqrt(np.diag(y_cov))
        residual = y-y_pred

        # make histogram
        plt.hist(residual, bins=30)
        plt.title('residual: y-y_pred')
        plt.show()

        # make scatter plots
        useless = np.arange(len(y))
        plt.title('residual: y-y_pred')
        plt.scatter(useless, residual)
        plt.show()

        print('calculating percentages that are 1,2,3,&4 sigma away')
        for i in range(1,5):
            within = np.logical_and(y-i*y_std<=y_pred, y_pred<=y+i*y_std)
            print('sigma:',i,(np.sum(within)/len(within)).round(3))
        print('coeff. of det.',self.coefficient_of_determination(y,y_pred))

