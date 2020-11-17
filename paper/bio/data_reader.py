import numpy as np
import matplotlib.pyplot as plt

class data_reader(object):
    def __init__(self, seed=420):
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
        self.rng = np.random.default_rng(seed)
        self.data = np.loadtxt('data/qsar_aquatic_toxicity.csv', delimiter=';')

    def get_data(self, percent):
        self.rng.shuffle(self.data)
        mark = int(len(self.data)*percent)
        training, test = self.data[:mark], data[mark:]

        X, y = training[:,:-1], training[:,-1]
        training_mean_x = np.mean(X, 0)
        X -= training_mean_x
        training_std_x = np.std(X, 0)
        X /= training_std_x
        training_mean_y = np.mean(y)
        y -= training_mean_y
        training_std_y = np.std(y)
        y /= training_std_y

        X_test, y_test = test[:,:-1], test[:,-1]
        X_test -= training_mean_x
        X_test /= training_std_x
        y_test -= training_mean_y
        y_test /= training_std_y

        X_train, y_train = X, y
        X_test,  y_test = X_test, y_test
        return (X_train, y_train), (X_test, y_test)
    """
    def R2(self,gp):
        for x, y, name in zip(
                [self.X_train, X_test],
                [y_train, y_test],
                ["train", "test"]):
            print(name,'---------------------------------------')
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
            plt.errorbar(useless, residual, yerr=y_std, fmt='.')
            plt.show()

            print('calculating percentage covered when 1,2,3,&4 sigma away')
            for i in range(1,5):
                within = np.logical_and(y-i*y_std<=y_pred, y_pred<=y+i*y_std)
                print('sigma:',i,(np.sum(within)/len(within)).round(3))

        press = np.sum(np.power(y_test-gp.predict(X_test),2))
        ss_tr = np.sum(np.power(y_test-np.mean(y_train),2))
        ss_ext = np.sum(np.power(y_test-np.mean(y_test),2))
        tss = np.sum(np.power(y_train-np.mean(y_train),2))
        f1 = 1. - press/ss_tr
        f2 = 1. - press/ss_ext
        f3 = 1. - press * self.X_train.shape[1] / (tss*X_test.shape[1])
        for f in ['f1','f2','f3']:
            print(f,':', eval(f).round(2))

        pass


    def Q2_cv(self,gp):
        for x, y, name in zip(
                [self.X_train, X_test],
                [y_train, y_test],
                ["train", "test"]):
            print(name,'---------------------------------------')
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
            plt.errorbar(useless, residual, yerr=y_std, fmt='.')
            plt.show()

            print('calculating percentage covered when 1,2,3,&4 sigma away')
            for i in range(1,5):
                within = np.logical_and(y-i*y_std<=y_pred, y_pred<=y+i*y_std)
                print('sigma:',i,(np.sum(within)/len(within)).round(3))

        press = np.sum(np.power(y_test-gp.predict(X_test),2))
        ss_tr = np.sum(np.power(y_test-np.mean(y_train),2))
        ss_ext = np.sum(np.power(y_test-np.mean(y_test),2))
        tss = np.sum(np.power(y_train-np.mean(y_train),2))
        f1 = 1. - press/ss_tr
        f2 = 1. - press/ss_ext
        f3 = 1. - press * self.X_train.shape[1] / (tss*X_test.shape[1])
        for f in ['f1','f2','f3']:
            print(f,':', eval(f).round(2))

        pass
    """
    def Q2_ext(self, train, test):
        X_train, y_train = train
        x_test, y_test = test
        press = np.sum(np.power(y_test-gp.predict(X_test),2))
        ss_tr = np.sum(np.power(y_test-np.mean(y_train),2))
        ss_ext = np.sum(np.power(y_test-np.mean(y_test),2))
        tss = np.sum(np.power(y_train-np.mean(y_train),2))
        #f1 = 1. - press/ss_tr
        #f2 = 1. - press/ss_ext
        f3 = 1. - press * X_train.shape[1] / (tss*X_test.shape[1])
        #for f in ['f1','f2','f3']:
            #print(f,':', eval(f).round(2))
        return f3

