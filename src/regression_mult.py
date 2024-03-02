import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import os

np.random.seed(1995)

class Regressor:
        def score(self, Y_pred, Y_test):
            Y_pred = np.array(Y_pred).reshape((len(Y_pred),1))
            Y_test = np.array(Y_test).reshape((len(Y_test),1))
            # Calculating r_square
            rss = np.sum(np.square((Y_test - Y_pred)))
            mean = np.mean(Y_test)
            sst = np.sum(np.square(Y_test - mean))
            r_square = 1 - (rss/sst)
            return r_square
            # return 1 - ((((y_pred - y_real)**2).sum()) / (((y_real - y_real.mean())**2).sum()))


class LinearRegresorMult(Regressor):
    def __init__(self):
        self.beta = 0
    
    def fit(self, X: 'np.array', Y: 'np.array'):
        ones = np.ones((len(X), 1))
        X = np.append(ones, X, axis=1)
        Y = np.array(Y).reshape((len(Y),1))

        self.beta = np.linalg.inv(X.T @ X) @ X.T @ Y
    
    def predict(self, X):
        ones = np.ones((len(X), 1))
        X = np.append(ones, X, axis=1)
        Y_pred = X @ self.beta
        Y_pred = np.array(Y_pred).reshape((len(Y_pred),1))
        return Y_pred
    



# (xtx)-1 xt y

if __name__ == '__main__':
    print('Script:')

    data = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/data_2d.csv'), names = ['x1', 'x2', 'y'])
    data.info()

    # print(data.head())
    # data.info()

    X = data[['x1', 'x2']]
    Y = data['y']

    Y = np.array(Y).reshape((len(Y),1))

    # print(X)
    # print(Y)

    # Regression Model
    # Model
    model = LinearRegresorMult()
    Y_model = model.fit(X, Y)
    Y_pred = model.predict(X)
    print(f'Y: {Y.shape}')
    print(f'Predicciones: {Y_pred.shape}')
    r2 = model.score(Y_pred, Y)
    print(r2)
    beta = model.beta

    print(X)
    print(beta[0])
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
 
    ax.scatter(X['x1'], X['x2'], Y, label='Y',
               s=5, color="dodgerblue")
 
    ax.scatter(X['x1'], X['x2'], beta[0] + beta[1]*X['x1'] + beta[2]*X['x2'],
               label='Regression', s=5, color="orange")
 
    ax.view_init(45, 0)
    ax.legend()
 
    plt.show()

        

    
    

    





