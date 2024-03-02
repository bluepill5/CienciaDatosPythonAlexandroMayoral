import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import os

np.random.seed(1995)

class Regressor:
        def score(self, y_pred, y_real):
             return 1 - ((((y_pred - y_real)**2).sum()) / (((y_real - y_real.mean())**2).sum()))


class LinearRegresor(Regressor):
    def __init__(self):
        self.alpha = 0
        self.beta = 0
    
    def fit(self, x: 'np.array', y: 'np.array'):
        self.beta = np.cov(x, y)[0, 1] / x.var(ddof = 1)
        self.alpha = y.mean() - self.beta * x.mean()
    
    def predict(self, x):
        return self.alpha + self.beta * x


if __name__ == '__main__':
    print('Script:')
    x = stats.norm(1.5, 2.5).rvs(100)
    res = stats.norm(0, 0.3).rvs(100)

    y_pred = 5 + 0.3 * x
    y_real = 5 + 0.3 * x + res

    # fig, ax = plt.subplots(1, 2, figsize = (13, 6))

    # ax[0].plot(x, y_real, color = 'r', linestyle = '', marker = 'o')

    # ax[1].plot(x, y_real, color = 'r', linestyle = '', marker = 'o')
    # ax[1].plot(x, y_pred, color = 'b')

    # fig, ax = plt.subplots(1, 2, figsize = (13, 6))

    # ax[0].plot(x, y_real, color = 'r', linestyle = '', marker = 'o')

    # ax[1].plot(x, y_real, color = 'r', linestyle = '', marker = 'o')
    # ax[1].plot(x, y_pred, color = 'b')

    # for x, Y, y in zip(x, y_pred, y_real):
    #     ax[1].plot([x, x], [Y, y], linestyle = ':', color = 'y')


    model = LinearRegresor()
    model.fit(x, y_real)
    
    print(model.alpha, model.beta)

    y_model = model.predict(x)
    r2 = model.score(y_model, y_real)
    print(f'R2: {r2}')

    fig, ax = plt.subplots(1, 2, figsize = (13, 6))

    ax[0].plot(x, y_real, color = 'r', linestyle = '', marker = 'o')

    ax[1].plot(x, y_real, color = 'r', linestyle = '', marker = 'o')
    ax[1].plot(x, y_pred, color = 'b')
    ax[1].plot(x, y_model, color = 'g')

    ax[0].set_title(f'Datos Originales')
    ax[1].set_title(f'Modelo R2: {np.round(r2, 4)}')

    data = pd.DataFrame({'x': x, 'Real Data': y_real, 'Prediction': y_model})

    data = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/auto-mpg.csv'))
    print(data.head())

    # mask = data['mpg'].isna()
    # data[mask]

    data.dropna(inplace = True)

    data.info()

    fig, ax = plt.subplots(figsize = (13, 6))
    ax.plot(data['horsepower'], data['mpg'], color = 'r', linestyle = '', marker = 'o')
    ax.set_xlabel('Horsepower')
    ax.set_xlabel('mpg')
    ax.set_title('Datos')

    x, y = data['horsepower'], data['mpg']

    # x**2, 1 / x
    # np.log(y)

    model = LinearRegresor()
    y_model = model.fit(x, y)
    y_pred = model.predict(x)
    r2 = model.score(y_pred, y)
    print(r2)

    # fig, ax = plt.subplots(1, 2, figsize = (13, 6))

    # ax[0].plot(x, y, color = 'r', linestyle = '', marker = 'o')

    # ax[1].plot(x, y, color = 'r', linestyle = '', marker = 'o')
    # ax[1].plot(x, y_pred, color = 'b')
    # ax[1].plot(x, y_model, color = 'g')

    # ax[0].set_title(f'Datos Originales')
    # ax[1].set_title(f'Modelo R2: {np.round(r2, 4)}')

    plt.show()














