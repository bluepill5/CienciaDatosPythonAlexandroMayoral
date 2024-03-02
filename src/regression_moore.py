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
    
def transitor_cleaner(s):
     s = s.replace(',', '')
     s = s.replace('~', '')
     s = s.replace(' ', '')
     s = s.split('[')[0]

     for c in s:
          if c.isalpha():
               s = s.replace(c, '')

     return float(s)

if __name__ == '__main__':
    print('Script:')

    data = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/moore.csv'), delimiter='\t', names = ['Model', 'Transitors', 'Year', 'Company', 'TS', 'area'])
    data.info()

    # print(data.head())
    # data.info()

    x_data = data['Year'].apply(lambda x: int(x.split('[')[0]))
    y_data = data['Transitors'].apply(transitor_cleaner)

    # Modelar Year vs Transitors
    # fig, ax = plt.subplots(figsize = (13, 6))
    # ax.plot(x_data, y_data, linestyle='', marker = 'o')
    # plt.show()

    # Model
    model = LinearRegresor()

    # x_data, y_data = data['Year'], data['Transitors']
    # Con logaritmos
    x = x_data
    y = np.log(y_data)

    model.fit(x, y)
    y_pred = model.predict(x)
    r2 = model.score(y_pred, y)
    print(r2)

    # Plotting
    fig, ax = plt.subplots(figsize = (13, 6))

    ax.plot(x_data, y_data, color = 'r', linestyle = '', marker = 'o')
    ax.plot(x, np.exp(y_pred), linestyle = '-', marker = '.')
    ax.set_xlabel("Year")
    ax.set_ylabel("Transitors")
    ax.set_title(f"Year vs Transitors $R^2$ = {np.round(r2, 2)}")

    plt.show()

    # Al cuadrado
    x = x_data
    y = y_data**2

    model.fit(x, y)
    y_pred = model.predict(x)
    r2 = model.score(y_pred, y)
    print(r2)

    # Plotting
    fig, ax = plt.subplots(figsize = (13, 6))

    ax.plot(x_data, y_data, color = 'r', linestyle = '', marker = 'o')
    ax.plot(x, np.sqrt(y_pred), linestyle = '-', marker = '.')
    ax.set_xlabel("Year")
    ax.set_ylabel("Transitors")
    ax.set_title(f"Year vs Transitors $R^2$ = {np.round(r2, 2)}")

    plt.show()

    # Sin transformar
    x = x_data
    y = y_data

    model.fit(x, y)
    y_pred = model.predict(x)
    r2 = model.score(y_pred, y)
    print(r2)

    # Plotting
    fig, ax = plt.subplots(figsize = (13, 6))

    ax.plot(x_data, y_data, color = 'r', linestyle = '', marker = 'o')
    ax.plot(x, y_pred, linestyle = '-', marker = '.')
    ax.set_xlabel("Year")
    ax.set_ylabel("Transitors")
    ax.set_title(f"Year vs Transitors $R^2$ = {np.round(r2, 2)}")

    plt.show()

        

    
    

    





