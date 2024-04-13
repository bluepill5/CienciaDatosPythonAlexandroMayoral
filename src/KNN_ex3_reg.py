import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import aztlan as az

class KNNReg(az.Regressors):
    def __init__(self, k=5, p=2):
        self.k = k
        self.p = p

    def __distance(self, p1, p2):
        return ((np.abs(p1 - p2) ** self.p).sum()) ** (1 / self.p)
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        c = {}.fromkeys(y)
        self.classes = list(c.keys())

    def predict(self, X):
        return np.array([self.__predict(x) for x in X])

    def __predict(self, x):
        distances = [(self.__distance(x, point), class_) for point, class_ in zip(self.X, self.y)]
        distances.sort(key = lambda x: x[0])

        neighbors = distances[:self.k]
        values = np.array([n[1] for n in neighbors])

        return values.mean()

if __name__ == '__main__':
    print('KNN Regressor: ')

    data = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/auto-mpg.csv'), index_col=False)
    data.dropna(inplace=True)

    # print(data.head())

    X = data['horsepower'].values
    X = X.reshape((data.shape[0], 1))
    y = data['mpg'].values

    model = KNNReg()
    model.fit(X, y)

    y_pred = model.predict(X)

    print(model.score(y_pred, y))

    fig, ax = plt.subplots(figsize=(13, 6))
    sns.scatterplot(data = data, x = 'horsepower', y = 'mpg', ax = ax)
    sns.lineplot(x = X.reshape((data.shape[0], )), y = y_pred, color='r', ax=ax)


    # Con SKlearn
    from sklearn.neighbors import KNeighborsRegressor

    model = KNeighborsRegressor()
    model.fit(X, y)

    y_pred = model.predict(X)
    print(f'Score: {model.score(X, y)}')

    fig, ax = plt.subplots(figsize=(13, 6))
    sns.scatterplot(data = data, x = 'horsepower', y = 'mpg', ax = ax)
    sns.lineplot(x = X.reshape((data.shape[0], )), y = y_pred, color='r', ax=ax)

    plt.show()











    
