import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

class Outliers:
  def __iqr(self, x):
    x = x.copy()
    q1, q3 = np.percentile(x, [25, 75])
    iqr =  q3 - q1

    mask = (x < (q1 - 1.5 * iqr)) | (x > (q3 + 1.5 * iqr))
    outliers = x[mask].copy()
    x[mask] = np.nan

    return x, outliers

  def __z_score(self, x, threshold = 3):
    x = x.copy()

    outliers = []

    while True:
      z = (x - np.nanmean(x)) / np.nanstd(x)

      mask = (z < -threshold) | (z > threshold)

      if mask.sum() == 0:
        break

      outliers.extend(np.where(mask)[0])
      x[mask] = np.nan

      return x, outliers

  def fit(self, x, method = 'iqr', threshold = 3):
    x = x.copy()
    if method == 'iqr':
      if len(x.shape) == 1:
         return self.__iqr(x)
      else:
        new_x = [self.__iqr(x[:, i])[0] for i in range(x.shape[1])]
        new_x = np.column_stack(new_x)

        outliers = [self.__iqr(x[:, i])[1] for i in range(x.shape[1])]
        return new_x, outliers

    elif method == 'z_score':
      if len(x.shape) == 1:
        return self.__z_score(x, threshold=threshold)
      else:
        mu = x.mean(axis = 0)
        distances = np.array([np.linalg.norm(v - mu) for v in x])

        _, outliers = self.__z_score(distances, threshold=threshold)
        x[outliers, :] = np.nan
        return x, outliers

if __name__ == '__main__':
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/Advertising.csv'))
    X = data.drop('Sales', axis = 1).values
    y = data['Sales'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LinearRegression()
    model.fit(X_train, y_train)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    y_test, y_pred

    print(model.score(X_train, y_train))
    print(model.score(X_test, y_test))

    model.coef_
    model.intercept_

    # Seleccion de Atributos (Feature Selection)
    # Filter method
    # Wrapper method
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/boston_clear.txt'))

    # Filter Method
    X = data.drop('MEDV', axis = 1).values.copy()
    y = data['MEDV'].values

    fig, ax = plt.subplots(figsize = (13, 6))
    corr = data.corr()

    sns.heatmap(corr, annot = True, ax = ax)
    plt.show()

    mask = corr['MEDV'].abs() > 0.5
    relevant_features = corr['MEDV'][mask]

    data.loc[:, relevant_features.index]

    x = data.drop('MEDV', axis = 1)
    x.drop('PTRATIO', axis = 1, inplace=True)

    model = LinearRegression()

    model.fit(x, y)
    model.score(x, y)

    # Wrapper Method
    

   

    

    
