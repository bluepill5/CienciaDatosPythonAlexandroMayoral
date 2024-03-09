import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

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
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/housing.csv'))

    outliers = Outliers()

    x = data[['total_rooms', 'median_income']].values.copy()

    new_x, _ = outliers.fit(x)

    fig, ax = plt.subplots(figsize = (13, 8))
    ax.plot(new_x[:, 0], new_x[:, 1], linestyle = '', color = 'b', marker = 'o')
    ax.plot(new_x[:, 0].mean(), new_x[:, 1].mean(), linestyle = '', marker = '*', markersize = 20, color = 'r')

    new_x, _ = outliers.fit(x, method='z_score')

    fig, ax = plt.subplots(figsize = (13, 8))
    ax.plot(new_x[:, 0], new_x[:, 1], linestyle = '', color = 'b', marker = 'o')
    ax.plot(new_x[:, 0].mean(), new_x[:, 1].mean(), linestyle = '', marker = '*', markersize = 20, color = 'r')


    plt.show()



