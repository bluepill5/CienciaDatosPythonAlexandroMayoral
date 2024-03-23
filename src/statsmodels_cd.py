import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

def vif(X):
    vif_data = pd.DataFrame()
    vif_data['Feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

    return vif_data


if __name__ == '__main__':
    print('Regresion: ')
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/Advertising.csv'))
    print(df.head())
    print(df.columns)
    model = smf.ols(formula='Sales~TV + Radio', data=df).fit()
    print(model)
    print(model.params)
    print(model.rsquared)
    # print(model.predict(df['TV']))
    print(model.pvalues)
    print(model.summary)

    X = df.drop('Sales', axis=1).values
    print(variance_inflation_factor(X, 2))

    X = df[['TV', 'Radio', 'Newspaper']]
    print(vif(X))

    df_h = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/housing.csv'), 
                     usecols=['total_rooms', 'total_bedrooms', 'median_income'])
    df_h.dropna(inplace=True, how='any')
    print(vif(df_h))

    # Con sklearn
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/Advertising.csv'))
    X = df.drop('Sales', axis=1)
    y = df['Sales'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    model.score(X_train, y_train)
    model.score(X_test, y_test)





