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
from sklearn.feature_selection import RFE

def vif(X):
    vif_data = pd.DataFrame()
    vif_data['Feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

    return vif_data


if __name__ == '__main__':
    print('Seleccion: ')
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/boston_clear.txt'))
    # print(df.head())

    X = df.drop('MEDV', axis=1).values.copy()
    y = df['MEDV'].values.copy()

    # Filter Method: N -> 1
    corr = df.corr()
    mask_corr = corr['MEDV'].abs() >= 0.5

    relevant_features = corr['MEDV'][mask_corr]
    # print(relevant_features)

    # Don't forget multicolianirity

    # Wrapper Method:
    # Backward Elimination
    nof= 0
    score_list = []   

    for n in range(1, X.shape[1] + 1):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = LinearRegression()
        rfe = RFE(model, n_features_to_select=n)
        
        X_train_rfe = rfe.fit_transform(X_train, y_train)
        X_test_rfe = rfe.transform(X_test)

        model.fit(X_train_rfe, y_train)
        score = model.score(X_test_rfe, y_test)
        score_list.append(score)

        if score == max(score_list):
            nof = n
    
    print(f'Total de variables: {X.shape[1]}')
    print(n)

    model = LinearRegression()
    rfe = rfe.fit_transform(X, y)
    model.fit(X_train_rfe, y_train)
    print(model.score(X_test_rfe, y_test))


    





