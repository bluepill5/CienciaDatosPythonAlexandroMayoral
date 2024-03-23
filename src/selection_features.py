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
    def backward_selection(data):
        df = data.copy()
        df_all = data.copy()

        y = df['MEDV'].values.copy()
        df.drop('MEDV', axis=1, inplace=True)
        X = df

        vars_string = '+'.join(df.columns)
        
        scores = {}

        for _ in range(len(df.columns)):
        # while vars_string:
            vars_string = '+'.join(df.columns)
            print(vars_string)
            model_string = 'MEDV~' + vars_string

            model = smf.ols(formula=model_string, data=df_all).fit()
            var_del = model.pvalues.nlargest(1).index
            if var_del == 'Intercept':
                try:
                    var_del = model.pvalues[1:].nlargest(1).index
                except:
                    print(model.pvalues)
                    break

            model_sk = LinearRegression()
            model_sk.fit(X, y)

            scores[vars_string] = model_sk.score(X, y)
            print(scores[vars_string])

            # Eliminamos la variable
            df.drop(var_del, axis=1, inplace=True)
        print(model.pvalues)
        return scores

    scores = backward_selection(df)
    # print(scores)
    





