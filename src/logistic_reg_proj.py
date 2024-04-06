import pandas as pd
import numpy as np
import aztlan as az
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

if __name__ == '__main__':
    print('Logistic Regression Imbalance:')
    x, y = make_classification(n_samples = 10_000  # number of samples
                          ,n_features = 2    # feature/label count
                          ,n_informative = 2 # informative features
                          ,n_redundant = 0   # redundant features
                          ,n_repeated = 0    # duplicate features
                          ,n_clusters_per_class = 1  # number of clusters per class; clusters during plotting
                          ,weights = [0.99]   # proportions of samples assigned to each class
                          ,flip_y = 0         # fraction of samples whose class is assigned randomly. 
                          ,random_state = 13)
    
    df = pd.DataFrame(x,columns=['x1','x2'])
    df['y'] = y

    # Porcentajes de etiquetas
    print(df.head())
    print(df['y'].value_counts() / df.shape[0])
    # Visualizacion de los datos
    fig, ax = plt.subplots(figsize = (13, 8))
    sns.scatterplot(data = df,x = 'x1',y = 'x2',hue = 'y')

    # Separamos los datos
    x = df.drop('y',axis=1)
    y = df['y']

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 13)

    clf = LogisticRegression(random_state=15, class_weight=None)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred))

    plotter = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
    plotter.plot()
    
    # Agregando pesos
    W = {0:1, 1:99}

    clf = LogisticRegression(random_state=15, class_weight=W)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred))

    plotter = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
    plotter.plot()

    # Ahora con una malla
    # define weight hyperparameter
    w = [{0:1000,1:100},{0:1000,1:10}, {0:1000,1:1.0}, 
         {0:500,1:1.0}, {0:400,1:1.0}, {0:300,1:1.0}, {0:200,1:1.0}, 
        {0:150,1:1.0}, {0:100,1:1.0}, {0:99,1:1.0}, {0:10,1:1.0}, 
        {0:0.01,1:1.0}, {0:0.01,1:10}, {0:0.01,1:100}, 
        {0:0.001,1:1.0}, {0:0.005,1:1.0}, {0:1.0,1:1.0}, 
        {0:1.0,1:0.1}, {0:10,1:0.1}, {0:100,1:0.1}, 
        {0:10,1:0.01}, {0:1.0,1:0.01}, {0:1.0,1:0.001}, {0:1.0,1:0.005}, 
        {0:1.0,1:10}, {0:1.0,1:99}, {0:1.0,1:100}, {0:1.0,1:150}, 
        {0:1.0,1:200}, {0:1.0,1:300},{0:1.0,1:400},{0:1.0,1:500}, 
        {0:1.0,1:1000}, {0:10,1:1000},{0:100,1:1000} ]
    
    hyperparam_grid = {'class_weight': w}

    clf = LogisticRegression(random_state=15)

    grid = GridSearchCV(clf, hyperparam_grid, scoring='f1', cv = 100, n_jobs=-1, refit=True)
    
    grid.fit(x, y)
    print(grid.best_score_)
    print(grid.best_params_)

    clf = LogisticRegression(random_state=15, **grid.best_params_)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred))

    plotter = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
    plotter.plot()

    x = np.linspace(-4, 4)
    # Visualizacion de los datos
    fig, ax = plt.subplots(figsize = (13, 8))
    sns.scatterplot(data = df,x = 'x1',y = 'x2',hue = 'y')
    ax.plot(x, clf.coef_[0][0] / clf.coef_[0][1] * x - (clf.intercept_[0][1]))
    
    
    
    plt.show()

