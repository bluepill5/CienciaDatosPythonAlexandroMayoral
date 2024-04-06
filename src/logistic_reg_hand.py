import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import aztlan as az

np.random.seed(1995)

class Classifier:
    def accuracy(self, y_pred, y_real):
        return (y_pred == y_real).mean()
    
    def precision(self, y_pred, y_real, true = 1):
        mask = y_pred == true
        return (y_pred[mask] == y_real[mask]).mean()
    
    def recall(self, y_pred, y_real, true = 1):
        mask = y_real == true
        return (y_pred[mask] == y_real[mask]).mean()
    
    def f1_score(self, y_pred, y_real, true = 1):
        n = self.recall(y_pred, y_real, true = true) * self.precision(y_pred, y_real, true = true)
        d = self.recall(y_pred, y_real, true = true) + self.precision(y_pred, y_real, true = true)

        return 2 * (n / d)



class LogisticRegressionP(Classifier):
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        
    def _cost_function(self, *w):
        w = np.array(w)
        n = len(self.y)
        h = self.__sigmoid(self.X @ w)

        J = -1 / n * (self.y * np.log(h) + (1 - self.y) * np.log(1 - h)).sum()

        return J
        
    def fit(self, X, y, epsilon = 0.0001, max_iter = 10_000, alpha = 0.3, delta = 1e-7, orbit = False, verbose=False):
        # Agrega una columna de unos a X para representar el término de sesgo
        ones = np.ones(X.shape[0])
        X = np.column_stack([X, ones])  

        self.y = y
        self.X = X
        self.w = np.random.random(self.X.shape[1])

        if orbit:
            self.w, self.orbit = az.gradient_des(self._cost_function, self.w, epsilon=epsilon, 
                                                 max_iter=max_iter, alpha=alpha, delta=delta, 
                                                 orbit=orbit, verbose=verbose)
        else:
            self.w = az.gradient_des(self._cost_function, self.w, epsilon=epsilon, max_iter=max_iter, 
                                     alpha=alpha, delta=delta, orbit=orbit, verbose=verbose)


    def predict(self, X):
        # Agrega una columna de unos a X para representar el término de sesgo
        ones = np.ones(X.shape[0])
        X = np.column_stack([X, ones])

        probabilities = self.__sigmoid(X @ self.w)
        return (probabilities >= 0.5).astype(int)
        # return np.round(probabilities)

if __name__ == '__main__':
    print('Logistic Regression:')
    np.random.seed(1995)

    x = [4 + np.random.normal() for i in range(20)] + [2 + np.random.normal() for i in range(20)]
    y = [4 + np.random.normal() for i in range(20)] + [2 + np.random.normal() for i in range(20)]
    z = [1] * 20 + [0] * 20
    data = pd.DataFrame({'x' : x, 'y' : y, 'Type' : z})
    
    X = data.drop('Type', axis = 1).values
    y = data['Type'].values

    model = LogisticRegressionP()

    model.fit(X, y, orbit=True)

    # Predice las etiquetas para los datos de prueba
    y_pred = model.predict(X)

    print(f'Precision: {model.accuracy(y_pred, y)}')
    print(f'Recall: {model.recall(y_pred, y)}')
    print(f'F1 sCORE: {model.f1_score(y_pred, y)}')

    # Con Sklearn
    print('Con Sklearn obtenemos:')
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import ConfusionMatrixDisplay
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report

    clf = LogisticRegression()
    clf.fit(X, y)

    y_pred = clf.predict(X)

    print(classification_report(y, y_pred))

    
    C = ConfusionMatrixDisplay(confusion_matrix(y, y_pred))

    C.plot()
    plt.show()

    from sklearn.model_selection import cross_val_score
    print(cross_val_score(clf, X, y, cv = 5, scoring='accuracy').mean())
    print(cross_val_score(clf, X, y, cv = 5, scoring='precision').mean())
    print(cross_val_score(clf, X, y, cv = 5, scoring='recall').mean())
    print(cross_val_score(clf, X, y, cv = 5, scoring='f1').mean())







    

