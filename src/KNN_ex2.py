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

class KNN(az.Classifier):
    def __init__(self, k=3, p=2):
        self.k = k
        self.p = p

    def __distance(self, p1, p2):
        return ((np.abs(p1 - p2) ** self.p).sum()) ** (1 / self.p)
    
    def fit(self, X, yn):
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
        votes = [n[1] for n in neighbors]
        counter = {}.fromkeys(votes, 0)

        for v in votes:
            counter[v] += 1

        max_ = max(counter.values())

        for k in counter.keys():
            if counter[k] == max_:
                return k

if __name__ == '__main__':
    print('KNN: ')

    data = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/Iris.csv'), index_col=False)
    data.drop('Id', axis=1, inplace=True)
    # print(data.head())

    # Separar características y etiquetas
    X = data.drop('Species', axis=1).values
    y = data['Species'].values

    # Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear una instancia de KNN para clasificación
    knn_cls = KNN(k=3, p=2)
    knn_cls.fit(X_train, y_train)
    predictions_cls = knn_cls.predict(X_test)

    msn = 'Class: {:15} -> Precision: {:8} '

    for c in knn_cls.classes:
        print(msn.format(c, knn_cls.precision(predictions_cls, y_test, true=c)))

    msn = 'Class: {:15} -> Recall: {:8} '

    for c in knn_cls.classes:
        print(msn.format(c, knn_cls.recall(predictions_cls, y_test, true=c)))


    # Con sklearn
    from sklearn.neighbors import KNeighborsClassifier

    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(f'Accuracy: {clf.score(X_test,y_test)}')

    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred))



    
