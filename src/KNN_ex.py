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

class KNN:
    def __init__(self, k=3, regression=False):
        self.k = k
        self.regression = regression
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for sample in X_test:
            distances = []
            for train_sample, label in zip(self.X_train, self.y_train):
                dist = np.linalg.norm(sample - train_sample)
                distances.append((dist, label))
            distances.sort(key=lambda x: x[0])
            k_nearest = distances[:self.k]
            k_nearest_labels = [label for dist, label in k_nearest]
            
            if self.regression:
                prediction = np.mean(k_nearest_labels)
            else:
                prediction = max(set(k_nearest_labels), key=k_nearest_labels.count)
            predictions.append(prediction)

        return predictions


if __name__ == '__main__':
    print('KNN: ')

    data = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/Iris.csv'), index_col=False)
    print(data.head())

    # Separar características y etiquetas
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # Codificar las etiquetas en números
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear una instancia de KNN para clasificación
    knn_cls = KNN(k=3, regression=False)
    knn_cls.fit(X_train, y_train)
    predictions_cls = knn_cls.predict(X_test)
    accuracy_cls = accuracy_score(y_test, predictions_cls)
    print("Accuracy (Classification):", accuracy_cls)

    # Crear una instancia de KNN para regresión
    knn_reg = KNN(k=3, regression=True)
    knn_reg.fit(X_train, y_train)
    predictions_reg = knn_reg.predict(X_test)
    mse_reg = np.mean((y_test - predictions_reg) ** 2)
    print("Mean Squared Error (Regression):", mse_reg)

