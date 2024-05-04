import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.datasets import make_blobs
import os
import aztlan as az
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm

class KMeansClass:
    def __init__(self, n_clusters=4, max_iter=1000):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None
        self.labels = None

    def fit(self, X):
        # Iniciliazacion de los centroides
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

        for _ in range(self.max_iter):
            # Calcular las distancias entre los puntos y los centroides
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))

            # Asignar cada punto al centroide mas cercano
            self.labels = np.argmin(distances, axis=0)

            # Actualizamos los centroides como la media de los puntos asignados a cada cluster
            new_centroids = np.array([X[self.labels == i].mean(axis=0) for i in range(self.n_clusters)])

            # Verificamos la convergencia
            if np.all(self.centroids == new_centroids):
                break

            self.centroids = new_centroids

class KMeansClassTwo:
    def __init__(self, n_clusters, rseed=2, metric='euclidean'):
        self.n_clusters = n_clusters
        self.rseed = rseed
        self.metric = metric

    def fit(self, X):
        rng = np.random.RandomState(self.rseed)
        i = rng.permutation(X.shape[0])[:self.n_clusters]
        centroids = X[i]

        while True:
            labels = pairwise_distances_argmin(X, centroids, metric=self.metric)
            new_centers = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])

            if np.all(centroids == new_centers):
                break

        self.centroids = centroids
        self.labels = labels

if __name__ == '__main__':
    print('K-Means: ')
    X, y_true = make_blobs(n_samples = 300, centers = 4, cluster_std = 0.60, random_state = 0)

    kmeans = KMeans(n_clusters = 4)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    kmeans.cluster_centers_

    fig, ax = plt.subplots(figsize = (13, 8))

    ax.scatter(X[:, 0], X[:, 1], c = y_kmeans, s = 50, cmap = 'viridis')

    centers = kmeans.cluster_centers_
    # ax.scatter(centers[:, 0], centers[:, 1], c = 'k', s = 200, alpha = 0.5)

    # Crear una instancia del modelo KMeans
    kmeans = KMeansClass(n_clusters=6)
    # Ajustar el modelo a los datos
    kmeans.fit(X)

    # Obtener las etiquetas de los clusters
    labels = kmeans.labels

    # Obtener los centroides finales
    centroids = kmeans.centroids

    print("Etiquetas de los clusters:", labels)
    print("Centroides finales:\n", centroids)

    # Graficar los puntos y los centroides
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', s=200, c='red')
    plt.title('Clustering con K-Means')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    plt.show()


    







