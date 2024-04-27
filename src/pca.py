import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
    
    def fit(self, X):
        # Calculamos la media de los datos
        self.mean = np.mean(X, axis=0)

        # Centramos los datos
        X_centered = X - self.mean

        # Calculamos la matriz de covarianzas
        cov_matrix = np.cov(X_centered, rowvar=False)

        # Calculamos los autovalores y autovectores
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Ordenamos los autovalores y autovectore de mayor a menor
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        # Seleccionamos los k autovectores correspondientes a los mayores autovalores
        self.components = eigenvectors[:, :self.n_components]

    def transform(self, X):
        # Centramos los datos usando la media calculada durante el ajuste
        X_centered = X - self.mean

        # Proyectamos los datos en el nuevo espacio de caracteristicas
        return np.dot(X_centered, self.components)


if __name__ == '__main__':
    print('PCA: ')
    
    # Ejemplo de uso con datos de Iris
    iris = load_iris()
    X = iris.data
    y = iris.target

    print(X)

    # Aplicamos PCA con 2 componentes
    pca = PCA(n_components=2)
    pca.fit(X)
    X_pca = pca.transform(X)

    # Graficamos los datos proyectados
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
    plt.xlabel('Componente principal 1')
    plt.ylabel('Componente principal 2')
    plt.title('PCA de datos iris')
    plt.colorbar(label='Clase')
    plt.show()




    