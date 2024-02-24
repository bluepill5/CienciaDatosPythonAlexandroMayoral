import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sympy import *

def deriv(func: "Function", x, delta = 0.0000001):
    return (func(x + delta) - func(x)) / delta

def partial(func: "Function", vals: tuple, index, delta = 0.000001): #(x, y)
    vals = np.array(vals, dtype = float)
    vals_delta = vals.copy()
    vals_delta[index] += delta    
    return (func(*vals_delta) - func(*vals)) / delta

def gradiente(func: "Function", vals: tuple, delta = 0.00001):
    return np.array([partial(func, vals, i, delta) for i in range(len(vals))])

def jacobiana(func: "Function", vals: tuple, delta = 0.00001):
    gradientes = []
    for i in range(len(vals)):
        gradientes.append(partial(func, vals, i, delta =  0.00001))
    
    return np.array(gradientes).reshape(len(vals), len(func(*vals))).T

def hessiana(func: "Function", vals: tuple, delta = 0.00001):
    n = len(vals)
    hessiana = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            def df(*vals):
                return partial(func, vals, i, delta = delta)
            # Derivada cruzada de las derivadas parciales
            hessiana[i, j] = partial(df, vals, j, delta)
    return hessiana

def descenso_gradiente(func, x0, tasa_aprendizaje = 0.01, max_iter = 1000, tol = 1e-6):
    punto_actual = np.array(x0)
    iteracion = 0

    while iteracion < max_iter:
        # Calculamos el gradiente en el punto actual
        grad = gradiente(func, punto_actual)
        # Actualizamos el punto utilizando la regla del gradiente descendiente
        nuevo_punto = punto_actual - tasa_aprendizaje * grad
        # Comprobamos el criterio de parada
        if np.linalg.norm(nuevo_punto - punto_actual) < tol:
            break
        # Alctualizar el punto actual
        punto_actual = nuevo_punto
        iteracion += 1
    
    print(f'Numero de iteraciones: {iteracion}')
    print(f'Punto minimo encontrado: {punto_actual}')
    
    return punto_actual

# Funciones de ejemplo
def sig(x):
    return (1 / (1 + np.exp(-x)))

def mass(r, h, t = 0.1, rho = 7.85):
    return 2 * np.pi * rho * t * r **2 + 2 * np.pi * r * h * t * rho

def funcion_ejemplo(x, y, z):
    return x**2 + y**2 + z**2

def f(x, y):
    return -(4*x) / (x**2 + y**2 + 1)

def f(x, y):
    return -(x**2 + y**2)

def f(x, y):
    return x**2 + y**2

def f(x, y):
    return x**2 - y**2

def f1(x, y, z):
    return x * y * z

def f2(x, y, z):
    return y - z**3

def g(x, y, z):
    return np.array([f1(x, y, z), f2(x, y, z)])

def g(x, y, z):
    return x**2 + y**2 + z**2

def g(x, y, z):
    return np.array([x* y *z, y - z**3])

def f(x, y):
    return -(x**2 + y**2)

def f(x, y):
    return 3*x**2 + 7*y**2

if __name__ == '__main__':

    # print(jacobiana(f, (1, 2)))
    # h = hessiana(f, (0, 0))
    # print(np.linalg.det(h))
    
    print('Descenso del gradiente:')
    punto_minimo = descenso_gradiente(f, (1, 1))
    print(punto_minimo)
    
    fig, ax = plt.subplots(subplot_kw = {'projection': '3d'}, figsize = (13, 8))

    x, y = np.meshgrid(np.linspace(-3, 3, 1000),
                       np.linspace(-3, 3, 1000))
    z = f(x, y)

    ax.plot_surface(x, y, z, cmap = cm.magma)
    plt.show()

    fig, ax = plt.subplots(figsize = (13, 8))

    x, y = np.meshgrid(np.linspace(-3, 3, 1000),
                       np.linspace(-3, 3, 1000))
    z = f(x, y)

    ax.contour(x, y, z, 21, cmap = cm.magma)
    plt.show()

    fig, ax = plt.subplots(figsize = (13, 8))

    x, y = np.meshgrid(np.linspace(-3, 3, 1000),
                       np.linspace(-3, 3, 1000))
    z = f(x, y)

    x1, y1 = np.meshgrid(np.linspace(-3, 3, 15),
                         np.linspace(-3, 3, 15))
    u = [gradiente(f, (x, y))[0] for x, y in zip(x1.flat, y1.flat)]
    v = [gradiente(f, (x, y))[1] for x, y in zip(x1.flat, y1.flat)]

    ax.contour(x, y, z, 20, cmap = cm.magma)
    ax.quiver(x1, y1, u, v)

    plt.show()





















