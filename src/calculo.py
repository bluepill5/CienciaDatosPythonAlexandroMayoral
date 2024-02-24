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
        # print(partial(func, vals, i, delta =  0.00001))
    
    return np.array(gradientes).reshape(len(vals), len(func(*vals))).T

def jacobian(funcs, vals, delta = 0.00001):
    j = [gradiente(funcs[i], vals, delta = delta) for i in range(len(funcs))]
    return np.array(j)

def hessiana(func: "Function", vals: tuple, delta = 0.00001):
    #jac = jacobiana(func, vals, delta)
    n = len(vals)
    hessiana = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            def df(*vals):
                return partial(func, vals, i, delta = delta)
            # Derivada cruzada de las derivadas parciales
            hessiana[i, j] = partial(df, vals, j, delta)
    return hessiana

def hessian(func: "Function", vals: tuple, delta = 0.00001):
    h = []
    for i in range(len(vals)):
        r = []
        for j in range(len(vals)):
            def df(*vals):
                return partial(func, vals, i, delta = delta)
            r.append(partial(df, vals, j, delta = delta))
        
        h.append(r)
    return np.array(h)

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

def gradient_dec(func: "Function", x0: tuple, 
                 epsilon: float = 0.001, 
                 max_iter: int = 10_00,
                 alpha: float = 0.3,
                 delta: float = 1e-6,
                 orbit: bool = False,
                 verbose: bool = False):
    x0 = np.array(x0, dtype = float)
    x1 = x0 - alpha * gradiente(func, x0, delta = delta)

    points = [x0, x1]

    i = 1
    while np.linalg.norm(x1 - x0) > epsilon and i <= max_iter:
        x0 = x1
        x1 = x0 - alpha * gradiente(func, x0, delta = delta)
        i += 1

        points.append(x1)
    
    if verbose:
        print(f'Iteraciones: {i}')
    
    if orbit:
        return x1, points
    
    return x1


def sig(x):
    return (1 / (1 + np.exp(-x)))


# gradiente, jacobiana, hessiana, descenso del gradiente

# Funciones de ejemplo
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

    # print(len((1, 1, 1)), len(g(1, 1, 1)))
    print(jacobiana(g, (1, 2, 3)))
    print(jacobian((f1, f2), (1, 2, 3)))
    h = hessiana(f, (0, 0))

    print(np.linalg.det(h))
    print('Descenso del gradiente:')
    punto_minimo = descenso_gradiente(f, (1, 1))
    print(punto_minimo)

    x0 = np.random.random(2)
    #x0 = (2, 28)
    min_, orbit = gradient_dec(f, x0, epsilon= 1e-5,alpha = 0.05, orbit = True, verbose = True)

    fig, ax = plt.subplots(1, 2, figsize = (13, 6))
    x = [p[0] for p in orbit]
    y = [p[1] for p in orbit]

    ax[0].plot(x)
    ax[1].plot(y)

    ax[0].set_title('Componente x del gradiente')
    ax[0].set_xlabel('Iteraciones')
    ax[0].set_ylabel('Valor de la componente')

    ax[1].set_title('Componente x del gradiente')
    ax[1].set_xlabel('Iteraciones')
    ax[1].set_ylabel('Valor de la componente')

    fig, ax = plt.subplots(figsize = (13, 6))

    x = [f(*p) for p in orbit]

    ax.plot(x)
    ax.set_title('Valor de la funcion a minimizar')

    plt.show()

    # fig, ax = plt.subplots(subplot_kw = {'projection': '3d'}, figsize = (13, 8))

    # x, y = np.meshgrid(np.linspace(-3, 3, 1000),
    #                    np.linspace(-3, 3, 1000))
    # z = f(x, y)

    # ax.plot_surface(x, y, z, cmap = cm.magma)
    # plt.show()

    # fig, ax = plt.subplots(figsize = (13, 8))

    # x, y = np.meshgrid(np.linspace(-3, 3, 1000),
    #                    np.linspace(-3, 3, 1000))
    # z = f(x, y)

    # ax.contour(x, y, z, 21, cmap = cm.magma)
    # plt.show()

    # fig, ax = plt.subplots(figsize = (13, 8))

    # x, y = np.meshgrid(np.linspace(-3, 3, 1000),
    #                    np.linspace(-3, 3, 1000))
    # z = f(x, y)

    # x1, y1 = np.meshgrid(np.linspace(-3, 3, 15),
    #                      np.linspace(-3, 3, 15))
    # u = [gradient(f, (x, y))[0] for x, y in zip(x1.flat, y1.flat)]
    # v = [gradient(f, (x, y))[1] for x, y in zip(x1.flat, y1.flat)]

    # ax.contour(x, y, z, 20, cmap = cm.magma)
    # ax.quiver(x1, y1, u, v)

    # plt.show()



    # fig, ax = plt.subplots(1, 2, figsize=(13, 8))
    # x = np.linspace(-10, 10, 1000)
    # ax[0].plot(x, sig(x))
    # ax[0].axhline(0.5, color = 'r')
    # ax[1].plot(x, np.tanh(x))
    # ax[1].axhline(0, color = 'r')
    # plt.show()





















