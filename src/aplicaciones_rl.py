import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sympy import *

from functions_calc import *


if __name__ == '__main__':
    print('Aplicaciones:')
    
    x = np.random.normal(1.6, 0.3, 1000)
    y = 3 * x + 2 + np.random.normal(0.8, 0.2, 1000)

    def s(w0, w1):
        r = (y - w1 * x - w0)**2
        return r.sum()

    def s(w0, w1):
        r = (y**2).sum() - (2 * (x * y).sum()) * w1 - (2 * y.sum()) * w0
        r += (x**2).sum() * w1**2 + (2 * x.sum()) * w1 * w0 + len(x) * w0**2
        return r


    fig, ax = plt.subplots(figsize = (13, 6))

    ax.scatter(x, y)
    ax.plot(np.linspace(0.75, 2.75, 1000), 3 * np.linspace(0.75, 2.75, 1000) + 2.8, color = 'r')
    ax.plot([1.75], [9], marker = '*', markersize = 15, color = 'g')
    # plt.show()

    x0 = np.random.random(2)
    min_, orbit = gradient_dec(s, x0, orbit = True, verbose = True, alpha = 2e-5, epsilon = 0.0000001)
    mini_ = descenso_gradiente(s, x0, tol = 0.0000001, tasa_aprendizaje = 2e-5)
    print(min_)

    def model(x):
        return min_[0] + min_[1] * x

    x, a0, a1, a2, a3, a4, a5 = symbols('x, a_0, a_1, a_2, a_3, a_4, a_5')
    p = a0 + a1 * x + a2 * x**2 + a3 * x**3 + a4 * x**4 + a5 * x**5
    print(p)

    j = (p - sin(x)) **2

    print(integrate(j, (x, -pi, pi)))
    
    # Definir la función objetivo (en este caso, la función seno)
    def target_function(x):
        np.sin(x)

    # Definir la función que queremos aproximar (polinomio de grado 5)
    def polynomial_function(coeffs, x):
        return np.polyval(coeffs, x)


