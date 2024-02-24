import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sympy import *

from functions_calc import *


if __name__ == '__main__':
    print('Aplicaciones: Sin(x) en (-pi, pi)')

    x, a0, a1, a2, a3, a4, a5 = symbols('x, a_0, a_1, a_2, a_3, a_4, a_5')
    p = a0 + a1 * x + a2 * x**2 + a3 * x**3 + a4 * x**4 + a5 * x**5

    j = (p - sin(x)) **2

    # print(integrate(j, (x, -pi, pi)))
        
    def target_function(x):
        return np.sin(x)
    
    def polynomial_function(coeffs, x):
        return np.polyval(coeffs, x)
    
    def polynomial_function(x):
        return min_[0] + min_[1] * x + min_[2] * x**2 + min_[3] * x**3 + min_[4] * x**4 + min_[5] * x**5
    
    def loss(a0, a1, a2, a3, a4, a5):
        r = (2*np.pi*a0**2) + ((4*np.pi**3*a0*a2)/3) + ((4*np.pi**5*a0*a4)/5) + ((2*np.pi**3*a1**2)/3) + ((4*np.pi**5*a1*a3)/5) + ((4*np.pi**7*a1*a5)/7) - (4*np.pi*a1) + ((2*np.pi**5*a2**2)/5) + ((4*np.pi**7*a2*a4)/7) + ((2*np.pi**7*a3**2)/7) + ((4*np.pi**9*a3*a5)/9) - (4*np.pi**3*a3) + (24*np.pi*a3) + ((2*np.pi**9*a4**2)/9) + ((2*np.pi**11*a5**2)/11) - (480*np.pi*a5) - (4*np.pi**5*a5) + (80*np.pi**3*a5) + (np.pi)
        return r

    x0 = np.random.randn(6)
    x0 = [-0.011229650946330222, -0.00419889384407729, 0.38533405081449215, -0.005108049161826688, -1.1002419929751505, 0.006905270127071032]
    x0 = [-6.54727150e-05, 9.87276462e-01, 6.80122959e-05, -1.54990162e-01, -8.15011308e-06, 5.61723721e-03]
    x0 = [0, 0, 0, 0, 0, 0]
    # print(loss(*x0))
    a, e = 0.5e-5, 1e-6
    a, e = 0.16e-4, 1e-6
    a, e = 0.18e-4, 1e-6
    a, e = 0.184e-4, 1e-9
    a, e, iter = 0.184e-4, 1e-12, 500_000
    min_, orbit = gradient_dec(loss, x0, orbit = True, verbose = True, 
                               alpha = a, 
                               epsilon = e,
                               delta = 1e-6,
                               max_iter = iter)
    
    print(min_)

    # Coeficientes del polinomio aproximado
    approx_coeffs = min_
    # approx_coeffs = [-0.011229650946330222, -0.00419889384407729, 0.38533405081449215, -0.005108049161826688, -1.1002419929751505, 0.006905270127071032]
    # approx_coeffs = [-6.54727150e-05, 9.87276462e-01, 6.80122959e-05, -1.54990162e-01, -8.15011308e-06, 5.61723721e-03]


    # Valores de x para graficar
    x_values = np.linspace(-np.pi, np.pi, num=100)
    # Valores de la función seno
    sin_values = target_function(x_values)
    # Valores de la función polinómica aproximada
    # approx_values = polynomial_function(approx_coeffs, x_values)
    approx_values = polynomial_function(x_values)

    print(target_function(3))
    print(polynomial_function(3))

    # Graficar el error
    fig, ax = plt.subplots(figsize = (13, 6))
    vals = [loss(*p) for p in orbit]

    ax.plot(vals)
    ax.set_title(f'Valores de S en el Descenso del Gradiente')
    ax.set_xlabel('Epocas')
    ax.set_ylabel('Valor de S')

    # Graficar las funciones
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, sin_values, label='Seno Real')
    plt.plot(x_values, approx_values, label='Aproximación Polinómica')
    plt.title('Comparación entre la función seno real y la aproximación polinómica')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.ylim((-2, 2))
    plt.legend()
    plt.grid(True)
    plt.show()






