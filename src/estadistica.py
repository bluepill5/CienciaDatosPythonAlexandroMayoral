import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from scipy import stats
from sympy import *

from functions_calc import *


if __name__ == '__main__':
    print('Estadistica')
    h0 = stats.norm(25, 2)
    h1 = stats.norm(28, 2)

    x = np.linspace(10, 40, 1000)

    fig, ax = plt.subplots(figsize = (13, 6))

    ax.plot(x, h0.pdf(x), label = '$H_0$')
    ax.plot(x, h1.pdf(x), label = '$H_1$')
    ax.axvline(21, color = 'r', alpha = 0.5)
    ax.axvline(29, color = 'r', alpha = 0.5)
    # ax.fill_between(x, h0.pdf(x), where=(x <= 23), color = 'blue', alpha = 0.3)
    # ax.fill_between(x, h0.pdf(x), where=(x > 29), color = 'blue', alpha = 0.3)
    ax.fill_between(x, h0.pdf(x), where=((x <= 27) & (x > 23)), color = 'blue', alpha = 0.3)
    ax.legend()
    plt.show()

    print(h0.cdf(23))
    print(1 - h0.cdf(29))
    print(h0.cdf(27) - h0.cdf(23))

    






