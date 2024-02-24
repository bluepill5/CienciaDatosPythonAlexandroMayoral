import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

dist = stats.binom(10, 1/6)
dist_geom = stats.geom(1/6)
dist_poi = stats.poisson(3)

N, d, n = 52, 4, 5

dist_hiper = stats.hypergeom(N, d, n)



if __name__ == '__main__':
    print('Probabilidades:')

    # print(dist.pmf(5))
    # print(dist.pmf(1))
    # print(dist.rvs(15))
    # print(dist.mean())
    # print(dist.var())

    # print(dist_geom.pmf(2))

    # print(dist_poi.mean())
    # print(dist_poi.var())
    # print(dist_poi.pmf(8))

    print(dist_hiper.pmf(4) / (dist_hiper.pmf(4) + dist_hiper.pmf(3)))
    
    x1, x2, x3 = np.arange(51), np.arange(51), np.arange(51)
    bi1, bi2, bi3 = stats.binom(50, 1/6), stats.binom(100, 1/6), stats.binom(150, 1/6)
    y1, y2, y3 = bi1.pmf(x1), bi2.pmf(x1), bi3.pmf(x3)

    fig, ax = plt.subplots(3, 1, figsize = (13, 6))
    ax[0].plot(x1, y1, linestyle = '', marker = 'o', color = 'b')
    ax[0].axvline(bi1.mean(), color = 'b')
    ax[1].plot(x2, y2, linestyle = '', marker = 'o', color = 'r')
    ax[1].axvline(bi2.mean(), color = 'r')
    ax[2].plot(x3, y3, linestyle = '', marker = 'o', color = 'g')
    ax[2].axvline(bi3.mean(), color = 'g')

    
    x1, x2, x3 = np.arange(51), np.arange(51), np.arange(51)
    bi1, bi2, bi3 = stats.geom(1/6), stats.geom(1/6), stats.geom(1/6)
    y1, y2, y3 = bi1.pmf(x1), bi2.pmf(x1), bi3.pmf(x3)

    fig, ax = plt.subplots(3, 1, figsize = (13, 6))
    ax[0].plot(x1, y1, linestyle = '', marker = 'o', color = 'b')
    ax[0].axvline(bi1.mean(), color = 'b')
    ax[1].plot(x2, y2, linestyle = '', marker = 'o', color = 'r')
    ax[1].axvline(bi2.mean(), color = 'r')
    ax[2].plot(x3, y3, linestyle = '', marker = 'o', color = 'g')
    ax[2].axvline(bi3.mean(), color = 'g')


    x1, x2, x3 = np.arange(51), np.arange(51), np.arange(1200)
    bi1, bi2, bi3 = stats.poisson(6), stats.poisson(15), stats.poisson(1000)
    y1, y2, y3 = bi1.pmf(x1), bi2.pmf(x1), bi3.pmf(x3)

    fig, ax = plt.subplots(3, 1, figsize = (13, 6))
    ax[0].plot(x1, y1, linestyle = '', marker = 'o', color = 'b')
    ax[0].axvline(bi1.mean(), color = 'b')
    ax[1].plot(x2, y2, linestyle = '', marker = 'o', color = 'r')
    ax[1].axvline(bi2.mean(), color = 'r')
    ax[2].plot(x3, y3, linestyle = '', marker = 'o', color = 'g')
    ax[2].axvline(bi3.mean(), color = 'g')

    plt.show()



