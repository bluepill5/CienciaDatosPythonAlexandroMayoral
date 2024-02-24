import numpy as np
import matplotlib.pyplot as plt
from scipy import stats



if __name__ == '__main__':
    print('Probabilidades:')

    dist = stats.uniform(-2, 4) # (donde empiezas, cual es el ancho del escalon)

    fig, ax = plt.subplots(figsize = (13, 6))
    x = np.linspace(-3, 3, 1000)
    ax.plot(x, dist.pdf(x))

    dist = stats.norm(1.68, 0.15)

    n = 3
    fig, ax = plt.subplots(figsize = (13, 6))
    x = np.linspace(1, 2.5, 1000)
    ax.plot(x, dist.pdf(x))
    ax.axvline(dist.mean(), color = 'r')
    ax.axvline(dist.mean() + n  * dist.std(), color = 'y')
    ax.axvline(dist.mean() - n  * dist.std(), color = 'y')
    
    intervalo = ((dist.mean() - n * dist.std()) <= x) & (x <= (dist.mean() + n  * dist.std()))
    ax.fill_between(x, dist.pdf(x), where = intervalo, alpha = 0.3)

    probabilidades = np.array([1/4, 1/4, 1/8, 1/8, 1/8, 1/8])
    values = np.array([1, 2, 3, 4, 5, 6])

    # print(values.dot(probabilidades))
    E = values @ probabilidades

    population = np.random.choice(values, 1_000_000, p = probabilidades)

    k = 50
    n = 5_000

    sample_means = []

    for i in range(n):
        sample = np.random.choice(population, size = i + 1)
        sample_means.append(sample.mean())

    fig, ax = plt.subplots(figsize = (13, 8))

    ax.plot(sample_means, label = 'Sample Mean')
    ax.axhline(E, linewidth = 4, color = 'r', alpha = 0.5, label = 'Expected Value')
    ax.legend()

    # Ejemplo estaturas
    dist = stats.norm(1.68, 0.15)
    population = dist.rvs(120_000)

    sample_size = 30
    number_exp = 500

    sample_means = []

    for i in range(number_exp):
        sample = np.random.choice(population, size = sample_size)
        sample_means.append(sample.mean())

    sample_means = np.array(sample_means)

    # np.cumsum(sample_means) / np.arange(1, number_exp + 1)

    fig, ax = plt.subplots(2, 1, figsize = (13, 6))
    
    ax[0].plot(sample_means, label = 'Sample Means', marker = 's')
    ax[0].axhline(dist.mean(), color = 'r', linewidth = 4, alpha = 0.5, label = 'Population Mean')
    ax[0].legend()

    ax[1].plot(np.cumsum(sample_means) / np.arange(1, number_exp + 1), label = 'Sample Means', marker = 's')
    ax[1].axhline(dist.mean(), color = 'r', linewidth = 4, alpha = 0.5, label = 'Population Mean')
    ax[1].legend()

    population_size = 1_000
    dist = stats.poisson(5)
    population = dist.rvs(population_size) ** 2

    # fig, ax = plt.subplots(2, 1, figsize = (13, 6))

    import pandas as pd

    data = pd.DataFrame({'X': population})
    data['X'].sample(15).mean()

    sample_size = 15
    number_exp = 500

    sample_means = [data['X'].sample(sample_size).mean() for i in range(number_exp)]
    sample_means = np.array(sample_means)

    fig, ax = plt.subplots(figsize = (13, 6))

    x = np.linspace(0, 2, 1000)
    dist = stats.norm(sample_means.mean(), sample_means.std())

    ax.plot(x, dist.pdf(x), color = 'r')
    ax.hist(sample_means, bins = 50, density = True)








    plt.show()



