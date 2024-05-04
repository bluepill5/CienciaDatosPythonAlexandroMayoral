import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
import aztlan as az

class HierarchicalClustering:
    def __init__(self, p=2, link='centroid'):
        self.p = p

        if link == 'centroid':
            self.__link = self.__centroid_link
        elif link == 'single':
            self.__link = self.__single_link
        elif link == 'complete':
            self.__link = self.__complete_link
        elif link == 'average':
            self.__link = self.__average_link

        self.link_matrix = []


    def __distance(self, p1, p2):
        p1 = np.array(p1)
        p2 = np.array(p2)

        return ((np.abs(p1 - p2) ** self.p).sum()) ** (1 / self.p)
    
    def __single_link(self, cluster1, cluster2):
        distances = [self.__distance(p1, p2) for p1 in cluster1 for p2 in cluster2]
        return min(distances)
    
    def __complete_link(self, cluster1, cluster2):
        distances = [self.__distance(p1, p2) for p1 in cluster1 for p2 in cluster2]
        return max(distances)
    
    def __average_link(self, cluster1, cluster2):
        distances = [self.__distance(p1, p2) for p1 in cluster1 for p2 in cluster2]
        return np.mean(distances)
    
    def __centroid_link(self, cluster1, cluster2):
        cluster1 = np.array(cluster1)
        cluster2 = np.array(cluster2)

        if len(cluster1) == 1:
            c1 = cluster1
        else:
            c1 = np.mean(cluster1, axis = 0)

        if len(cluster2) == 1:
            c2 = cluster2
        else:
            c2 = np.mean(cluster2, axis = 0)

        return self.__distance(c1, c2)

    def __find_cluster(self, clusters):
        if len(clusters) == 1:
            return clusters
        
        points = [c[1] for c in clusters]

        p_ = []
        q_ = []
        distance = 1e99

        for p in points:
            for q in points:
                if np.array_equal(p, q):
                    continue
                else:
                    if np.array_equal(p, q_) and np.array_equal(q, p_):
                        continue
                    else:
                        d = self.__link(p, q)

                        if d < distance:
                            distance, p_, q_ = d, p, q
                        
                        else:
                            continue
        index = max([c[0] for c in clusters]) + 1
        new_cluster = []

        for element in p_:
            new_cluster.append(element)

        for element in q_:
            new_cluster.append(element)

        new_cluster = [index, new_cluster, distance]

        clusters = [c for c in clusters if not np.array_equal(c[1], p_) and not np.array_equal(c[1], q_)]
        clusters.append(new_cluster)

        self.link_matrix.append(new_cluster)

        return self.__find_cluster(clusters)


    def fit(self, X):
        clusters = [[i, list(p), 0] for i, p in enumerate(X)]
        self.__find_cluster(clusters)

        

if __name__ == '__main__':
    print('Herarchical Clustering: ')
    X = np.array([[5,3],
    [10,15],
    [15,12],
    [24,10],
    [30,30],
    [85,70],
    [71,80],
    [60,78],
    [70,55],
    [80,91],])

    fig, ax = plt.subplots(figsize = (13, 8))
    labels = range(1, 11)

    ax.scatter(X[:,0], X[:,1], label = 'True Position')

    for label, x, y in zip(labels, X[:, 0], X[:, 1]):
        plt.annotate(
            label,
            xy=(x, y), xytext = (-3, 3),
            textcoords ='offset points', ha = 'right', va = 'bottom')
    
    plt.show()

    model = HierarchicalClustering()
    model.fit(X)

    print(model.link_matrix)





