import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import aztlan as az

class TreeClassifier(az.Classifier):
    def __init__(self, cost='gini', verbose = True, max_depth = 1, min_size = 1):
        self.__cost = cost
        self.__verbose = verbose
        self.max_depth = max_depth
        self.min_size = min_size 

    @property
    def cost(self):
        return self.__cost
    
    @cost.setter
    def cost(self, new_value):
        if isinstance(new_value, str):
            if new_value.lower in ['gini', 'entropy']:
                self.__cost = new_value.lower()
        else:
            raise ValueError('Cost need to be a string "gini" or "entropy"')
        

    @property
    def verbose(self):
        return self.__verbose
    
    @verbose.setter
    def verbose(self, new_value):
        if isinstance(new_value, bool):
            self.__verbose = new_value
        else:
            raise ValueError('Cost need to be a boolean')

    def __gini_impurity(self, k, column, target):
        mask = column == k
        p = [((target[mask] == c).sum() / len(target[mask])) for c in target[mask].unique()]
        p = np.array(p)
        return 1 - (p**2).sum()
    
    def __gini_index(self, column, target):
        classes = column.unique()
        p = np.array([(column == k).sum() / len(column) for k in classes])
        gini = np.array([self.__gini_impurity(k, column, target) for k in classes])

        return (p * gini).sum()
    
    def __entropy(self, E, column, target):
        mask = column == E
        aux = 0
        for c in target.unique():
            p = (target[mask] == c).sum() / len(target[mask])

            if p != 0:
                aux += p * np.log2(p)
            else:
                return 0
        return -aux

    def __total_entropy(self, target):
        aux = 0
        for c in target.unique():
            p = (target == c).sum() / len(target)

            if p != 0:
                aux += p * np.log2(p)
            else:
                return 0
        return -aux
    
    def __information_gain(self, column, target):
        classes = column.unique()
        p = np.array([(column == k).sum() / len(column) for k in classes])
        h = np.array([self.__entropy(k, column, target) for k in classes])

        return self.__total_entropy(target) - (p * h).sum()
    
    def __get_candidates(self, column, target):
        candidates = []
        index = column.sort_values().index

        column = column[index]
        target = target[index]

        for i in range(1, len(index)):
            if target[index[i]] != target[index[i - 1]]:
                candidates.append(column[index[i]])
        
        return candidates
    
    def __get_split(self, column, target):
        candidates = self.__get_candidates(column, target)

        if self.cost == 'gini':
            ginis = [self.__gini_index(column < c, target) for c in candidates]

            if self.verbose:
                msn = 'Candidate: {} -> Gini: {}'
                for c, g in zip(candidates, ginis):
                    print(msn.format(c, g))

            if ginis == []:
                return None
            
            return candidates[np.argmin(ginis)]

        else: # entropy
            entropy = [self.__information_gain(column < c, target) for c in candidates]

            if self.verbose:
                msn = 'Candidate: {} -> Information Gain: {}'
                for c, g in zip(candidates, entropy):
                    print(msn.format(c, g))

            if entropy == []:
                return None
            
            return candidates[np.argmax(entropy)]
    
    def __make_node(self, X, target):
        if self.cost == 'gini':
            ginis = []
            values = []

            for c in X.columns:
                value = self.__get_split(X[c], target)
                aux = X[c] < value

                ginis.append(self.__gini_index(aux, target))
                values.append(value)

                if self.verbose:
                    msn = 'Column: {} -> Ginis: {}'
                    print(msn.format(c, self.__gini_index(aux, target)))

            column, value = X.columns[np.argmin(ginis)], values[np.argmin(ginis)]
            
            mask = X[column] < value

            left, right = (X[mask], target[mask]), (X[~mask], target[~mask])

            return {'column': column, 'value': value, 'groups': (left, right)}


        else: # entropy
            entropy = []
            values = []

            for c in X.columns:
                value = self.__get_split(X[c], target)
                aux = X[c] < value

                ginis.append(self.__information_gain(aux, target))
                values.append(value)

                if self.verbose:
                    msn = 'Column: {} -> Information Gain: {}'
                    print(msn.format(c, self.__information_gain(aux, target)))

            column, value = X.columns[np.argmax(entropy)], values[np.argmax(entropy)]
            
            mask = X[column] < value

            left, right = (X[mask], target[mask]), (X[~mask], target[~mask])

            return {'column': column, 'value': value, 'groups': (left, right)}
        
    def __terminal_node(self, target):
        values = target.values.tolist()
        return max(values, key = values.count)
    
    def __split(self, node, depth):
        left, right = node['groups']
        del(node['groups'])

        if len(left[0]) == 0:
            node['right'] = self.__terminal_node(right[1])
            return
        
        if len(right[0]) == 0:
            node['left'] = self.__terminal_node(left[1])
            return

        # Check depth
        if depth > self.max_depth:
            node['right'] = self.__terminal_node(right[1])
            node['left'] = self.__terminal_node(left[1])
            return 
        
        # Left child
        if len(left[0]) <= self.min_size:
            node['left'] = self.__terminal_node(left[1])
        else:
            node['left'] = self.__make_node(left[0], left[1])
            self.__split(node['left'], depth + 1)

        # Right Child
        if len(right[0]) <= self.min_size:
            node['right'] = self.__terminal_node(right[1])
        else:
            node['right'] = self.__make_node(right[0], right[1])
            self.__split(node['right'], depth + 1)


    def fit(self, X, target):
        root = self.__make_node(X, target)
        self.__split(root, 1)

        self.tree = root

    def __predict_one(self, node, x):
        if node['value'] == None:
            return node['right']
        
        if x[node['column']] < node['value']:
            if isinstance(node['left'], dict):
                return self.__predict_one(node['left'], x)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.__predict_one(node['right'], x)
            else:
                return node['right']
            
    def predict(self, X):
        predictions = []
        for i in X.index:
            predictions.append(self.__predict_one(self.tree, X.loc[i, :]))

        return predictions




classification   = ['Freshman','Freshman','Sophomore','Junior','Freshman','Sophomore']
hour_of_practice = ['>2h','>2h','>2h','<2h','>2h','<2h']
pass_the_quiz    = ['Yes','Yes','Yes', 'Yes', 'No','No']

data = pd.DataFrame({'Classification':classification, 'hour of practice':hour_of_practice, "Pass the quiz":pass_the_quiz })

np.random.seed(1995)
x1 = np.random.random(5) + 5
y1 = np.random.random(5) + 4

x2 = np.random.random(5) + 4
y2 = np.random.random(5) + 4

x3 = np.random.random(5) + 5
y3 = np.random.random(5) + 5

x4 = np.random.random(5) + 4
y4 = np.random.random(5) + 5


x = list(x1) + list(x2) + list(x3) + list(x4)
y = list(y1) + list(y2) + list(y3) + list(y4)
target = ['g'] * 5 + ['r'] * 15
data_num = pd.DataFrame({'x1' :  x, 'x2' : y, 'y' : target})


if __name__ == '__main__':
    print('Arboles de desiciones: ')
    clf = TreeClassifier()
    clf.fit(data_num[['x1', 'x2']], data_num['y'])


