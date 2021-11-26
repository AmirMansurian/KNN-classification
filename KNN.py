import numpy as np
import pandas

class KNN :

    def __init__(self, K):
        self.K = K


    def fit(self, X, Y):

        distances = list()
        for i in range(X.shape[0]):
            dis = self.distance(X.iloc[i, :], Y)
            distances.append((i, dis))
        distances.sort(key=lambda tup: tup[1])
        neighbors = list()

        for i in range(self.K):
            neighbors.append(distances[i][0])

        return neighbors


    def predict(self, X, Y_train,  Y_test):

        predic = [0]*Y_test.count()[0]
        for i in range(Y_test.count()[1]):
            neighbors = self.fit(X, Y_test.iloc[i, :])
            output_values = [res for res in Y_train.iloc[neighbors]]
            predic[i] = max(set(output_values), key=output_values.count)

        return predic


    def distance(self, X1, X2):

        return ((X1 - X2) ** 2).sum() / X1.count()