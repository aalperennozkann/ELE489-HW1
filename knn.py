import numpy as np
from collections import Counter

class KNNClassifier:
    def __init__(self, k=3, distance_metric="euclidean"):
        self.k = k
        self.metric = distance_metric

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for test_point in X_test:
            distances = []

            for i in range(len(self.X_train)):
                if self.metric == "euclidean":
                    d = np.linalg.norm(test_point - self.X_train[i])
                elif self.metric == "manhattan":
                    d = np.sum(np.abs(test_point - self.X_train[i]))
                elif self.metric == "chebyshev":
                    d = np.max(np.abs(test_point - self.X_train[i]))
                else:
                    raise ValueError("Unknown distance metric")

                distances.append((d, self.y_train[i]))

            # Sort distances and pick k nearest
            distances.sort(key=lambda x: x[0])
            neighbors = [label for (_, label) in distances[:self.k]]

            # Vote
            vote = Counter(neighbors).most_common(1)[0][0]
            predictions.append(vote)

        return np.array(predictions)
