import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
import pytest


class KMeans:
    def __init__(self, data=None, test=None, k=None, tol=0.001, max_iter=300):
        self.data = data
        self.test = test
        self.k = k
        # how much the centroid is going to move. percentage change.
        self.tol = tol
        self.max_iter = max_iter
        self.centroids = {}

    @staticmethod
    def _calc_eucleadian_distance(x, y):
        assert type(x, y) in [int, float, np.ndarray], "The input feature can only be a float or an integer."
        return ((x[0]-y[0])**2 + (x[1]-y[1])**2)**0.5

    # @pytest.mark.parametrize("dtype", [np.ndarray])
    @staticmethod
    def _average(features):
        assert [isinstance(feature, (np.ndarray, float, int)) for feature in features], "The input feature can only be a " \
                                                                                        "float or an integer."
        # print(isinstance(features, list))
        # print(features.dtype())
        assert len(features) != 0, "There are no data points passed in."
        return sum(features) / len(features)

    #  Euclidean norms
    @staticmethod
    def _calc_vector_magnitude(points):
        assert [isinstance(point, (np.ndarray, float, int)) for point in points], "The input feature can only be a " \
                           "float or an integer."
        return (sum(point ** 2 for point in points))**0.5

    def _update_centroids(self):
        for cluster in range(self.k):
            self.centroids[cluster] = self.data[cluster]
        return self.centroids

    def _get_index_min_distance(self, data):
        # assert len(self.centroids) != 0, "The list of centroids is empty."
        # if we have two clusters, we would need two centroids, the distances will include the subtracted value
        # of each point to each of these centroids. This line needs to be modified if we are dealing with
        # Eucleadian distance.
        distances = [KMeans._calc_vector_magnitude(points=data - self.centroids[centroid]) for centroid in self.centroids]
        # get the index value of the minimum of distances, in order to classify each point to their nearest
        # centroid.
        return distances.index(min(distances))

    def _get_clutser_centers(self):
        return [self.centroids[centerPoint] for centerPoint in self.centroids]

    def _get_labels(self):
        return [self._get_index_min_distance(data=point) for point in self.data]

    def fit(self):
        # iterates through data to get centroids
        self.centroids = self._update_centroids()
        # optimisation process
        # keys will be the centroids and the values will be the feature set.
        for iteration in range(self.max_iter):
            #  for every iteltartion the cluster_assignments are going to be cleared out as they will change with every
            # iteration.
            # this will contain keys for the selected centroids at each round.
            self.cluster_assignments = {}
            # this will hold the classified points.

            self.cluster_assignments = {cluster: [] for cluster in range(self.k)}
            for point in self.data:
                #  here we will compute the distance between each point and the centroids and take the index of
                #  whichever centroid they are closest to.
                minDistanceIndex = self._get_index_min_distance(data=point)


                # points belongs to that centroid.
                self.cluster_assignments[minDistanceIndex].append(point)
            # print(self.cluster_assignments)
            # this will used for to compare how much centroids are changed. if we directly equalise this to centroids
            # then it will always copy the centroids set and change as self.centroids changed.
            prev_centroids = dict(self.centroids)

            # here we are trying to find the new centroids, taking the average points that are classified together.
            for clusteredPoint in self.cluster_assignments:
                self.centroids[clusteredPoint] = KMeans._average(features=self.cluster_assignments[clusteredPoint])

            # we assume the process is optimised and if through the next loop we find that it isn't, the flag will turn
            # to false.

            optimized = True

            #  here we are comparing the new centroids with the original centroids to see if they have moved at all.
            #  the tolerance between the difference of the two centroids is the self.tol value which is a hyperparameter
            #  we have already pre-defined and defines the percentage changed.

            for centerPoints in self.centroids:
                original_centroid = prev_centroids[centerPoints]

                current_centroid = self.centroids[centerPoints]
                # assert original_centroid > 0, "cannot divide by zero"
                # print(current_centroid, original_centroid)
                if sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    # if any of these centroids move more than it is tolerated, we tag the process as not optimised.
                    optimized = False

            # the process is optimised we break out of the for iterating loop, and the final centroids is hence chosen.
            while optimized:
                break

    def predict(self, unseen):
        return self._get_index_min_distance(data=unseen)


X = np.array([
    [1, 2],
    [1.5, 1.8],
    [5, 8],
    [8, 8],
    [1, 0.6],
    [9, 11]
])

unknowns = np.array([
    [1, 3],
    [8, 9],
    [0, 3],
    [5, 4],
    [6, 4]
])

colors = 10 * ['green', 'red', 'cyan', 'blue', 'black', 'yellow', 'magenta']

clf = KMeans(data=X, test=unknowns, k=2)
clf.fit()

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1], marker='o', color='k', s=150, linewidths=5)

for classification in clf.cluster_assignments:
    color = colors[classification]
    for featureSet in clf.cluster_assignments[classification]:
        plt.scatter(featureSet[0], featureSet[1], marker='x', color=color, s=150, linewidths=5)


for unknown in unknowns:
    classification = clf.predict(unseen=unknown)
    plt.scatter(unknown[0], unknown[1], marker='*', color=colors[classification], s=150, linewidths=5)
plt.show()
