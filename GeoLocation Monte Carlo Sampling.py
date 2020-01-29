import os
import pandas as pd
import statistics
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt
import numpy as np



class Monte_Carlo_Sampling(Preprocessor):

    def __init__(self, iterations=None, campaignHorizonPath=None, poiPath=None, date=None, campaignReference=None):
        super().__init__(campaignHorizonPath=campaignHorizonPath, poiPath=poiPath, date=date,
                         campaignReference=campaignReference)
        self.score = 0.1
        self.iterations = iterations
        self.noImpression = {}

    def count_impressions(self):
        campDataFrame = self._get_filtered_campaign_dataframe().to_numpy()
        impressions = [row[0] for row in campDataFrame]
        for key in np.unique(impressions):
            self.noImpression.update({f'{key}': impressions.count(key)})
        return self.noImpression

    def fit(self):
        distances = self._get_distance().to_numpy()
        r = self._get_poi_dataframe()['radius'].to_numpy()
        self.noImpression = self.count_impressions()
        cauchyDistribution = np.linspace(cauchy.ppf(0.01), cauchy.ppf(0.99), 100)

        for iteration in range(self.iterations):
            distanceSet = [np.random.choice(cauchyDistribution) for i in range(self.noImpression['benjm001'])]
            originalDistances = [distance[-1] for distance in distances]
            di = [d + e for d, e in zip(originalDistances, distanceSet)]
            currentScore = len([value for value in di if value < r[0]]) / len(di)
            optimised = False
            if abs((currentScore - self.score) / self.score) < 0.05:
                optimised = True

            self.score = statistics.mean([self.score, currentScore])

            if optimised == True:
                break
