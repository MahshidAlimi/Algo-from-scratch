import os
import pandas as pd
import statistics
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt
import numpy as np


class Preprocessor(object):
    def __init__(self, campaignHorizonPath=None, poiPath=None, date=None, campaignReference=None):
        self.campaignHorizonPath = campaignHorizonPath
        self.poiPath = poiPath
        self.date = date
        self.campaignReference = campaignReference

    @staticmethod
    def haversine(lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(radians, [float(lon1), float(lat1), float(lon2), float(lat2)])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        r = 6371
        return c * r

    @staticmethod
    def _filter_rogue_lat_lon(lat, lon):
        flag_good_coord = 0
        if (lon >= -180.0) and (lon <= 180.0) and (lat >= -90.0) and (lat <= 90.0):
            flag_good_coord = 1
        return flag_good_coord

    def _get_poi_dataframe(self):
        pois = pd.read_csv(self.poiPath, error_bad_lines=False)
        pois = pois.rename(columns={'Latitude': 'poi_lat', 'Longitude': 'poi_lon', 'Radius': 'radius',
                                    'Campaign Reference': 'clientId'})
        pois = pois.loc[pois.clientId == f'{self.campaignReference}']
        pois['radius'] = pois['radius'] / 1000.0
        return pois

    def _get_filtered_campaign_dataframe(self):
        campaignData = pd.read_csv(self.campaignHorizonPath, error_bad_lines=False)
        selectColumns = ['clientId', 'latitude', 'longitude', 'date', 'placement']
        fc = campaignData[selectColumns].copy()
        fc['flag_good_coord'] = fc.apply(lambda x: self._filter_rogue_lat_lon(x['latitude'], x['longitude']), axis=1)
        fc = fc[fc['flag_good_coord'] == 1]
        # fc = fc.loc[(fc.clientId == f'{self.campaignReference}') & (fc.placement != 'Blis')]
        return fc

    def _get_count_dataframe(self):
        filteredCampaignDf = self._get_filtered_campaign_dataframe()
        filteredCampaignDf.groupby('date')['latitude'].count()
        filteredCampaignDf = filteredCampaignDf[filteredCampaignDf['date'] == self.date]
        filteredCampaignDf['location'] = filteredCampaignDf.apply(
            lambda x: str(x['latitude']) + '_' + str(x['longitude']), axis=1)
        countsDataframe = filteredCampaignDf.groupby(['location', 'clientId'])['latitude'].count().reset_index()
        countsDataframe = countsDataframe.rename(columns={'latitude': 'count_centroids'})
        return countsDataframe.sort_values('count_centroids', ascending=False)

    def _get_distance(self):
        poisDataframe = self._get_poi_dataframe()
        countsDevDataframe = self._get_count_dataframe()
        countsDevDataframe[['latitude', 'longitude']] = countsDevDataframe['location'].str.split('_', expand=True)
        locationDataframe = countsDevDataframe[['clientId', 'location', 'latitude', 'longitude']]
        mergedLocPoiDf = pd.merge(locationDataframe, poisDataframe, on='clientId')
        mergedLocPoiDf = mergedLocPoiDf.drop(['Target'], axis=1)
        mergedLocPoiDf['distance'] = mergedLocPoiDf.apply(
            lambda x: Preprocessor.haversine(x['longitude'], x['latitude'], x['poi_lon'], x['poi_lat']), axis=1)
        return mergedLocPoiDf
    
    
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
