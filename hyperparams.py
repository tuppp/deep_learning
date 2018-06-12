import numpy as np
from settings import *

class Hyperparams():

    def __init__(self):
        pass

    def getPoolSize(self):
        self.poolsize_range_city = np.arange(nr_cities)+1
        self.poolsize_range_time = np.array([2, 3, 4, 5, 6, 7])

    def getFilters(self):
        self.filter_range_city = np.arange(nr_cities)+1
        self.filter_range_time = np.array( [2,3,4,5,6,7] )

    def generateChannelSizes(self, nr_convs):
        channel_sequence = np.random.uniform(5,35,nr_convs)
        return channel_sequence



    def getRandomHyperparameter(self):
        self.getFilters()
        self.getPoolSize()
        channel_sequence = self.generateChannelSizes()
        hyperparams = {
            "ksize" : [ np.random.choice(self.poolsize_range_city), np.random.choice(self.poolsize_range_time)],
            "filter" : [ np.random.choice(self.filter_range_city), np.random.choice(self.filter_range_time)],
            "nr_convs" : 3,
            "channel_sequence" : channel_sequence
        }
        return hyperparams


