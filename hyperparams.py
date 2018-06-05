import numpy as np
from settings import *

class Hyperparams():

    def __init__(self):
        pass

    def getFilters(self):
        self.filter_range_city = np.arange(nr_cities)+1
        self.filter_range_time = np.array( [2,3,4,5,6,7] )

    def generateChannelSizes(self, nr_convs):
        channel_sequence = np.random.uniform(5,35,nr_convs)
        return channel_sequence



    def getRandomHyperparameter(self):
        self.getFilters()
        channel_sequence = self.generateChannelSizes()
        hyperparams = {
            "filter" : [ self.filter_range_city[0], self.filter_range_time[0]],
            "nr_convs" : 3,
            "channel_sequence" : channel_sequence
        }
        return hyperparams


