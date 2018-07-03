import numpy as np
from settings import *
import pdb

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
        channel_sequence = np.random.randint(5,35,nr_convs)
        return channel_sequence




    def getRandomHyperparameter(self):
        possible_conv_layers = np.array([4,5,6])
        nr_convs = np.random.choice(possible_conv_layers)
        self.getFilters()
        self.getPoolSize()
        channel_sequence = self.generateChannelSizes(nr_convs)
        hyperparams = {
            "nr_neurons_in_convlayer" : 200,
            "ksize" : [ np.random.choice(self.poolsize_range_city), np.random.choice(self.poolsize_range_time)],
            "filter" : [ np.random.choice(self.filter_range_city), np.random.choice(self.filter_range_time)],
            "nr_convs" : nr_convs,
            "channel_sequence" : channel_sequence,
            "nr_fully_connected_layers" : 3,
            "learning_rate" : 0.001,
            "batch_size" : 500,
            "batches_per_fold" : 10
            }
        return hyperparams


