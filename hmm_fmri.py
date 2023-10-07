#!/usr/bin env python

# imports
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from itertools import repeat, chain

# Class for simulated data
class SimSimpData:
    """Class for simulating simple voxel by time data with embedded events.

    Parameters
    ----------
    n_events: int
        Number of events

    n_voxels: int
        Number of voxels

    noise: int
        Embedded variance

    skew: bool, default: False
        Symmetrical or skewed data.
        The HMM will have a harder time fitting skewed data

    Attributes
    ----------
    data: ndarray
        Voxel by time numpy array

    labels: list
        List of event labels used to generate data

    """

    # initialize object
    def __init__(self, n_events, n_voxels, noise, skew=False):
        self.n_events = n_events
        self.n_voxels = n_voxels
        self.noise = noise
        self.skew = skew

    def data(self):
        """Function for actually creating the data
        
        """
        # create empty labels vector
        labels=[]

        # check if skew is True or False and generate event labels
        if self.skew is False:
            for i in range(self.n_events):
                x = list(repeat(i, 5))
                labels.append(x)
        elif self.skew is True:
            _l = np.arange(0, self.n_events)
            x = list(repeat(_l[0], 10))
            labels.append(x)
            for i in range(1, self.n_events):
                x = list(repeat(i, 5))
                labels.append(x)
        
        # clean 
        labels = list(chain.from_iterable(labels))
        
        # get event blocks
        event_blocks = self.n_events + 1

        # make event pattern
        pattern = np.random.rand(event_blocks, self.n_voxels)

        # simulate data
        data = np.zeros((len(labels), self.n_voxels))
        for i in range(len(labels)):
            data[i, :] = pattern[labels[i], :] +\
                self.noise * np.random.rand(self.n_voxels)
        
        # return data and labels to class
        self.data = data
        self.labels = labels
            
    def plot(self, save=False):
        """Plot the data

        Paramters
        ---------
        save: bool, default: False
            Save plot
        """
        plt.imshow(stats.zscore(self.data.T), origin='lower')
        plt.xlabel('Time')
        plt.ylabel('Voxels')
        plt.title('Simulated data with events')
        plt.xticks(np.arange(0, self.data.shape[0], 5))
        boundaries = np.where(np.diff(self.labels))[0] + 0.5 # find boundaries for plotting purposes
        for boundary in boundaries:
            plt.axvline(boundary, 0, 1, color='red')

        plt.legend(handles=[plt.Line2D([0], [0], color='red', label='boundary')])
        plt.tight_layout()

        if save is True:
            plt.savefig('data.png')