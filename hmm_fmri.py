#!/usr/bin/env brainiak_env

# imports
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from itertools import repeat, chain
from dataclasses import dataclass

# Class for simulated data
@dataclass
class SimSimpData:
    """Class for simulating simple voxel by time data with embedded events.

    Parameters
    ----------
    n_events: int
        Number of events

    size: tuple
        Voxel x timepoints

    noise: int
        Embedded variance

    skew: bool, default: False
        Symmetrical or skewed data.
        The HMM will have a harder time fitting skewed data

    Optional Keyword Args
    ---------------------
    skewf: int
        Skew factor; 1 = high skew, 10 = low skew

    Attributes
    ----------
    data: ndarray
        Voxel by time numpy array

    labels: list
        List of event labels used to generate data

    """
    # initialize object
    def __init__(self, n_events, size: tuple, noise: int, skew=False, **kwargs):
        self.n_events = n_events
        self.size = size
        self.noise = noise
        self.skew = skew
        self.n_voxels = size[0]
        self.n_timepts = size[1]
        self.skewf = kwargs.get('skewf')

    def data(self):
        """Function for actually creating the data
        
        """

        # create empty labels vector
        labels=[]

        # check if skew is True or False and generate event labels
        if self.skew is False:
            for i in range(self.n_events):
                x = list(repeat(i, self.n_timepts))
                labels.append(x)
        elif self.skew is True:
            _l = np.arange(0, self.n_events)
            x = list(repeat(_l[0], self.n_timepts))
            labels.append(x)
            for i in range(1, self.n_events):
                x = list(repeat(i, self.skewf))
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

        return self
            
    def plot(self, save=False):
        """Plot the data

        Paramters
        ---------
        save: bool, default: False
            Save plot
        """
        plt.imshow(stats.zscore(self.data.T), origin='lower', aspect='auto')
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

@dataclass
class Dataset:
    """Create dataset based on simple data
    
        Parameters
        ----------
        base: obj
            SimSimpData object

        n: int
            number of subjects to include in dataset
    """
    def __init__(self, base, n):

        # initialize dataset attributes based on single subject (i.e., SimSimpData class)
        self.n_voxels = base.n_voxels
        self.dataset = np.empty((n), dtype="object")
        self.n_events = base.n_events
        self.size = base.size
        self.noise = base.noise
        self.skew = base.skew
        self.n_voxels = base.size[0]
        self.n_timepts = base.size[1]
        try:
            self.skewf = base.skewf
        except:
            ValueError

        self.dataset[0] = base

        # create empty labels vector
        labels=[]

        # check if skew is True or False and generate event labels
        if base.skew is False:
            for i in range(self.n_events):
                x = list(repeat(i, self.n_timepts))
                labels.append(x)
        elif base.skew is True:
            _l = np.arange(0, self.n_events)
            x = list(repeat(_l[0], self.n_timepts))
            labels.append(x)
            for i in range(1, self.n_events):
                x = list(repeat(i, self.skewf))
                labels.append(x)

        # clean 
        labels = list(chain.from_iterable(labels))
        self.labels = labels

        # get event blocks
        self.event_blocks = self.n_events + 1

    def make_dataset(self):
        """Method call that creates dataset"""

        dataset = self.dataset

        for i in range(1, np.size(self.dataset)):

            # make event pattern
            pattern = np.random.rand(self.event_blocks, self.n_voxels)

            # simulate data
            data = np.zeros((len(self.labels), self.n_voxels))
            for j in range(len(self.labels)):
                data[j, :] = pattern[self.labels[j], :] +\
                    self.noise * np.random.rand(self.n_voxels)
                
            dataset[i] = data

        self.dataset = dataset
        return self

