import numpy as np
from tensorflow.keras.utils import Sequence

"""
Keras Fasta Generator

This implements a generator compatible with the Keras model implemented
in prot2vec adapting the code that can be found in the following tutorial:
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
"""

__author__ = 'Mateo Torres'
__email__ = 'Mateo.Torres.2015@live.rhul.ac.uk'
__copyright__ = 'Copyright (c) 2018, Mateo Torres'
__license__ = 'MIT'
__version__ = '1.0'


class KerasFastaGenerator(Sequence):

    def __init__(self, deep_fasta, neighbourhoods, neighbourhood_size, batch_size=32, shuffle=True):
        """
        Initialiser for the class

        Parameters
        ----------
        deep_fasta : DeepFasta instance
            A DeepFasta instance with the encoded sequences
        neighbourhoods : list of protein neighbourhoods
            these will be used to generate the data by batches
        batch_size : int
        shuffle : boolean
            whether to shuffle the order of data samples for each epoch
        """
        #super(KerasFastaGenerator, self).__init__()
        self.deep_fasta = deep_fasta
        self.neighbourhoods = neighbourhoods
        self.neighbourhood_size = neighbourhood_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.neighbourhoods))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.neighbourhoods)) / self.batch_size)

    def __getitem__(self, index):
        """
        generate one batch data

        Parameters
        ----------
        index : int
            the number of batch to retrieve

        Returns
        -------
        X : numpy.array
            training samples in one-hot encoded format
        y : list of numpy arrays
            training labels in one-hot encoded format
        """
        # generate indices for this batch
        indices = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # find list of ids
        samples = [self.neighbourhoods[k] for k in indices]

        # generate data
        return self.__data_generation(samples)

    def on_epoch_end(self):
        """
        Updates indices after each epoch
        """
        self.indexes = np.arange(len(self.neighbourhoods))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, neighbours):
        """
        Generates the encoding data for the batch of samples
        Parameters
        ----------
        neighbours : list
            The list of protein ids to generate the data from

        Returns
        -------
        X : numpy.array
            training samples in one-hot encoded format
        y : list of numpy arrays
            training labels in one-hot encoded format
        """
        # Initialisation
        train_proteins = []
        interactors = {i: [] for i in range(self.neighbourhood_size)}
        for prot, neighs in [(n[0], n[1:1+self.neighbourhood_size]) for n in neighbours]:
            train_proteins.append(np.array(self.deep_fasta.data[prot]).T)
            for i, p in enumerate(neighs):
                interactors[i].append(np.array(self.deep_fasta.data[p]).T)

        train_proteins = np.stack(train_proteins)
        for i in interactors.keys():
            interactors[i] = np.stack(interactors[i])

        return train_proteins, list(interactors.values())