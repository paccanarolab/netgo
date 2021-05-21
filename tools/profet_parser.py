from Utils import FancyApp
import pandas as pd
import numpy as np


class ProFETParser(FancyApp.FancyApp):

    def __init__(self, filename):
        super(ProFETParser, self).__init__()
        self.filename = filename

        self.tell(f'Reading ProFET file: {self.filename}')
        self.features = pd.read_csv(self.filename)
        # here, we assume that proteinname is of the uniprotkb format, and that
        # we are dealing with accession numbers
        self.features['proteinname'] = self.features['proteinname'].str.split('|').str[1]
        self.tell(f'Loaded ProFET features for {self.features.shape[0]} proteins')

        self.feature_cols = self.features.columns[
            ~self.features.columns.isin(['proteinname', 'classname'])]
        self.tell(f'The ProFET file contains {len(self.feature_cols)} features')

    def __getitem__(self, item):
        condition = self.features['proteinname'] == item
        if self.features[condition].shape[0] == 0:
            return np.zeros(len(self.feature_cols))
        return self.features[condition][self.feature_cols].values.flatten()
