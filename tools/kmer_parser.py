from Utils import FancyApp
import pandas as pd
import numpy as np


class KMerParser(FancyApp.FancyApp):

    def __init__(self, filename):
        super(KMerParser, self).__init__()
        self.filename = filename

        self.tell(f'Reading KMer file: {self.filename}')
        self.features = pd.read_csv(self.filename,
                                    sep='\t',
                                    names=['accession', 'kmer', 'value'])
        self.features = self.features.drop_duplicates().pivot(
            'accession', 'kmer', 'value').fillna(0)
        self.features = self.features.reset_index()
        self.feature_cols = self.features.columns[
            ~self.features.columns.isin(['accession'])]
        self.tell(f'The KMer file contains {len(self.feature_cols)} features')

    def __getitem__(self, item):
        condition = self.features['accession'] == item
        if self.features[condition].shape[0] == 0:
            return np.zeros(len(self.feature_cols))
        return self.features[condition][self.feature_cols].values.flatten()
