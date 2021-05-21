from Utils import FancyApp
import pandas as pd
import numpy as np


class InterProParser(FancyApp.FancyApp):

    def __init__(self, filename):
        super(InterProParser, self).__init__()
        self.filename = filename

        self.tell(f'Reading InterPro file: {self.filename}')
        self.features = pd.read_csv(self.filename,
                                    sep='\t',
                                    names=['accession', 'interpro', 'name', 'database_id', 'val1', 'val2'])
        self.features = self.features.drop(columns=['name', 'database_id', 'val1', 'val2'])
        self.features['value'] = 1
        self.features = self.features.drop_duplicates().pivot('accession', 'interpro', 'value').fillna(0)
        self.features = self.features.reset_index()
        self.feature_cols = self.features.columns[
            ~self.features.columns.isin(['accession'])]
        self.tell(f'The InterPro file contains {len(self.feature_cols)} features')

    def __getitem__(self, item):
        condition = self.features['accession'] == item
        if self.features[condition].shape[0] == 0:
            return np.zeros(len(self.feature_cols))
        return self.features[condition][self.feature_cols].values.flatten()