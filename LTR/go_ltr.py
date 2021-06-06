from Utils import FancyApp
from xgboost import DMatrix
from component_methods import UntrainedComponentError, ComponentMethod
import numpy as np
import pandas as pd
import xgboost as xgb

import os


class LearnToRankGO(ComponentMethod):

    def __init__(self, mode='netgo'):
        """Learning to Rank for GO terms

        Parameters
        ----------
        mode : str, optional
            determines which component model are considered for LTR.
            'netgo' uses all component models, and 'golabeler' ignores
            Net-kNN.
            by default 'netgo'
        """
        super(LearnToRankGO, self).__init__()
        self.mode_ = mode
        self.trained_ = False
        self.model_ = None

    def _build_dataset(self, component_models_dir):
        models = ['BLAST-kNN.tsv', 'GO_frequency-per_domain.tsv', 'LR-InterPro.tsv', 'LR-kmer.tsv', 'LR-ProFET.tsv']
        if self.mode_ == 'netgo':
            models.append('NET-kNN.tsv')
        ltr_dataset = None
        for m in models:
            model_name = m.split('.')[0]
            self.tell(f'adding {model_name} to the feature set')
            df = pd.read_csv(os.path.join(component_models_dir, m), sep='\t').rename(
                columns={'score': model_name})
            if ltr_dataset is None:
                ltr_dataset = df
            else:
                ltr_dataset = ltr_dataset.merge(df,
                                                left_on=['protein', 'goterm', 'domain'],
                                                right_on=['protein', 'goterm', 'domain'],
                                                how='outer').fillna(0.)
        return models, ltr_dataset

    def train(self, function_assignment, **kwargs):
        """

        Parameters
        ----------
        function_assignment : pandas DataFrame
            functional assignment used for establishing the relevance scores (training + LTR are expected)
        kwargs:
            Required information to train this model:
            * component_models_dir : str
                Path to the folder containing the output of the training models in TSV format
        """
        component_models_dir = kwargs['component_models_dir']
        function_assignment['relevance'] = 1.0

        self.tell('Building LTR training dataset')
        models, ltr_dataset = self._build_dataset(component_models_dir)
        ltr_dataset = ltr_dataset.merge(function_assignment,
                                        left_on=['protein', 'goterm'],
                                        right_on=['protein', 'goterm'],
                                        how='left').fillna(0.)
        features = [i.split('.')[0] for i in models]
        X = ltr_dataset[features].values
        y = ltr_dataset['relevance'].values
        train_dmatrix = DMatrix(X, y)
        params = {'objective': 'rank:pairwise', 'max_depth': 3}
        self.tell('Training LTR model using xgboost')
        self.model_ = xgb.train(params, train_dmatrix, num_boost_round=4)
        self.trained_ = True

    def save_trained_model(self, output, **kwargs):
        """
        Saves the trained model onto disk.
        Parameters
        ----------
        output : str
            Where to store the model, it could be a file or a directory
            depending on the model.
        """
        if not self.trained_:
            self.warning('The model is not trained, cannot save the model')
            raise UntrainedComponentError
        self.tell('Saving BLAST-kNN matrix')
        self.model_.save_model(output)

    def load_trained_model(self, model_filename, **kwargs):
        """
        Recovers the state of the component model after training, in order to
        make predictions without re-training the component

        Parameters
        ----------
        model_filename : str
            Filename to store the frequency information
        """
        self.tell(f'Loading LTR model from {model_filename}')
        self.model_ = xgb.Booster()
        self.model_.load_model(model_filename)
        self.trained_ = True

    def predict(self, component_models_dir, k=-1, **kwargs):
        """
        Predicts GO terms for every protein in `proteins`

        Parameters
        ----------
        component_models_dir : str
            Path to the folder containing the output of the training models in TSV format

        Returns
        -------
        pandas DataFrame
            colums are 'protein', 'goterm', 'domain', 'score', as well as the scores for the componen model, depending
            on the cofigured mode

        """
        if not self.trained_:
            self.warning('The model is not trained, predictions are not possible')
            raise UntrainedComponentError
        self.tell('Bulding Dataset for LTR')
        models, ltr_dataset = self._build_dataset(component_models_dir)
        features = [i.split('.')[0] for i in models]
        X = ltr_dataset[features].values
        train_dmatrix = DMatrix(X)
        self.tell('Predicting using LTR')
        ltr_dataset['score'] = self.model_.predict(train_dmatrix)
        return ltr_dataset