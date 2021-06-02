from Utils import FancyApp
from xgboost import DMatrix
from component_methods import UntrainedComponentError
import numpy as np
import xgboost as xgb


class LTR(FancyApp.FancyApp):

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
        super(LTR).__init__()
        self.mode_ = mode
        self.trained_ = False

    def train(self, function_assignment, output):
        pass

    def predict(self, proteins, output):
        pass