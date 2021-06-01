from component_methods import ComponentMethod, UntrainedComponentError
from Utils import ColourClass
import numpy as np

import os


class NETkNN(ComponentMethod):

    def __init__(self):
        super(NETkNN, self).__init__()
        self.model_ = None
        self.trained_ = False

    def train(self, function_assignment, **kwargs):
        """

        Parameters
        ----------
        function_assignment : pandas DataFrame
            must contain columns 'protein' and 'goterm', with an optional 'score' column
        kwargs

        Returns
        -------

        """