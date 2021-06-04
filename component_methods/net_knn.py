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
            Required information to train this model:
            * string_links : str
                Path to a file detailing the links in STRING, mapped to UniProt accession numbers
            * blast_to_string : str
                Path to a file containing the BLAST output between proteins in `function_assignment` and proteins in
                STRING. The assumption is that p1 is contained in `function_assignment` and p2 is a string ID mapped
                to UniProt
            * evalue_thr : float, default 0.001
                threshold to use for the BLAST output, only entries with e-value <= `evalue_thr` will be considered
        """
