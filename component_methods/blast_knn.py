from component_methods import ComponentMethod, UntrainedComponentError
from Utils import ColourClass
from tools.blast_parser import BLASTParser
from rich.progress import track
import numpy as np
import pandas as pd

import os

class BLASTkNN(ComponentMethod):

    def __init__(self):
        super(BLASTkNN, self).__init__()
        self.colour = ColourClass.bcolors.BOLD_CYAN
        self.B_ = None
        self.I_ = None
        self.trained_ = False

    def train(self, function_assignment, **kwargs):
        """

        This method in particular will create "conveniently loadable" matrices
        to perform the calculations, which mainly involves parsing the BLAST
        output and building the :math:`I` and :math:`B` matrices above:

        `H` does not need to be computed explicitly, as it is simply a binary version of `B`

        Dimensionalities:

        * :math:`I` is (n_annotated_proteins, n_goterms)
            a binary version of `function_assignment`
        * :math:`B` is (n_proteins, n_annotated_proteins)
            :math:`B_{i, j}` contains the BLAST bit scores for proteins :math:`i` and :math:`j`

        Note that this model can only be reused to predict for a subset of proteins included in the training.
        If you need to make predictions for new proteins, you need to compute the BLAST yourself and call
        this method again.

        Parameters
        ----------
        function_assignment : pandas DataFrame
            must contain columns 'protein' and 'goterm', with an optional 'score' column
        kwargs
            Required information to train this model

            * blast_file : str
                Path to the BLAST output file:
                BLAST(training proteins, GOA)
                "training proteins" is in this case those contained in `function_assignment`
            * evalue_thr : float, default 0.001
                threshold for the BLAST evalue, only entries with evalue <= `evalue_thr` will
                be considered in the analysis
            * proteins : list of str
                The list of proteins to use for matrix extraction.
        """
        evalue_thr = kwargs.get('evalue_thr', 0.001)
        blast_file = kwargs['blast_file']
        self.tell('Building GOA indicator matrix')
        self.I_ = function_assignment.pivot_table(index='protein', columns='goterm', aggfunc=lambda x: 1, fill_value=0)
        training_proteins = sorted(self.I_.index.tolist())
        proteins = sorted(kwargs['proteins'])
        blast_parser = BLASTParser(blast_file)
        self.B_ = blast_parser.get_homologs(proteins, training_proteins, evalue_thr,
                                            query_acc=True,subject_acc=True)
        self.trained_ = True

    def predict(self, proteins, k=30, **kwargs):
        """
        Predicts the BLAST kNN component method. This is:

        .. math:: S(G_i, P_j) = \\frac{\\sum_{p \\in H_j} I(G_i, p) * B(P_j, p)}{\\sum{p \\in H_j, p}}

        Parameters
        ----------
        proteins : set or list
            set of proteins to use for prediction
        k : int, default 30
            the number of GO terms to be returned per protein, the top-k
            will be returned
        kwargs
            Required information to predict with this model:
            * go : GOTool.GeneOntology.GeneOntology instance
                An instance of the Gene Ontology class
            * output_complete_prediction : str, default None
                Path to a file where the full prediction (not only the top k) will be saved.
                This can be used as a cache as well, to reuse the calculation. By default this
                is not stored, and only the top-k predictions are returned.
        
        Returns
        -------
        pandas DataFrame
            predictions of GO terms for each prediction and GO subdomain, columns:
            'protein', 'goterm', 'domain', 'score'

        Notes
        -----
        * k predictions will be made per (protein, domain) pair
        * the `go` object specified in `kwargs` is assume to have been built
        * even if `output_complete_prediction` is not set, ALL the predictions will be calculated,
        and then discarded.
        """
        if not self.trained_:
            self.warning('The model is not trained, predictions are not possible')
            raise UntrainedComponentError
        self.tell('Calculating BLAST-kNN prediction')
        go = kwargs['go']
        output_complete_prediction = kwargs.get('output_complete_prediction', None)

        if output_complete_prediction is None or \
            (output_complete_prediction and not os.path.exists(output_complete_prediction)):
            s = self.B_.sum(axis=1)
            ind_non_zeros = np.where(s != 0)
            P = self.B_ @ self.I_.values
            P[ind_non_zeros] = P[ind_non_zeros] / s[ind_non_zeros][:, np.newaxis]

            prediction = {key: [] for key in ['protein', 'goterm', 'domain', 'score']}

            # @TODO: we are calculating the entire thing, and then spitting out the top-k.
            # we could find the top-k from the P matrix instead, but this is useful for now
            # for caching purposes.
            tot = len(proteins)
            for p_idx, protein in track(enumerate(sorted(proteins)),
                                        total=tot, description="Predicting..."):
                for g_idx, goterm in enumerate(self.I_.columns):
                    prediction['protein'].append(protein)
                    prediction['goterm'].append(goterm)
                    prediction['domain'].append(go.find_term(goterm).domain)
                    prediction['score'].append(P[p_idx, g_idx])
            prediction = pd.DataFrame(prediction)
            if output_complete_prediction:
                prediction.to_pickle(output_complete_prediction)
        else:
            self.tell(f'Found a previously calculated BLAST-kNN file, loading {output_complete_prediction}')
            prediction = pd.read_pickle(output_complete_prediction)
        return prediction.sort_values('score', ascending=False).groupby(['protein', 'domain']).head(k)

    def save_trained_model(self, output, **kwargs):
        """
        Saves the trained model onto disk.
        Parameters
        ----------
        output : str
            Where to store the model, it could be a file or a directory
            depending on the model.

        Notes
        -----

        Only matrix :math:`B` will be saved, together with its indices. The `predict` command *requires* that the
        same `function_assignment` pandas used for training is passed. This save procedure is mainly to avoid
        re-parsing the BLAST file.
        """
        if not self.trained_:
            self.warning('The model is not trained, cannot save the model' )
            raise UntrainedComponentError
        self.tell('Saving BLAST-kNN matrix')
        np.save(output, self.B_)

    def load_trained_model(self, model_filename, **kwargs):
        """
        Recovers the state of the component model after training, in order to
        make predictions without re-training the component

        Parameters
        ----------
        model_filename : str
            Filename to store the frequency information
        kwargs
            No keyword arguments are used in this component
            * function_assignment : pandas DataFrame
                must contain columns 'protein' and 'goterm', with an optional 'score' column.
                Used to recreate matrix :math:`I`, therefore the functional_assignment must
                be compatible with it (the set of proteins must be the same)

        Notes
        -----
        This is designed to load models saved using the `save_trained_model` function of the same
        component, which expects the additional file with .info extension.
        """
        function_assignment = kwargs['function_assignment']
        self.tell('Recreating Matrix I')
        self.I_ = function_assignment.pivot_table(index='protein', columns='goterm', aggfunc=lambda x: 1, fill_value=0)
        self.tell('Loading Matrix B')
        self.B_ = np.load(model_filename)
        self.trained_ = True