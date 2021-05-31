from component_methods import ComponentMethod
from Utils import ColourClass

class BLASTkNN(ComponentMethod):

    def __init__(self):
        super(BLASTkNN, self).__init__()
        self.colour = ColourClass.bcolors.BOLD_CYAN
        self.model_ = None
        self.trained_ = False

    def train(self, function_assignment, **kwargs):
        """
        Trains the BLAST kNN component method. This is:

        .. math:: S(G_i, P_j) = \frac{\sum_{p \in H_j} I(G_i, p) * B(P_j, p)}{\sum{p \in H_j, p}}

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
            * evalue_thr : float
                threshold for the BLAST evalue, only entries with evalue <= `evalue_thr` will
                be considered in the analysis
            * output_dir : str
                Path to the directory where the trained model will be saved
            * goterms : str or list, default 'all'
                The list of goterms that will be used to train a logistic regression,
                all terms in `function_assignment` will create a LR model by default.
        """

    def predict(self, proteins, k, **kwargs):
        """

        Parameters
        ----------
        proteins : set or list
            set of proteins to use for prediction
        k : int, default 30
            the number of GO terms to be returned per protein, the top-k
            will be returned
        kwargs
            Required information to predict with this model:
            * blast_file : str
                Path to the BLAST output file:
                BLAST(proteins, GOA)
            * go : GOTool.GeneOntology.GeneOntology instance
                An instance of the Gene Ontology class
        
        Returns
        -------
        pandas DataFrame
            predictions of GO terms for each prediction and GO subdomain, columns:
            'protein', 'goterm', 'domain', 'score'

        Notes
        -----
        * k predictions will be made per (protein, domain) pair
        * the `go` object specified in `kwargs` is assume to have been built
        """