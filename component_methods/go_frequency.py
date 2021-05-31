import pandas as pd

from component_methods import ComponentMethod, UntrainedComponentError
from Utils import ColourClass
from GOTool.GeneOntology import GeneOntology
from rich.progress import track

class GOFrequency(ComponentMethod):

    def __init__(self):
        super(GOFrequency, self).__init__()
        self.colour = ColourClass.bcolors.BOLD_CYAN
        self.go_freqs = None
        self.trained_ = False

    def train(self, function_assignment, **kwargs):
        """
        Trains the underlying model based on the information available in
        `function_assignment`

        Parameters
        ----------
        function_assignment : pandas DataFrame
            must contain columns 'protein' and 'goterm', with an optional 'score' column
        kwargs
            Required information to train this model:
            * obo : str
                The path to a go.obo file that will be used to divide the GO terms into
                specific sub-domains

        Notes
        -----
        Priors are learnt separately per GO subdomain
        """
        obo = kwargs.get('obo')

        self.tell(f'Loading obo file {obo}')
        go = GeneOntology(obo)
        go.build_structure()
        data = {key: [] for key in ['goterm', 'domain', 'protein count']}

        self.tell('Extracting GO term frequencies')
        counts = function_assignment.groupby('goterm').count()
        for goterm, prot in counts.iterrows():
            data['goterm'].append(goterm)
            data['domain'].append(go.find_term(goterm).domain)
            data['protein count'].append(prot['protein'])
        self.go_freqs = pd.DataFrame(data)
        del data

        self.tell('Calculating GO term priors by domain')
        self.go_freqs['domain sum'] = self.go_freqs['protein count'].groupby(
            self.go_freqs['domain']).transform('sum')
        self.go_freqs['prior'] = self.go_freqs['protein count'] / self.go_freqs['domain sum']

        self.tell('Calculating GO term priors overall')
        all_sum = self.go_freqs['protein count'].sum()
        self.go_freqs['prior_overall'] = self.go_freqs['protein count'] / all_sum
        self.go_freqs.drop(columns='domain sum', inplace=True)
        self.trained_ = True

    def save_trained_model(self, output, **kwargs):
        """
        Saves the trained model onto disk.
        Parameters
        ----------
        output : str
            Filename to store the frequency information
        kwargs
            * fmt : str, default 'tsv'
                you can set this parameter to store the model in pickle format, which
                is faster to load later on. Possible values are 'tsv' and 'pickle'
        """
        fmt = kwargs.get('fmt', 'tsv')
        if fmt == 'tsv':
            self.go_freqs.to_csv(output, sep='\t', index=False)
        elif fmt == 'pickle':
            self.go_freqs.to_pickle(output)

    def load_trained_model(self, model_filename, **kwargs):
        """
        Recovers the state of the component model after training, in order to
        make predictions without re-training the component

        Parameters
        ----------
        model_filename : str
            Filename to store the frequency information
        kwargs
            * fmt : str, default 'tsv'
                you can set this parameter to store the model in pickle format, which
                is faster to load later on. Possible values are 'tsv' and 'pickle'
        """
        fmt = kwargs.get('fmt', 'tsv')
        if fmt == 'tsv':
            self.go_freqs = pd.read_csv(model_filename, sep='\t')
        elif fmt == 'pickle':
            self.go_freqs = pd.read_pickle(model_filename)
        self.trained_ = True

    def predict(self, proteins, k=30, **kwargs):
        """
        Parameters
        ----------
        proteins : set or list
            set of proteins to which
        k : int, default 30
            the number of GO terms to be returned per protein, the top-k
            will be returned
        kwargs
            * mode: str, default 'domain'
                you can set this  parameter to determine whether the prior is calculated
                across domains or overall, possible values are 'domain' and 'overall'

        Returns
        -------
        pandas DataFrame
            predictions of GO terms for each prediction and GO subdomain, columns:
            'protein', 'goterm', 'domain', 'score'

        Notes
        -----
            k predictions will be made per (protein, domain) pair

        """
        if not self.trained_:
            self.warning('The model is not trained, predictions are not possible')
            raise UntrainedComponentError
        mode = kwargs.get('mode', 'domain')
        mode = 'prior' if mode == 'domain' else 'prior_overall'
        if isinstance(proteins, list):
            proteins = set(proteins)
        prediction = {key: [] for key in ['protein', 'goterm', 'domain', 'score']}
        self.tell('predicting')
        # the prediction of this one is unique, so we need to calculate it once
        pred = self.go_freqs.sort_values(mode, ascending=False).groupby('domain').head(k)

        # @TODO: for sure there is a faster, better way to do this, but it's 3:00 am
        for protein in track(proteins):
            for _, p in pred.iterrows():
                prediction['protein'].append(protein)
                prediction['goterm'].append(p['goterm'])
                prediction['domain'].append(p['domain'])
                prediction['score'].append(p[mode])

        return pd.DataFrame(prediction)