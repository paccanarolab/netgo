import pandas as pd

from component_methods import ComponentMethod
from Utils import ColourClass
from GOTool.GeneOntology import GeneOntology


class GOFrequency(ComponentMethod):

    def __init__(self):
        super(GOFrequency, self).__init__()
        self.colour = ColourClass.bcolors.BOLD_CYAN
        self.go_freqs = None
        self.go = None

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
        for goterm, prot in counts.iterrow():
            data['goterm'].append(goterm)
            data['domain'].append(go.find_term(goterm).domain)
            data['protein count'].append(prot['protein'])
        self.go_freqs = pd.DataFrame(data)
        del data

        self.tell('Calculating GO term priors by domain')
        sums = self.go_freqs.groupby(['domain', 'goterm'])['protein count'].sum()
        sums = sums.rename(columns={'protein count': 'domain_sum'})
        self.go_freqs.merge(sums, left_on='domain', right_on='domain')
        self.go_freqs['prior'] = self.go_freqs['protein count'] / self.go_freqs['domain_sum']

        self.tell('Calculating GO term priors overall')
        all_sum = self.go_freqs['protein count'].sum()
        self.go_freqs['prior_overall'] = self.go_freqs['protein count'] / all_sum
        self.go_freqs.drop(columns='domain_sum', inplace=True)

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
        mode = kwargs.get('mode', 'domain')
        mode = 'prior' if mode == 'domain' else 'prior_overall'
        if isinstance(proteins, list):
            proteins = set(proteins)
        prediction = {key: [] for key in ['protein', 'goterm', 'domain', 'score']}
        self.tell('predicting')
        # the prediction of this one is unique, so we need to calculate it once
        pred = self.go_freqs.sort_values(mode, ascending=False).groupby('domain').head(k)

        # @TODO: for sure there is a faster, better way to do this, but it's 3:00 am
        for protein in proteins:
            for _, p in pred.iterrows():
                prediction['protein'].append(protein)
                prediction['goterm'].append(pred['goterm'])
                prediction['domain'].append(pred['domain'])
                prediction['score'].append(pred[mode])

        return pd.DataFrame(prediction)