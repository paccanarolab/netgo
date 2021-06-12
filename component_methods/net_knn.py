from component_methods import ComponentMethod, UntrainedComponentError
from Utils import ColourClass, Utilities
from tools.blast_parser import BLASTParser
from rich.progress import track
from scipy import sparse
import numpy as np
import pandas as pd


import os


class NETkNN(ComponentMethod):

    def __init__(self):
        super(NETkNN, self).__init__()
        self.colour = ColourClass.bcolors.BOLD_CYAN
        self.top_homologs_ = None
        self.neighborhood_ = None
        self.net_knn_ = None
        self.net_knn_index_ = None
        self.goterms_index_ = None
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
            * proteins : list of str
                The list of proteins to use for matrix extraction.
        """
        evalue_thr = kwargs.get('evalue_thr', 0.001)
        blast_to_string = kwargs['blast_to_string']
        links = kwargs['string_links']
        proteins = kwargs['proteins']

        self.tell('Finding homologs in STRING')
        names = "qaccver saccver pident length mismatch gapopen qstart qend sstart send evalue bitscore".split()
        blast_results = pd.read_csv(blast_to_string, sep='\t', names=names)

        self.tell('Filtering evalue')
        blast_results = blast_results[blast_results['evalue'] <= evalue_thr]

        self.tell('Getting top homologs to STRING')
        blast_results = (blast_results[['qaccver', 'saccver', 'bitscore']]
                         .sort_values('bitscore', ascending=False).groupby('qaccver').head(1))

        self.tell('Filling in not-found homologs')
        self.top_homologs_ = pd.DataFrame(proteins, columns=['qaccver'])
        self.top_homologs_ = self.top_homologs_.merge(blast_results,
                                                      left_on='qaccver', right_on='qaccver', how='left')
        self.top_homologs_['saccver'].fillna('NOHO', inplace=True)
        self.top_homologs_['bitscore'].fillna(0, inplace=True)
        interactors = self.top_homologs_['saccver'].unique()

        self.tell('Loading graph')
        self.neighborhood_ = pd.read_csv(links, sep='\t',
                                         names=['p1', 'p2', 'weight'],
                                         dtype={'p1': str, 'p2': str, 'weight': np.float64})

        self.tell('Filtering graph, keeping all links involving top homologs')
        self.neighborhood_ = self.neighborhood_[
            self.neighborhood_['p1'].isin(interactors) | self.neighborhood_['p2'].isin(interactors)]
        self.warning(f'Final graph contains {self.neighborhood_.shape[0]} interactions')
        self.neighborhood_ = pd.DataFrame(self.neighborhood_)

        self.tell('Building common protein index')
        all_proteins = set(function_assignment['protein'].unique()) | set(self.top_homologs_['qaccver'].unique()) | \
                       set(self.top_homologs_['saccver'].unique()) | set(self.neighborhood_['p1'].unique()) | \
                       set(self.neighborhood_['p2'].unique())
        all_proteins_index = pd.DataFrame(enumerate(sorted(all_proteins)), columns=['index', 'protein'])

        self.tell('Building GO-term index')
        self.goterms_index_ = pd.DataFrame(enumerate(sorted(function_assignment['goterm'].unique())),
                                     columns=['index', 'goterm'])

        self.tell('Preparing interaction data')
        neighbors_matrix = (self.neighborhood_.merge(all_proteins_index, left_on='p1', right_on='protein')
            .drop(columns='protein')
            .rename(columns={'index': 'index_p1'})
            .merge(all_proteins_index, left_on='p2', right_on='protein')
            .drop(columns='protein')
            .rename(columns={'index': 'index_p2'})[['index_p1', 'index_p2', 'weight']])
        functions_matrix = (function_assignment.merge(all_proteins_index, left_on='protein', right_on='protein')
            .drop(columns='protein')
            .rename(columns={'index': 'index_p'})
            .merge(self.goterms_index_, left_on='goterm', right_on='goterm')
            .drop(columns='goterm')
            .rename(columns={'index': 'index_go'}))
        functions_matrix['link'] = 1.

        self.tell('Building interaction matrices')
        N_prots = all_proteins_index.shape[0]
        N_goterms = self.goterms_index_.shape[0]
        W = sparse.coo_matrix((neighbors_matrix['weight'].astype('float').values,
                               (neighbors_matrix['index_p1'].values, neighbors_matrix['index_p2'].values)),
                              shape=(N_prots, N_prots))
        I = sparse.coo_matrix((functions_matrix['link'].values,
                               (functions_matrix['index_p'].values, functions_matrix['index_go'].values)),
                              shape=(N_prots, N_goterms))

        self.tell('Calculating Net-kNN scores')
        denominator = W.sum(axis=1).A1
        idx_non_zeros = np.where(denominator != 0)[0]
        numerator = W @ I
        self.net_knn_ = numerator[idx_non_zeros, :] / denominator[idx_non_zeros, np.newaxis]
        self.net_knn_ = np.array(self.net_knn_)
        self.net_knn_index_ = all_proteins_index[
            all_proteins_index['index'].isin(idx_non_zeros)].rename(columns={'index':'original_index'})
        self.net_knn_index_['index'] = range(self.net_knn_index_.shape[0])
        self.trained_ = True

    def save_trained_model(self, output, **kwargs):
        self.tell('Saving top homologs')
        self.top_homologs_.to_csv(os.path.join(output, 'Net-kNN.homologs.tsv'), sep='\t', index=False)
        self.tell('Saving neighborhood')
        self.neighborhood_.to_pickle(os.path.join(output, 'Net-kNN.neighborhood.pkl'))
        self.tell('Saving Net-kNN scores, and indices')
        np.save(os.path.join(output, 'Net-kNN.scores.npy'), self.net_knn_)
        self.net_knn_index_.to_csv(os.path.join(output, 'Net-kNN.scores-protein_index.tsv'), sep='\t', index=False)
        self.goterms_index_.to_csv(os.path.join(output, 'Net-kNN.scores-goterms_index.tsv'), sep='\t', index=False)

    def load_trained_model(self, model_filename, **kwargs):
        self.tell('Loading top homologs')
        self.top_homologs_ = pd.read_csv(os.path.join(model_filename, 'Net-kNN.homologs.tsv'), sep='\t')
        self.tell('Loading neighborhood')
        self.neighborhood_ = pd.read_pickle(os.path.join(model_filename, 'Net-kNN.neighborhood.pkl'))
        self.tell('Loading Net-kNN scores, and indices')
        self.net_knn_ = np.load(os.path.join(model_filename, 'Net-kNN.scores.npy'))
        self.net_knn_index_ = pd.read_csv(os.path.join(model_filename, 'Net-kNN.scores-protein_index.tsv'), sep='\t')
        self.goterms_index_ = pd.read_csv(os.path.join(model_filename, 'Net-kNN.scores-goterms_index.tsv'), sep='\t')
        self.trained_ = True

    def predict(self, proteins, k=30, **kwargs):
        if not self.trained_:
            self.warning('The model is not trained, cannot save the model' )
            raise UntrainedComponentError
        go = kwargs['go']
        terms = self.goterms_index_['goterm'].values
        domains = []
        for term in terms:
            domains.append(go.find_term(term).domain)
        domains = np.array(domains)
        homologs = self.top_homologs_[self.top_homologs_['qaccver'].isin(proteins)]['saccver'].values
        condition = self.net_knn_index_['protein'].isin(homologs)
        extract_indices = self.net_knn_index_[condition]['index'].values
        extract_homologs = self.net_knn_index_[condition]['protein'].values
        index_to_protein = {}
        for _, r in self.top_homologs_.merge(self.net_knn_index_,
                                             left_on='saccver', right_on='protein').iterrows():
            index_to_protein[r['index']] = r['qaccver']
        pred = self.net_knn_[extract_indices, :]
        prediction = {key: [] for key in ['protein', 'goterm', 'domain', 'score']}
        for d in np.unique(domains):
            self.tell(f'Extracting {k} predictions for {d}')
            domain_index = np.where(domains == d)[0]
            domain_terms = terms[domain_index]
            domain_pred = pred[:, domain_index]
            top_k = np.argsort(domain_pred)[:, -1:-k-1:-1]
            for p_idx, h_idx in track(enumerate(extract_indices),
                                      total=len(extract_indices),
                                      description='Extracting...'):
                for i in range(k):
                    prediction['protein'].append(index_to_protein[h_idx])
                    prediction['goterm'].append(domain_terms[top_k[p_idx, i]])
                    prediction['score'].append(domain_pred[p_idx, top_k[p_idx, i]])
                    prediction['domain'].append(d)
        return pd.DataFrame(prediction)