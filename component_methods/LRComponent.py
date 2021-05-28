from component_methods import ComponentMethod
from Utils import ColourClass
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler
from rich.progress import track
from scipy import sparse
from joblib import dump, load, Parallel, delayed
import numpy as np
import pandas as pd

import os


class LRComponent(ComponentMethod):

    def __init__(self):
        super(LRComponent, self).__init__()
        self.colour = ColourClass.bcolors.BOLD_CYAN
        self.model_ = None
        self.type_ = None
        self.trained_ = False

    def build_dataset_(self, feature_file, protein_index_file, fmt,
                       function_assignment=None,
                       goterms='all'):
        """
        From arguments passed to `train` or `predict`, return a sklearn-compatible dataset

        Returns
        -------
        X : array (n_proteins, n_features)
            features
        y : array (n_proteins, n_terms)
            labels (empty if `function_assigment` is set to None)
            the encoding is Binary, and each column is suitable for training a Logistic Regression
        features : array
            features used to build `X`
        protein_index : list
            protein index, this is used for both the training data as well as the labels matrix
        terms_index : list
            goterm index (compatible with the labels matrix)

        """
        features, index = None, None
        self.tell(f'Loading features from {feature_file}')
        if fmt == 'npy':
            features = np.load(feature_file)
        elif fmt == 'npz':
            features = sparse.load_npz(feature_file).toarray()
        elif fmt in ['tab', 'pickle']:
            if fmt == 'pickle':
                df = pd.read_pickle(feature_file)
            else:  # probably the worst format in this case, but it could be a useful choice
                df = pd.read_csv(feature_file, tab='\t')
            index = list(df['protein'])
            feature_columns = [c for c in df.columns if c != 'protein']
            features = df[feature_columns].values

        if fmt not in ['tab', 'pickle']:
            self.tell(f'Loading protein index file {protein_index_file}')
            index = [line.strip() for line in open(protein_index_file)]

        self.tell('Building dataset')

        pivot = function_assignment.pivot_table(index='protein', columns='goterm', aggfunc=lambda x: 1, fill_value=0)
        if goterms != 'all':
            pivot = pivot[goterms]

        term_index = pivot.columns.to_list()
        protein_index = pivot.index.to_list()
        X = features[np.where(np.isin(index, protein_index))[0]]
        y = pivot.values

        return X, y, features, protein_index, term_index

    def train_single_term(self, X, y, term_index, term, output_dir):
        """
        Utility function that trains a linear regression for a single term

        Parameters
        ----------
        X : array (n_proteins, n_features)
            features
        y : array (n_proteins, n_terms)
            labels
        term_index : list
            indices of the columns of `y`
        term : str
            GO term to train
        output_dir : str
            output_directory where the model will be saved
        """
        model = make_pipeline(
            MaxAbsScaler(),
            SGDClassifier(loss='log', n_jobs=35)
        )
        model.fit(X, y[:, term_index.index(term)])
        self.save_trained_model(output_dir,
                                goterm=term,
                                model=model)

    def train(self, function_assignment, **kwargs):
        """
        Trains a Logistic regression model based on one of three types of features
        * InterPro
        * k-mer
        * ProFET

        Parameters
        ----------
        function_assignment : pandas DataFrame
            must contain columns 'protein' and 'goterm', with an optional 'score' column
        kwargs
            Required information to train this model:
            * type : str
                The type of feature being used for the logistic regression. Possible
                options are: 'interpro', 'kmer', and 'profet'
            * lr_kwargs : dict
                Arguments to be passed to the scikit-learn LogisticRegression instance.
            * feature_file : str
                Path to a file where to read the features from. How the file will be
                read will depend on the `fmt` argument
            * protein_index_file : str
                Path to a file that has a single protein id that matches the rows of
                `feature_file`, this is ignored when the `fmt` is `tab` or `pickle`
            * fmt : str, default 'npy'
                Format of the features file, the file loader will be picked accordingly.
                Possible values are 'npy', 'npz', 'tab', 'pickle'
            * output_dir : str
                Path to the directory where the models will be saved
            * goterms : str or list, default 'all'
                The list of goterms that will be used to train a logistic regression,
                all terms in `function_assignment` will create a LR model by default.

        Notes
        -----
        Priors are learnt separately per GO subdomain
        """
        self.type_ = kwargs['type']
        self.handle_ = f'LR-{self.type_}'
        lr_kwargs = kwargs.get('lr_kwargs', None)
        feature_file = kwargs['feature_file']
        feature_index_file = kwargs.get('feature_index_file', None)
        fmt = kwargs.get('fmt', 'npy')
        output_dir = kwargs['output_dir']
        goterms = kwargs['goterms']

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            with open(os.path.join(output_dir, f'{output_dir}.info'), 'w', newline='\n') as f:
                f.write(f'{self.type_}\n')

        X, y, features, protein_index, term_index = self.build_dataset_(
            feature_file, feature_index_file,
            fmt, function_assignment=function_assignment,
            goterms=goterms
        )

        self.tell('training Logistic Regression')

        Parallel(n_jobs=4, max_nbytes=1e6)(
            delayed(self.train_single_term)(X, y, term_index, term, output_dir) for term in term_index
        )
        # for term in track(term_index, description='Training models'):
        #     self.train_single_term(X, y, term_index, term, output_dir)


        self.trained_ = True

    def save_trained_model(self, output, **kwargs):
        """
        Saves the trained model onto disk.
        Parameters
        ----------
        output : str
            Filename to store the sk-learn logistic regression model
        kwargs
            Required information to train this model:
            * goterm : str
                The GO term that corresponds to this model
            * model : sklearn model
                The model to be saved
        Notes
        -----
        * An additional file with extension .info will be generated in order to load the feature type
        * sklearn models are saved/loaded using joblib
        """
        goterm = kwargs['goterm'].replace(':' ,'_')
        model = kwargs['model']
        dump(model, os.path.join(output, f'{goterm}.lr_model'))

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

        Notes
        -----
        This is designed to load models saved using the `save_trained_model` function of the same
        component, which expects the additional file with .info extension.
        """
        with open(model_filename) as f:
            self.type_ = f.read().strip()
        self.model_ = load(model_filename)
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
            Required information to predict with this model:
            * feature_file : str
                Path to a file where to read the features from. How the file will be
                read will depend on the `fmt` argument
            * feature_index_file : str
                Path to a file that has a single protein id that matches the rows of
                `feature_file`, this is ignored when the `fmt` is `tab` or `pickle`
            * fmt : str, default 'npy'
                Format of the features file, the file loader will be picked accordingly.
                Possible values are 'npy', 'npz', 'tab', 'pickle'
            * go : GOTool.GeneOntology.GeneOntology instance
                An instance to a gene ontology object
        Returns
        -------
        pandas DataFrame
            predictions of GO terms for each prediction and GO subdomain, columns:
            'protein', 'goterm', 'domain', 'score'

        Notes
        -----
        * k predictions will be made per (protein, domain) pair
        * files specified in `kwargs` are expected to match those used for training.
        * the `go` object specified in `kwargs` is assumed to have built
        """
        go = kwargs['go']
        feature_file = kwargs['feature_file']
        feature_index_file = kwargs.get('feature_index_file', None)
        fmt = kwargs.get('fmt', 'npy')

        X, _, features, index = self.build_dataset_(feature_file,
                                                    feature_index_file,
                                                    fmt)

        y_pred = self.model_.predict_proba(X)