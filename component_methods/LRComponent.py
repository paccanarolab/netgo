from component_methods import ComponentMethod
from Utils import ColourClass
from sklearn.linear_model import LogisticRegression
from scipy import sparse
import numpy as np
import pandas as pd
from joblib import dump, load


class LRComponent(ComponentMethod):

    def __init__(self):
        super(LRComponent, self).__init__()
        self.colour = ColourClass.bcolors.BOLD_CYAN
        self.model_ = None
        self.type_ = None
        self.trained_ = False

    def build_dataset_(self, feature_file, feature_index_file, fmt, function_assignment=None):
        """
        From arguments passed to `train` or `predict`, return a sklearn-compatible dataset

        Returns
        -------
        X : array
            features
        y : array
            labels (empty if `function_assigment` is set to None)
        features : array
            features used to build `X`
        index : list
            protein of the feature file
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
            self.tell(f'Loading feature index file {feature_index_file}')
            index = [
                line.strip() for line in open(feature_index_file)
            ]

        self.tell('Building dataset')
        X = []
        y = []
        for _, row in function_assignment.iterrows():
            y.append(row['goterm'])
            X.append(features[index[row['protein']]])
        X = np.array(X)
        y = np.array(y)
        return X, y, features, index

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
            * feature_index_file : str
                Path to a file that has a single protein id that matches the rows of
                `feature_file`, this is ignored when the `fmt` is `tab` or `pickle`
            * fmt : str, default 'npy'
                Format of the features file, the file loader will be picked accordingly.
                Possible values are 'npy', 'npz', 'tab', 'pickle'
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

        X, y, features, index = self.build_dataset_(feature_file,
                                                    feature_index_file,
                                                    fmt,
                                                    function_assignment=function_assignment)

        self.tell('training Logistic Regression')
        if lr_kwargs:
            self.model_ = LogisticRegression(**lr_kwargs)
        else:
            self.model_ = LogisticRegression(multi_class='ovr')
        self.model_.fit(X, y)
        self.trained_ = True

    def save_trained_model(self, output, **kwargs):
        """
        Saves the trained model onto disk.
        Parameters
        ----------
        output : str
            Filename to store the sk-learn logistic regression model

        Notes
        -----
        * An additional file with extension .info will be generated in order to load the feature type
        * sklearn models are saved/loaded using joblib
        """
        dump(self.model_, output)
        with open(f'{output}.info', 'w', newline='\n') as f:
            f.write(f'{self.type_}\n')

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