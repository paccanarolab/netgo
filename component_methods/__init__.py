import abc

from Utils import FancyApp

class ComponentMethod(FancyApp.FancyApp):

    @abc.abstractmethod
    def predict(self, proteins, k=30, **kwargs):
        """
        Predicts GO terms for every protein in `proteins`

        Parameters
        ----------
        proteins : set or list
            set of proteins to which
        k : int, default 30
            the number of GO terms to be returned per protein, the top-k
            will be returned
        kwargs
            will be handled by the implementation accordingly

        Returns
        -------
        pandas DataFrame
            predictions of GO terms for each prediction and GO subdomain, columns:
            'protein', 'goterm', 'domain', 'score'

        Notes
        -----
            k predictions will be made per (protein, domain) pair
        """

    @abc.abstractmethod
    def train(self, function_assignment, **kwargs):
        """
        Trains the underlying model based on the information available in
        `function_assignment`

        Parameters
        ----------
        function_assignment : pandas DataFrame
            must contain columns 'protein' and 'goterm', with an optional 'score' column
        kwargs
            will be handled by the implementation accordingly

        Notes
        -----
        The train method will be called once from the predict command, but for other
        uses the underlying implementation might indicate to run it more than once.
        """