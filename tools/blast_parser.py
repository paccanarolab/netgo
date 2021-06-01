from Utils import FancyApp, ColourClass
import numpy as np


class BLASTParser(FancyApp.FancyApp):

    def __init__(self, blast_output):
        """
        BLAST tabular parser
        Parameters
        ----------
        blast_output : str
            Path to the BLAST output file, must have been computed with "outfmt 7"
        """
        super(BLASTParser, self).__init__()
        self.blast_output = blast_output

    def get_homologs(self, query_proteins, subject_proteins, evalue_th):
        """
        Calculates a homology matrix with the configuration

        Parameters
        ----------
        query_proteins : list
            proteins that will be considered to output the matrix
        subject_proteins : list
            proteins that will be considered to output the matrix
        evalue_th : float
            proteins with BLAST e-value <= `evalue_th` are considered homologs

        Returns
        -------
        B : numpy array (n_query, n_subject)
            the bitscores of the associations between `query_proteins` and `subject_proteins`

        Notes
        -----
        the indices of will be the sorted versions of `query_proteins` and `subject_proteins` for rows
        and columns respectively
        """
        query = sorted(query_proteins)
        subject = sorted(subject_proteins)
        B = np.zeros((len(query), len(subject)))
        self.tell('Parsing Blast File')
        for line in open(self.blast_output):
            qaccver, saccver, _, _, _, _, _, _, _, _, evalue, bitscore = line.strip().split()
            evalue = float(evalue)
            bitscore = float(bitscore)
            if evalue <= evalue_th:
                if qaccver in query and saccver in subject:
                    B[query.index(qaccver), subject.index(saccver)] = bitscore
                # this could be superfluous, but I'm not 100% sure, so just in case...
                if qaccver in subject and saccver in query:
                    B[subject.index(qaccver), query.index(saccver)] = bitscore
        return B
