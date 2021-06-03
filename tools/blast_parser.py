from Utils import FancyApp, Utilities, ColourClass
from rich.progress import track
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
        self.colour = ColourClass.bcolors.OKGREEN

    def get_homologs(self, query_proteins, subject_proteins, evalue_th,
                     query_acc=False, subject_acc=False):
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
        query_acc : boolean, default False
            if True, consider the uniprot accession instead of the full id
        subject_acc : boolean, default False
            if True, consider the uniprot accession instead of the full id
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
        total = Utilities.line_count(self.blast_output)
        for line in track(open(self.blast_output), total=total, description='Parsing...'):
            qaccver, saccver, _, _, _, _, _, _, _, _, evalue, bitscore = line.strip().split()
            if query_acc:
                qaccver = qaccver.split('|')[1]
            if subject_acc:
                saccver = saccver.split('|')[1]
            evalue = float(evalue)
            bitscore = float(bitscore)
            if evalue <= evalue_th:
                if qaccver in query and saccver in subject:
                    B[query.index(qaccver), subject.index(saccver)] = bitscore
                # this could be superfluous, but I'm not 100% sure, so just in case...
                if qaccver in subject and saccver in query:
                    B[query.index(saccver), subject.index(qaccver)] = bitscore
        return B
