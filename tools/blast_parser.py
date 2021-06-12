from Utils import FancyApp, Utilities, ColourClass
from rich.progress import track
import numpy as np
import pandas as pd
from scipy import sparse

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

    def get_top_homologs(self, query_proteins, evalue_th):
        """
        This is the same as `get_homologs`, but the condition is that one of the interactors needs to be
        included in `query_proteins`. In order to build the matrix it needs to load all the blast results into
        memory, therefore it can be slower.

        Parameters
        ----------
        query_proteins : list
            proteins that will be considered to output the matrix
        evalue_th : float
            proteins with BLAST e-value <= `evalue_th` are considered homologs

        Returns
        -------
        B : pandas Dataframe with n_query rows
            columns are 'protein', 'homolog', 'bitscore'
        subjects_index : list of str
            the list of proteins on the columns of `B`

        Notes
        -----
        This method does not make any assumption about the format of the protein ids you must
        take care of such cleaning before calling this method.

        This method assumes that query proteins are included in the first column of the tabulas format,
        that is, that all query proteins are in "qaccver". "saccver" is NOT filtered, and after filtering
        "qaccver", the remaining entries in "saccver" will be sorted to create `subjects_index`.
        """
        names = "qaccver saccver pident length mismatch gapopen qstart qend sstart send evalue bitscore".split()
        self.tell(f'Loading BLAST results from {self.blast_output}')
        blast_entries = pd.read_csv(self.blast_output, sep='\t', names=names)
        self.tell(f'Filtering BLAST results according to e-value (<= {evalue_th})')
        condition = blast_entries['evalue'] <= evalue_th
        blast_entries = blast_entries[condition]
        self.tell('Filtering BLAST query column')
        condition = blast_entries['qaccver'].isin(query_proteins)
        blast_entries = blast_entries[condition]
        self.tell('Building subjects index')
        subjects_index = sorted(blast_entries['saccver'].unique())
        self.tell('Building Bitscore matrix')
        B = sparse.csr_matrix((len(query_proteins), len(subjects_index)))
        total = blast_entries.shape[0]
        for _, r in track(blast_entries.iterrows(), total=total, description='Filling matrix...'):
            B[query_proteins.index(r['qaccver']), subjects_index.index(r['saccver'])] = r['bitscore']
        return B, subjects_index
