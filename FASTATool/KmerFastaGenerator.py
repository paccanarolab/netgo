"""
k-mer Fasta Encoder

This implements a generator that will output a feature vector of k-mers per protein
"""

__author__ = 'Mateo Torres'
__email__ = 'torresmateo@gmail.com'
__copyright__ = 'Copyright (c) 2021, Mateo Torres'
__license__ = 'MIT'
__version__ = '1.0'

import numpy as np
from scipy import sparse
from FASTATool import FastaParser
from Utils import FancyApp
from itertools import product

import os

class KMerFasta(FancyApp.FancyApp):

    AMINOACIDS = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
        'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y', 'Z'
    ]

    def __init__(self, fasta, k=3, dense=True):
        """
        Parameters
        ----------
        fasta : str
            path to the fasta file to parse
        k : int, default 3
            length of the subsequences (k-mers)
        dense : bool, default True
            whether to return dense arrays, if False, a sparse matrix will be returned instead
        """
        super(KMerFasta, self).__init__()
        self.fasta = os.path.realpath(fasta)
        self.k = k
        self.dense = dense
        self.kmer_idx = list(''.join(i) for i in product(self.AMINOACIDS, repeat=self.k))

        self.tell('parsing fasta file')
        self.fp = FastaParser.FastaFile(self.fasta)
        self.fp.buildBrowsableDict()

        self.tell('building protein index')
        self.protein_idx = sorted(self.fp.information['proteins'].keys())

        self.tell('Initialising k-mer matrix')
        self.shape = len(self.protein_idx), len(self.kmer_idx)
        self.matrix = None

    def encode(self):
        self.tell(f'Encoding fasta into kMer format')
        encoding = {}
        for p_i, protein in enumerate(self.protein_idx):
            sequence = self.fp.information['proteins'][protein]
            for i in range(len(sequence) - self.k):
                kmer = sequence[i: i+self.k]
                kmer_idx = self.kmer_idx.index(kmer)
                key = p_i, kmer_idx
                if not key in encoding:
                    encoding[key] = 0
                encoding[key] += 1
        self.tell(f'Building encoding into matrix')
        if self.dense:
            self.matrix = np.zeros(self.shape)
            for (p_i, k_i), v in encoding.items():
                self.matrix[p_i, k_i] = v
        else:
            row = []
            col = []
            data = []
            for (p_i, k_i), v in encoding.items():
                row.append(p_i)
                col.append(k_i)
                data.append(v)
            self.matrix = sparse.coo_matrix((data, (row, col)),
                                            shape=self.shape)
        return self.matrix.copy()

    def save(self, outfile):
        template = '{prot}\t{kmer}\t{freq}\n'
        with open(outfile, 'w', newline='\n') as out:
            if self.dense:
                row, col = np.where(self.matrix > 0)
                for i in range(len(row)):
                    p_i = row[i]
                    k_i = col[i]
                    out.write(template.format(prot=self.protein_idx[p_i],
                                              kmer=self.kmer_idx[k_i],
                                              freq=self.matrix[p_i, k_i]))
            else:
                for i in range(len(self.matrix.data)):
                    p_i = self.matrix.row[i]
                    k_i = self.matrix.col[i]
                    out.write(template.format(prot=self.protein_idx[p_i],
                                              kmer=self.kmer_idx[k_i],
                                              freq=self.matrix.data[i]))

