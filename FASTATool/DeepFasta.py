from Utils import *
from FASTATool import FastaParser

import numpy as np

import os
import pickle
import h5py

class DeepFasta(FancyApp.FancyApp):

    AMINOACIDS = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
        'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y', 'Z', 'X', '$'
    ]

    def __init__(self, fasta, output='infer', padded=True, max_len='infer', truncated=True):
        super(DeepFasta, self).__init__()
        self.padded = padded
        self.truncated = truncated
        self.max_len = max_len
        self.fasta = os.path.realpath(fasta)
        fasta_name = os.path.basename(self.fasta)
        self.output = os.path.join(
            os.path.dirname(fasta), '{f}-deep.pkl'.format(f=fasta_name)) if output == 'infer' else output
        self.data = {}

    def encode(self, mode='onehot'):
        pad_value = DeepFasta.AMINOACIDS.index('$')
        if not os.path.exists(self.output):
            self.tell('encoding fasta file', self.fasta)
            fp = FastaParser.FastaFile(self.fasta)
            fp.buildBrowsableDict()
            if self.max_len == 'infer':
                self.max_len = -1
                for seq in fp.information['proteins'].values():
                    if len(seq) > self.max_len:
                        self.max_len = len(seq)+1
            for prot_id, sequence in fp.information['proteins'].items():
                if self.truncated & len(sequence) > self.max_len:
                    continue
                seq = [DeepFasta.AMINOACIDS.index(i) for i in sequence]
                if self.padded:
                    # if we have space to spare, fill it with the stop char
                    if len(seq) < self.max_len:
                        seq.extend([pad_value] * (self.max_len - len(seq)))
                    # keep only max_len aminoacids
                    seq = seq[:self.max_len]
                    # make sure the last aminoacid is the stop char
                    seq[-1] = pad_value
                if mode == 'onehot':
                    m = np.zeros((len(seq), len(DeepFasta.AMINOACIDS)))
                    m[range(len(seq)), seq] = 1.0
                    self.data[prot_id] = m
                elif mode == 'index':
                    self.data[prot_id] = seq
            self.save()
        else:
            self.tell('deep encoding file found, loading...')
            self.tell('File location: ', self.output)
            self.data = pickle.load(open(self.output, 'rb'))
            self.max_len = len(next(iter(self.data.values())))

    def save(self, output='self'):
        output = self.output if output == 'self' else output
        out = open(output, 'wb')
        self.tell('saving data to', output)
        pickle.dump(self.data, out, 2)
