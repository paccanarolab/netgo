from Utils import ColourClass, FancyApp, Utilities
from tools.profet_parser import ProFETParser
from tools.interpro_parser import InterProParser
from tools.kmer_parser import KMerParser
from tools.gaf_parser import GAFParser
from FASTATool.FastaParser import FastaFile
from component_methods.go_frequency import GOFrequency
from component_methods.LRComponent import LRComponent
from rich.progress import track
import numpy as np
import pandas as pd

import os


class Train(FancyApp.FancyApp):

    def __init__(self, args):
        super(Train, self).__init__()
        self.fasta_components_file = args.fasta_components
        self.fasta_components = None
        self.fasta_ltr_file = args.fasta_ltr
        self.fasta_ltr = None
        self.goa_components_file = args.goa_components
        self.goa_components = None
        self.goa_ltr_file = args.goa_ltr
        self.goa_ltr = None
        self.homologs = args.homologs
        self.output_directory = args.output_directory
        self.protein_index = os.path.join(self.output_directory, 'protein_idx-train.txt')
        self.interpro = args.interpro_output
        self.interpro_features_npy = os.path.join(self.output_directory,
                                                  f'{os.path.basename(self.interpro)}-train.npy')
        self.interpro_features_index = os.path.join(
            self.output_directory,
            f'{os.path.basename(self.interpro)}-train-feature-index.txt')
        self.profet = args.profet_output
        self.profet_features_npy = os.path.join(self.output_directory,
                                                f'{os.path.basename(self.profet)}-train.npy')
        self.profet_features_index = os.path.join(
            self.output_directory,
            f'{os.path.basename(self.profet)}-train-feature-index.txt')
        self.kmer = args.kmer
        self.kmer_features_npy = os.path.join(self.output_directory,
                                              f'{os.path.basename(self.kmer)}-train.npy')
        self.kmer_features_index = os.path.join(
            self.output_directory,
            f'{os.path.basename(self.kmer)}-train-feature-index.txt')
        self.blast = args.blast_output
        # Note: the difference between blast and homologs is that
        # blast is to train BLAST-kNN. For training, that means
        # this is a blast between SwissProt and proteins found in
        # GOA_components
        self.LR_goterms = args.goterms
        if self.LR_goterms != 'all':
            self.LR_goterms = [line.strip() for line in open(args.goterms)]

    def run(self):
        self.load_fastas()
        self.load_annotations()
        # NOTE:
        # It's exremely important that the features for the training and
        # testing sets are matching.
        self.make_feature_matrices()
        self.train_component_models()

    def train_component_models(self):
        # logistic regression models
        lr_kmer_model = os.path.join(self.output_directory, 'LR-kmer.model')
        if not os.path.exists(lr_kmer_model):
            self.tell('Training LR-kmer')
            lr_kmer = LRComponent()
            lr_kmer.train(self.goa_components,
                          type='kmer',
                          feature_file=self.kmer_features_npy,
                          feature_index_file=self.protein_index,
                          output_dir=lr_kmer_model,
                          goterms=self.LR_goterms)
        else:
            self.tell(f'LR-kmer model is already trained and'
                      f' located here: {lr_kmer_model}')

        lr_interpro_model = os.path.join(self.output_directory, 'LR-InterPro.model')
        self.tell('Training LR-InterPro')
        if not os.path.exists(lr_interpro_model):
            lr_interpro = LRComponent()
            lr_interpro.train(self.goa_components,
                              type='interpro',
                              feature_file=self.interpro_features_npy,
                              feature_index_file=self.protein_index,
                              output_dir=lr_interpro_model,
                              goterms=self.LR_goterms)
        else:
            self.tell(f'LR-InterPro model is already trained and'
                      f' located here: {lr_interpro_model}')

        lr_profet_model = os.path.join(self.output_directory, 'LR-ProFET.model')
        if not os.path.exists(lr_profet_model):
            self.tell('Training LR-ProFET')
            lr_profet = LRComponent()
            lr_profet.train(self.goa_components,
                            type='profet',
                            feature_file=self.profet_features_npy,
                            feature_index_file=self.protein_index,
                            output_dir=lr_profet_model,
                            goterms=self.LR_goterms)
        else:
            self.tell(f'LR-ProFET model is already trained and'
                      f' located here: {lr_profet_model}')


    def make_feature_matrices(self):
        self.tell('Getting proteins for training index')
        proteins = set(self.fasta_components.information['proteins'].keys())
        proteins |= set(self.fasta_ltr.information['proteins'].keys())
        proteins = sorted(list(proteins))

        self.tell('Saving protein index')
        Utilities.save_list_to_file(proteins, self.protein_index)

        if not os.path.exists(self.profet_features_npy):
            self.tell('Building ProFET Feature Matrix')
            profet_features = ProFETParser(self.profet)
            profet_matrix = np.zeros((len(proteins), len(profet_features.feature_cols)))
            for idx, protein in track(enumerate(proteins),
                                      total=len(proteins),
                                      description='Building ProFET Feature Matrix'):
                profet_matrix[idx] = profet_features[protein]
            self.tell('Saving ProFET feature matrix')
            np.save(self.profet_features_npy, profet_matrix)
            self.tell('Saving ProFET feature index')
            Utilities.save_list_to_file(profet_features.feature_cols, self.profet_features_index)
        else:
            self.tell('ProFET feature matrix already exists, skipping calculation')

        if not os.path.exists(self.interpro_features_npy):
            self.tell('Building InterPro Feature Matrix')
            interpro_features = InterProParser(self.interpro)
            interpro_matrix = np.zeros((len(proteins), len(interpro_features.feature_cols)))
            for idx, protein in track(enumerate(proteins),
                                      total=len(proteins),
                                      description='Building ProFET Feature Matrix'):
                interpro_matrix[idx] = interpro_features[protein]
            self.tell('Saving InterPro feature matrix')
            np.save(self.interpro_features_npy, interpro_matrix)
            self.tell('Saving InterPro feature index')
            Utilities.save_list_to_file(interpro_features.feature_cols, self.interpro_features_index)
        else:
            self.tell('InterPro feature matrix already exists, skipping calculation')

        if not os.path.exists(self.kmer_features_npy):
            self.tell('Building KMer Feature Matrix')
            kmer_features = KMerParser(self.kmer)
            kmer_matrix = np.zeros((len(proteins), len(kmer_features.feature_cols)))
            for idx, protein in track(enumerate(proteins),
                                      total=len(proteins),
                                      description='Building ProFET Feature Matrix'):
                kmer_matrix[idx] = kmer_features[protein]
            self.tell('Saving KMer feature matrix')
            np.save(self.kmer_features_npy, kmer_matrix)
            self.tell('Saving KMer feature index')
            Utilities.save_list_to_file(kmer_features.feature_cols, self.kmer_features_index)
        else:
            self.tell('KMer feature matrix already exists, skipping calculation')

    def load_annotations(self):
        self.tell('Loading GO annotations for component models')
        self.goa_components = pd.read_csv(self.goa_components_file,
                                          sep='\t',
                                          names=['protein', 'goterm'])
        self.tell('Loading GO annotations for LTR')
        self.goa_ltr = pd.read_csv(self.goa_ltr_file,
                                   sep='\t',
                                   names=['protein', 'goterm'])

    def load_fastas(self):
        self.tell('Loading Components Fasta')
        self.fasta_components = FastaFile(
            self.fasta_components_file,
            custom_header=r">(?P<db>[a-z]+)\|(?P<UniqueIdentifier>\w+)\|(?P<EntryName>\w+)")
        self.fasta_components.buildBrowsableDict()

        self.tell('Loading LTR Fasta')
        self.fasta_ltr = FastaFile(
            self.fasta_ltr_file,
            custom_header=r">(?P<db>[a-z]+)\|(?P<UniqueIdentifier>\w+)\|(?P<EntryName>\w+)")
        self.fasta_ltr.buildBrowsableDict()