from Utils import ColourClass, FancyApp, Utilities
from tools.profet_parser import ProFETParser
from tools.interpro_parser import InterProParser
from tools.kmer_parser import KMerParser
from FASTATool.FastaParser import FastaFile
from GOTool.GeneOntology import GeneOntology
from component_methods.go_frequency import GOFrequency
from component_methods.LRComponent import LRComponent
from component_methods.blast_knn import BLASTkNN
from component_methods.net_knn import NETkNN
from LTR.go_ltr import LearnToRankGO
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
        # homologs here contains ALL homologs for the training, this is
        # * proteins involved in the component models vs GOA
        # * proteins used for the LTR model vs GOA
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

        self.LR_goterms = args.goterms
        if self.LR_goterms != 'all':
            self.LR_goterms = [line.strip() for line in open(args.goterms)]

        self.graph_homology = args.graph_homology
        self.graph = args.graph
        # Note: the difference between blast and homologs is that
        # blast is to train Net-kNN. This means that it contains the
        # BLAST information between the proteins used for training the LTR model
        # and STRING. There is no need to do a BLAST of all training proteins
        # and STRING, since those "prediction" will never be used.

        self.go_frequency_file = os.path.join(self.output_directory,
                                              'GO-frequencies.txt')
        self.obo = args.obo
        self.go = GeneOntology(self.obo)
        self.ltr_traininig_directory = os.path.join(self.output_directory, 'LTR-training-input')
        self.ltr_mode = args.ltr_mode
        self.ltr_model = os.path.join(self.output_directory, f'LTR-model-{self.ltr_mode}.json')

    def run(self):
        self.go.build_structure()
        self.load_fastas()
        self.load_annotations()
        # NOTE:
        # It's exremely important that the features for the training and
        # testing sets are matching.
        self.make_feature_matrices()
        self.train_component_models()
        self.predict_for_ltr()
        self.train_ltr()

    def train_ltr(self):
        ltr = LearnToRankGO(mode=self.ltr_mode)
        fa = pd.concat([self.goa_components, self.goa_ltr]).drop_duplicates()
        ltr.train(fa, component_models_dir=self.ltr_traininig_directory)
        ltr.save_trained_model(self.ltr_model)

    def predict_for_ltr(self):
        """
        This method is simply to organize the prediction of the input to the LTR model.
        It relies on the models trained before if any training is relevant.

        Basically, all models are saved to disk at this point.
        """
        ltr_proteins = sorted(self.fasta_ltr.information['proteins'].keys())
        if not os.path.exists(self.ltr_traininig_directory):
            os.makedirs(self.ltr_traininig_directory)

        self.tell('Using component models to create input for LTR model...')
        self.tell('GO Frequency model...')
        per_domain_outfile = os.path.join(self.ltr_traininig_directory, 'GO_frequency-per_domain.tsv')
        overall_outfile = os.path.join(self.ltr_traininig_directory, 'GO_frequency-overall.tsv')
        if not os.path.exists(per_domain_outfile) or not os.path.exists(overall_outfile):
            go_frequency_model = GOFrequency()
            go_frequency_model.load_trained_model(self.go_frequency_file, fmt='tsv')
            if not os.path.exists(per_domain_outfile):
                self.tell('per domain')
                per_domain = go_frequency_model.predict(proteins=ltr_proteins, mode='domain')
                self.tell(f'Saving prediction to {per_domain_outfile}')
                per_domain.to_csv(per_domain_outfile, sep='\t', index=False)
            if not os.path.exists(overall_outfile):
                self.tell('overall')
                overall = go_frequency_model.predict(proteins=ltr_proteins, mode='overall')
                self.tell(f'Saving prediction to {overall_outfile}')
                overall.to_csv(overall_outfile, sep='\t', index=False)
        else:
            self.tell('GO frequency LTR files already exist, skipping computation')

        self.tell('BLAST-kNN')
        blast_knn_prediction = os.path.join(self.ltr_traininig_directory, 'BLAST-kNN.tsv')
        if not os.path.exists(blast_knn_prediction):
            blast_knn_B = os.path.join(self.output_directory, 'BLAST-kNN-B.npy')
            blast_knn = BLASTkNN()
            if not os.path.exists(blast_knn_B):
                blast_knn.train(self.goa_components,
                                blast_file=self.homologs,
                                proteins=ltr_proteins)
                blast_knn.save_trained_model(blast_knn_B)
            else:
                self.tell(f'Found pretrained BLAST-kNN, loading file {blast_knn_B}')
                blast_knn.load_trained_model(blast_knn_B,
                                             function_assignment=self.goa_components)
            blast_knn_cache = os.path.join(self.output_directory, 'BLAST-kNN-cache.pkl')
            blast_knn_pred = blast_knn.predict(ltr_proteins,
                                               go=self.go,
                                               output_complete_prediction=blast_knn_cache)
            blast_knn_pred.to_csv(blast_knn_prediction, sep='\t', index=False)
        else:
            self.tell('BLAST-kNN LTR file already exist, skipping computation')


        self.tell('LR-ProFET')
        lr_profet_prediction = os.path.join(self.ltr_traininig_directory, 'LR-ProFET.tsv')
        if not os.path.exists(lr_profet_prediction):
            lr_profet = LRComponent()
            lr_profet_model = os.path.join(self.output_directory, 'LR-ProFET.model')
            lr_profet_cache = os.path.join(self.output_directory, 'LR-ProFET.cache')
            lr_profet.load_trained_model(lr_profet_model)
            lr_pred = lr_profet.predict(ltr_proteins, go=self.go,
                                        feature_file=self.profet_features_npy,
                                        protein_index_file=self.protein_index,
                                        prediction_cache=lr_profet_cache)
            lr_pred.to_csv(lr_profet_prediction, sep='\t', index=False)
        else:
            self.tell('LR-ProFET LTR file already exists, skipping computation')

        self.tell('LR-kmer')
        lr_kmer_prediction = os.path.join(self.ltr_traininig_directory, 'LR-kmer.tsv')
        if not os.path.exists(lr_kmer_prediction):
            lr_kmer = LRComponent()
            lr_kmer_model = os.path.join(self.output_directory, 'LR-kmer.model')
            lr_kmer_cache = os.path.join(self.output_directory, 'LR-kmer.cache')
            lr_kmer.load_trained_model(lr_kmer_model)
            lr_pred = lr_kmer.predict(ltr_proteins, go=self.go,
                                      feature_file=self.kmer_features_npy,
                                      protein_index_file=self.protein_index,
                                      prediction_cache=lr_kmer_cache)
            lr_pred.to_csv(lr_kmer_prediction, sep='\t', index=False)
        else:
            self.tell('LR-kmer LTR file already exists, skipping computation')

        self.tell('LR-InterPro')
        lr_interpro_prediction = os.path.join(self.ltr_traininig_directory, 'LR-InterPro.tsv')
        if not os.path.exists(lr_interpro_prediction):
            lr_interpro = LRComponent()
            lr_interpro_model = os.path.join(self.output_directory, 'LR-InterPro.model')
            lr_interpro_cache = os.path.join(self.output_directory, 'LR-InterPro.cache')
            lr_interpro.load_trained_model(lr_interpro_model)
            lr_pred = lr_interpro.predict(ltr_proteins, go=self.go,
                                          feature_file=self.interpro_features_npy,
                                          protein_index_file=self.protein_index,
                                          prediction_cache=lr_interpro_cache)
            lr_pred.to_csv(lr_interpro_prediction, sep='\t', index=False)
        else:
            self.tell('LR-InterPro LTR file already exists, skipping computation')

        self.tell('Net-kNN')
        net_knn_prediction = os.path.join(self.ltr_traininig_directory, 'Net-kNN.tsv')
        if not os.path.exists(net_knn_prediction):
            net_knn = NETkNN()
            net_knn_homologs = os.path.join(self.output_directory, 'Net-kNN.homologs.tsv')
            net_knn_neighborhood = os.path.join(self.output_directory, 'Net-kNN.neighborhood.pkl')
            if os.path.exists(net_knn_homologs) and os.path.exists(net_knn_neighborhood):
                self.tell(f'Found pretrained Net-kNN, loading file {net_knn_neighborhood}')
                net_knn.load_trained_model(self.output_directory)
            else:
                net_knn.train(self.goa_components,
                              string_links=self.graph,
                              blast_to_string=self.graph_homology,
                              proteins=ltr_proteins)
                net_knn.save_trained_model(self.output_directory)
            net_knn_pred = net_knn.predict(ltr_proteins, go=self.go)
            net_knn_pred.to_csv(net_knn_prediction, sep='\t', index=False)
        else:
            self.tell('Net-kNN LTR file already exists, skipping computation')


    def train_component_models(self):
        """
        The idea here is to train the models that need training using the
        component training data, and generate the predictions that will be used
        for the LTR model later on. Some of the models are "properly" trained
        before using them for prediciton, and some are similarity based to the
        training dataset, and will make predictions without supervision.
        """
        if not os.path.exists(self.go_frequency_file):
            self.tell('Learning GO frequencies from component annotation file')
            go_frequency_model = GOFrequency()
            go_frequency_model.train(self.goa_components,
                                     obo=self.obo)
            go_frequency_model.save_trained_model(self.go_frequency_file, fmt='tsv')
        else:
            self.tell(f'GO-frequency model is already trained and located here: '
                      f'{self.go_frequency_file}')

        # logistic regression models
        lr_kmer_model = os.path.join(self.output_directory, 'LR-kmer.model')
        if not os.path.exists(lr_kmer_model):
            self.tell('Training LR-kmer')
            lr_kmer = LRComponent()
            lr_kmer.train(self.goa_components,
                          type='kmer',
                          feature_file=self.kmer_features_npy,
                          protein_index_file=self.protein_index,
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
                              protein_index_file=self.protein_index,
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
                            protein_index_file=self.protein_index,
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
                                      description='Building InterPro Feature Matrix'):
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
                                      description='Building KMer Feature Matrix'):
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