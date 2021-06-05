from Utils import ColourClass, FancyApp, Utilities
from tools.profet_parser import ProFETParser
from tools.interpro_parser import InterProParser
from tools.kmer_parser import KMerParser
from FASTATool.FastaParser import FastaFile
from GOTool.GeneOntology import GeneOntology
from component_methods.go_frequency import GOFrequency
from component_methods.LRComponent import LRComponent
from component_methods.blast_knn import BLASTkNN
from LTR.go_ltr import LearnToRankGO
from rich.progress import track
import pandas as pd
import numpy as np
import os

class Predict(FancyApp.FancyApp):
    
    def __init__(self, args):
        super(Predict, self).__init__()
        self.training_directory = args.training_directory
        self.ltr_mode = args.ltr_mode
        self.goa_components_file = args.goa_components
        self.goa_components = None
        self.goa_ltr_file = args.goa_ltr
        self.goa_ltr = None
        self.graph_homology = args.graph_homology
        self.graph = args.graph
        self.homologs = args.homologs
        self.fasta_file = args.fasta
        self.fasta = None
        self.output_directory = args.output_directory
        self.protein_index = os.path.join(self.output_directory, 'protein_idx-predict.txt')
        self.proteins = None
        self.obo = args.obo
        self.go = GeneOntology(self.obo)
        self.interpro = args.interpro_output
        self.interpro_features_npy = os.path.join(self.output_directory, 'interpro.npy')
        self.kmer = args.kmer_output
        self.kmer_features_npy = os.path.join(self.output_directory, 'kmers.npy')
        self.profet = args.profet_output
        self.profet_features_npy = os.path.join(self.output_directory, 'profet.npy')
        self.go_frequency_file = os.path.join(self.training_directory,
                                              'GO-frequencies.txt')

    def run(self):
        self.load_fasta()
        self.load_annotations()
        self.make_feature_matrices()

    def predict_for_ltr(self):
        """
        This method is simply to organize the prediction of the input to the LTR model.
        It relies on the models trained before if any training is relevant.

        Basically, all models are saved to disk at this point.
        """
        self.tell('Using component models to create input for LTR model...')
        self.tell('GO Frequency model...')
        per_domain_outfile = os.path.join(self.output_directory, 'GO_frequency-per_domain.tsv')
        overall_outfile = os.path.join(self.output_directory, 'GO_frequency-overall.tsv')
        if not os.path.exists(per_domain_outfile) or not os.path.exists(overall_outfile):
            go_frequency_model = GOFrequency()
            go_frequency_model.load_trained_model(self.go_frequency_file, fmt='tsv')
            if not os.path.exists(per_domain_outfile):
                self.tell('per domain')
                per_domain = go_frequency_model.predict(proteins=self.proteins, mode='domain')
                self.tell(f'Saving prediction to {per_domain_outfile}')
                per_domain.to_csv(per_domain_outfile, sep='\t', index=False)
            if not os.path.exists(overall_outfile):
                self.tell('overall')
                overall = go_frequency_model.predict(proteins=self.proteins, mode='overall')
                self.tell(f'Saving prediction to {overall_outfile}')
                overall.to_csv(overall_outfile, sep='\t', index=False)
        else:
            self.tell('GO frequency LTR files already exist, skipping computation')

        self.tell('BLAST-kNN')
        blast_knn_prediction = os.path.join(self.output_directory, 'BLAST-kNN.tsv')
        if not os.path.exists(blast_knn_prediction):
            blast_knn_B = os.path.join(self.output_directory, 'BLAST-kNN-B.npy')
            blast_knn = BLASTkNN()
            if not os.path.exists(blast_knn_B):
                blast_knn.train(self.goa_components,
                                blast_file=self.homologs,
                                proteins=self.proteins)
                blast_knn.save_trained_model(blast_knn_B)
            else:
                self.tell(f'Found pretrained BLAST-kNN, loading file {blast_knn_B}')
                blast_knn.load_trained_model(blast_knn_B,
                                             function_assignment=self.goa_components)
            blast_knn_cache = os.path.join(self.output_directory, 'BLAST-kNN-cache.pkl')
            blast_knn_pred = blast_knn.predict(self.proteins,
                                               go=self.go,
                                               output_complete_prediction=blast_knn_cache)
            blast_knn_pred.to_csv(blast_knn_prediction, sep='\t', index=False)
        else:
            self.tell('BLAST-kNN LTR file already exist, skipping computation')

        self.tell('LR-ProFET')
        lr_profet_prediction = os.path.join(self.output_directory, 'LR-ProFET.tsv')
        if not os.path.exists(lr_profet_prediction):
            lr_profet = LRComponent()
            lr_profet_model = os.path.join(self.training_directory, 'LR-ProFET.model')
            lr_profet_cache = os.path.join(self.output_directory, 'LR-ProFET.cache')
            lr_profet.load_trained_model(lr_profet_model)
            lr_pred = lr_profet.predict(self.proteins, go=self.go,
                                        feature_file=self.profet_features_npy,
                                        protein_index_file=self.protein_index,
                                        prediction_cache=lr_profet_cache)
            lr_pred.to_csv(lr_profet_prediction, sep='\t', index=False)
        else:
            self.tell('LR-ProFET LTR file already exists, skipping computation')

        self.tell('LR-kmer')
        lr_kmer_prediction = os.path.join(self.output_directory, 'LR-kmer.tsv')
        if not os.path.exists(lr_kmer_prediction):
            lr_kmer = LRComponent()
            lr_kmer_model = os.path.join(self.training_directory, 'LR-kmer.model')
            lr_kmer_cache = os.path.join(self.output_directory, 'LR-kmer.cache')
            lr_kmer.load_trained_model(lr_kmer_model)
            lr_pred = lr_kmer.predict(self.proteins, go=self.go,
                                      feature_file=self.kmer_features_npy,
                                      protein_index_file=self.protein_index,
                                      prediction_cache=lr_kmer_cache)
            lr_pred.to_csv(lr_kmer_prediction, sep='\t', index=False)
        else:
            self.tell('LR-kmer LTR file already exists, skipping computation')

        self.tell('LR-InterPro')
        lr_interpro_prediction = os.path.join(self.output_directory, 'LR-InterPro.tsv')
        if not os.path.exists(lr_interpro_prediction):
            lr_interpro = LRComponent()
            lr_interpro_model = os.path.join(self.training_directory, 'LR-InterPro.model')
            lr_interpro_cache = os.path.join(self.output_directory, 'LR-InterPro.cache')
            lr_interpro.load_trained_model(lr_interpro_model)
            lr_pred = lr_interpro.predict(self.proteins, go=self.go,
                                          feature_file=self.interpro_features_npy,
                                          protein_index_file=self.protein_index,
                                          prediction_cache=lr_interpro_cache)
            lr_pred.to_csv(lr_interpro_prediction, sep='\t', index=False)
        else:
            self.tell('LR-InterPro LTR file already exists, skipping computation')

    def make_feature_matrices(self):
        self.tell('Getting proteins for prediction index')
        proteins = set(self.fasta.information['proteins'].keys())
        self.proteins = sorted(list(proteins))

        self.tell('Saving protein index')
        Utilities.save_list_to_file(self.proteins, self.protein_index)

        if not os.path.exists(self.profet_features_npy):
            self.tell('Building ProFET Feature Matrix')
            profet_features = ProFETParser(self.profet)
            profet_matrix = np.zeros((len(self.proteins), len(profet_features.feature_cols)))
            for idx, protein in track(enumerate(self.proteins),
                                      total=len(self.proteins),
                                      description='Building ProFET Feature Matrix'):
                profet_matrix[idx] = profet_features[protein]
            self.tell('Saving ProFET feature matrix')
            np.save(self.profet_features_npy, profet_matrix)
        else:
            self.tell('ProFET feature matrix already exists, skipping calculation')

        if not os.path.exists(self.interpro_features_npy):
            self.tell('Building InterPro Feature Matrix')
            interpro_features = InterProParser(self.interpro)
            interpro_matrix = np.zeros((len(self.proteins), len(interpro_features.feature_cols)))
            for idx, protein in track(enumerate(self.proteins),
                                      total=len(self.proteins),
                                      description='Building ProFET Feature Matrix'):
                interpro_matrix[idx] = interpro_features[protein]
            self.tell('Saving InterPro feature matrix')
            np.save(self.interpro_features_npy, interpro_matrix)
        else:
            self.tell('InterPro feature matrix already exists, skipping calculation')

        if not os.path.exists(self.kmer_features_npy):
            self.tell('Building KMer Feature Matrix')
            kmer_features = KMerParser(self.kmer)
            kmer_matrix = np.zeros((len(self.proteins), len(kmer_features.feature_cols)))
            for idx, protein in track(enumerate(self.proteins),
                                      total=len(self.proteins),
                                      description='Building ProFET Feature Matrix'):
                kmer_matrix[idx] = kmer_features[protein]
            self.tell('Saving KMer feature matrix')
            np.save(self.kmer_features_npy, kmer_matrix)
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

    def load_fasta(self):
        self.tell('Loading Fasta')
        self.fasta = FastaFile(
            self.fasta_file,
            custom_header=r">(?P<db>[a-z]+)\|(?P<UniqueIdentifier>\w+)\|(?P<EntryName>\w+)")
        self.fasta.buildBrowsableDict()
