"""

NetGO - Main Script

This is the main entry point for NetGO, 
all the commands can be run by running this script using python 3.x

=======
License
=======

Copyright (c) 2021 Mateo Torres <mateo.torres@fgv.br>

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

"""

__author__ = 'Mateo Torres'
__email__ = 'mateo.torres@fgv.br'
__copyright__ = 'Copyright (c) 2021, Mateo Torres'
__license__ = 'MIT'
__version__ = '0.1'

import argparse
import commands
import os
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='NetGO'
    )
    subparsers = parser.add_subparsers(help='sub-command help', dest='subcommand')

    predict = subparsers.add_parser(
        'predict',
        description='NetGO predict: given a fasta file, a path to STRING files and a directory of trained models, it'
                    'will predict function for each protein in the FASTA file using NetGO',
        help='prediciton command'
    )
    predict.set_defaults(func=commands.predict)
    predict.add_argument('--training-directory', help='Training directory', required=True)
    predict.add_argument('--ltr-mode',
                         help='Determines which model will be trained (NetGO or GOLabeler)',
                         default='netgo',
                         choices=['netgo', 'golabeler'])
    predict.add_argument('--goa-components',
                         help='GOA annotations in TSV format to be used for train the component models.'
                              'Columns are: "protein" and "goterm", no header.',
                         required=True)
    predict.add_argument('--goa-ltr',
                         help='GOA annotations in TSV format to be used for train the LTR model.'
                              'Columns are: "protein" and "goterm", no header',
                         required=True)
    predict.add_argument('--graph-homology',
                         help='A BLAST output file where the queries are proteins in the provided fasta, and the'
                              ' subjects are proteins in the provided graph file. the parser assumes blastp with'
                              ' "outfmt 6" was run to generate this file',
                         required=True)
    predict.add_argument('--graph',
                         help='Path to the protein interactions file TSV format with p1, p2 and weight columns, '
                              'no header',
                         required=True)
    predict.add_argument('--homologs',
                         help='BLAST output file for BLAST-kNN the parser assumes blastp with'
                              ' "outfmt 6" was run to generate this file',
                         required=True)
    predict.add_argument('--fasta', help='Path to the protein sequence file', required=True)
    predict.add_argument('--output-directory',
                         help='path to an (ideally) empty directory where prediction files will be written',
                         required=True)
    predict.add_argument('--obo',
                         help='Path to a go.obo file containing the GO structure')
    predict.add_argument('--kmer-output',
                         help='path to the pre-processed fasta into a kmer file, it should be compatible with the'
                              'training kmer file',
                         required=True)
    predict.add_argument('--interpro-output',
                         help='InterPro output file',
                         required=True)
    predict.add_argument('--profet-output',
                         help='ProFET output file',
                         required=True)


    train = subparsers.add_parser(
        'train',
        description='NetGO training: given a training dataset, train and save the component and the LTR models to disk',
        help='train command'
    )
    train.set_defaults(func=commands.train)
    train.add_argument('--fasta-components',
                       help='FASTA file used for training the component models',
                       required=True)
    train.add_argument('--fasta-ltr',
                       help='FASTA file used for training the LTR model',
                       required=True)
    train.add_argument('--goa-components',
                       help='GOA annotations in TSV format to be used for train the component models.'
                            'Columns are: "protein" and "goterm", no header.',
                       required=True)
    train.add_argument('--goa-ltr',
                       help='GOA annotations in TSV format to be used for train the LTR model.'
                            'Columns are: "protein" and "goterm", no header',
                       required=True)
    train.add_argument('--graph-homology',
                       help='A tab separated file that lists a protein in the fasta file, a protein from the '
                            'STRING database, the bit-score, trand the NCBI taxonomy id of the STRING protein in '
                            'each line',
                       required=True)
    train.add_argument('--graph',
                         help='Path to the protein interactions file TSV format with p1, p2 and weight columns, '
                              'no header',
                         required=True)
    train.add_argument('--output-directory',
                       help='path to an (ideally) empty directory where trained model files will be stored',
                       required=True)
    train.add_argument('--kmer',
                       help='KMer file: a tab separated file with columns '
                            'accession, kmer, frequency (no header)',
                       required=True)
    train.add_argument('--interpro-output',
                       help='InterPro output file',
                       required=True)
    train.add_argument('--profet-output',
                       help='ProFET output file (usually named trainingSetFeatures.csv)',
                       required=True)
    train.add_argument('--homologs',
                       help='BLAST output file for BLAST-kNN',
                       required=True)
    train.add_argument('--goterms',
                       help='List of GO terms to create LR models. a file with a '
                            'GO term per line.',
                       default='all')
    train.add_argument('--obo',
                       help='Path to a go.obo file containing the GO structure')
    train.add_argument('--ltr-mode',
                       help='Determines which model will be trained (NetGO or GOLabeler)',
                       default='netgo',
                       choices=['netgo', 'golabeler'])

    filter_gaf = subparsers.add_parser(
        'filter-gaf',
        description='NetGO filter GAF: Given a GAF file and a list of allowed GO terms, it generates'
                    'an annotation file compatible with the training and testing commands. Annotations'
                    'are uppropagated.',
        help='filter GAF utility'
    )
    filter_gaf.set_defaults(func=commands.filter_gaf)
    filter_gaf.add_argument('--gaf',
                            help='GOA annotation file in GAF format',
                            required=True)
    filter_gaf.add_argument('--allowed-terms',
                            help='text file containing allowed go term (one per line)',
                            required=True)
    filter_gaf.add_argument('--obo',
                            help='go.obo file, used to build the structure of the Gene Ontology',
                            required=True)
    filter_gaf.add_argument('--output',
                            help='output file',
                            required=True)

    filter_string = subparsers.add_parser(
        'filter-string',
        description='NetGO filter STRING: Given the STRING links and mapping to UniProt files, and a list of uniprot '
                    'acesssions, this command filters the STRING database to include only interactors of the valid '
                    'proteins. A new file containing uniprot to uniprot identifiers will be created',
        help='filter STRING utility'
    )
    filter_string.set_defaults(func=commands.filter_string)
    filter_string.add_argument('--string-links',
                               help='Path to STRING links',
                               required=True)
    filter_string.add_argument('--string-mapping',
                               help='STRING to UniProt mapping')
    filter_string.add_argument('--networks',
                               help='which STRING networks to extract',
                               nargs='+', required=True)
    filter_string.add_argument('--output',
                               help='output file',
                               required=True)
    filter_string.add_argument('--proteins',
                               help='proteins file, one UNIPROT ACCESSION per line',
                               required=True)

    map_string = subparsers.add_parser(
        'map-string',
        description='NetGO map STRING: maps string IDs to UniProt accession IDs. This is necessary for the '
                    'train and predict commands.',
        help='filter STRING utility'
    )
    map_string.set_defaults(func=commands.map_string)
    map_string.add_argument('--string-links',
                               help='Path to STRING links',
                               required=True)
    map_string.add_argument('--string-mapping',
                            help='STRING to UniProt mapping',
                            required=True)
    map_string.add_argument('--output',
                            help='output file',
                            required=True)
    map_string.add_argument('--p1-uniprot',
                            help='If true, the first element will be treated as UniProt and transformed to accession '
                                 'if necessary.',
                            action='store_true')

    args = parser.parse_args()
    args.func(args)


    # try:
    #     args.func(args)
    # except AttributeError as e:
    #     print(e)
    #     parser.parse_args(['--help'])
