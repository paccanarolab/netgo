import argparse
import commands
import os
import json

"""

prot2vec - Main Script

This is the main entry point for prot2vec, all the commands can be run by running this script using python 3.x

=======
License
=======

Copyright (c) 2018 Mateo Torres <Mateo.Torres.2015@live.rhul.ac.uk>

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
__email__ = 'Mateo.Torres.2015@live.rhul.ac.uk'
__copyright__ = 'Copyright (c) 2018, Mateo Torres'
__license__ = 'MIT'
__version__ = '0.1'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='prot2vec: dense embedding for proteins'
    )
    subparsers = parser.add_subparsers(help='sub-command help', dest='subcommand')

    neighbourhood = subparsers.add_parser(
        'neighbourhood',
        description='prot2vec neighbourhood: given a PPI file and a fasta file, builds a set of '
                    'neighbourhood pairs for training',
        help='neighbourhood generator command'
    )
    neighbourhood.set_defaults(func=commands.neighbourhood_generator)
    neighbourhood.add_argument('--fasta', help='Path to the protein sequence file', required=True)
    neighbourhood.add_argument('--graph', help='Path to the protein interaction file', required=True)
    neighbourhood.add_argument('--weighted', help='indicates that the graph is weighted (defaults to unweighted)',
                               action='store_true')
    neighbourhood.add_argument('--directed', help='indicates that the graph is directed (defaults to undirected)',
                               action='store_true')
    neighbourhood.add_argument('--weight-threshold',
                               help='only links with a weight >= threshold will be considered to build the '
                                    'neighbourhoods (defaults to 0.0)',
                               default=0.0, type=float)
    neighbourhood.add_argument('--output', help='Path to the output file', required=True)
    neighbourhood.add_argument('--sampling-method',
                               help='choose a sampling method (defaults to node2vec)',
                               default='node2vec', choices=['node2vec', 'sliding-window'])
    neighbourhood.add_argument('--max-len',
                               help='only sequences if this length of lower will be kept',
                               default='infer')
    neighbourhood.add_argument('--sampling-parameters',
                               help='parameters for the neighbourhood sampling method',
                               default={'num_walks': 10, 'walk_length': 80, 'p': 1.0, 'q': 1.0},
                               type=json.loads)
    # predict.add_argument('--cpu', help='Number of CPUs to use for parallelisable computations', default='infer')

    embedding = subparsers.add_parser(
        'embedding',
        description='prot2vec embedding: given a list of neighbourhoods, perform a dense embedding',
        help='embedding generator command'
    )
    embedding.set_defaults(func=commands.sequence_embedding)
    embedding.add_argument('--neighbourhoods', help='Path to the neighbourhoods file', required=True)
    embedding.add_argument('--output', help='Path to the output file', required=True)
    embedding.add_argument('--embedding-method',
                           help='choose an embedding method (defaults to word2vec)',
                           default='word2vec', choices=['word2vec', 'prot2vec'])
    embedding.add_argument('--embedding-parameters',
                           help='parameters for the embedding method',
                           default={'dimensions': 128, 'window_size': 10, 'workers': 'infer', 'iter': 1},
                           type=json.loads)

    graph_generator = subparsers.add_parser(
        'graphs',
        description='prot2vec graph generator: given a ppi, it generates all the transformed versions to replicate our '
                    'experiments',
        help='graph generator command'
    )
    graph_generator.set_defaults(func=commands.graph_generator)
    graph_generator.add_argument('--graph', help='path to the graph file', required=True)
    graph_generator.add_argument('--alias', help='name of the graphs to be generated', required=True)
    graph_generator.add_argument('--power-series', help='exponents to use to generate the power graphs',
                                 default='infer')

    calculate_similarity = subparsers.add_parser(
        'similarity',
        description='prot2vec similarity calculator: given an embedding, calculate the pairwise similarity between the'
                    'proteins',
        help='similarity calculator command'
    )
    calculate_similarity.set_defaults(func=commands.calculate_similarity)
    calculate_similarity.add_argument('--embedding', help='path to the embedding file', required=True)
    calculate_similarity.add_argument('--output', help='path to the output file', required=True)

    args = parser.parse_args()
    args.func(args)

    # try:
    #     args.func(args)
    # except AttributeError as e:
    #     print(e)
    #     parser.parse_args(['--help'])
