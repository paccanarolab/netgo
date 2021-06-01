from Utils import FancyApp, Utilities
from rich.progress import track
import pandas as pd


class STRINGFilter(FancyApp.FancyApp):

    def __init__(self, links_file, mapping_file):
        """
        Filter STRING file
        Parameters
        ----------
        links_file : str
            Path to the STRING link file in txt format, this is usually
            named proteins.link.full.<version>.txt
        mapping_file : str
            Path to the UniProt to STRING mapping, this is usually
            named all_organisms.uniprot_2_string.<year>.tsv
        """
        super(STRINGFilter, self).__init__()
        self.links = links_file
        self.mapping = None
        self._load_mapping(mapping_file)

    def filter_string(self, proteins, networks, output):
        """
        Filters the string file, where all the interactions are related to
        the provided list of proteins.

        Parameters
        ----------
        proteins : list of str
            proteins to keep,
            IT ONLY WORKS WITH UNIPROT ACCESSION IDS
        networks : list of str
            networks to write to the output file
        output : str
            Path to the output file.
        """
        self.tell('Filtering STRING file')
        condition = self.mapping['accession'].isin(proteins)
        valid_mapping = self.mapping[condition]
        valid_string = self.mapping[condition]['string_id'].tolist()
        # this is probably a very long process, so we count the number of lines to estimate the time it will take to
        # filter STRING
        total_lines = Utilities.line_count(self.links)
        names = ['protein1', 'protein2', 'neighborhood', 'neighborhood_transferred', 'fusion', 'cooccurence',
                 'homology', 'coexpression', 'coexpression_transferred', 'experiments', 'experiments_transferred',
                 'database', 'database_transferred', 'textmining', 'textmining_transferred', 'combined_score']
        skip = 1
        with open(output, 'w', newline='\n') as f:
            for line in track(open(self.links), total=total_lines, description='Processing...'):
                if skip > 0:
                    skip -= 1
                    continue
                fields = line.strip().split()
                protein1 = fields[names.index('protein1')]
                protein2 = fields[names.index('protein2')]
                if protein1 in valid_string or protein2 in valid_string:
                    # retrieve both uniprot accessions
                    condition = valid_mapping['string_id'] == protein1
                    aliases_p1 = valid_mapping[condition]['accession'].tolist()

                    condition = valid_mapping['string_id'] == protein2
                    aliases_p2 = valid_mapping[condition]['accession'].tolist()

                    if len(aliases_p1) > 0 and len(aliases_p2) > 0:
                        for p1 in aliases_p1:
                            for p2 in aliases_p2:
                                entry = [p1, p2]
                                for n in networks:
                                    entry.append(fields[names.index(n)])
                                f.write('\t'.join(entry))
                                f.write('\n')

    def _load_mapping(self, mapping_file):
        self.tell('Loading UniProt to STRING mapping file')
        names = "species   uniprot_ac|uniprot_id   string_id   identity   bit_score".split()
        self.mapping = pd.read_csv(mapping_file,
                                   sep='\t',
                                   comment='#',
                                   names=names)
        self.mapping['accession'] = self.mapping['uniprot_ac|uniprot_id'].str.split('|').str[0]
        self.mapping.drop(columns=['species', 'uniprot_ac|uniprot_id', 'bit_score'])
