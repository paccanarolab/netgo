from Utils import FancyApp, Utilities
from rich.progress import track
import pandas as pd

class STRING2UniProt(FancyApp.FancyApp):

    def __init__(self, links_file, mapping_file, p1_is_uniprot=False):
        super(STRING2UniProt, self).__init__()
        self.links = links_file
        self.mapping = {}
        self._load_mapping(mapping_file)
        self.p1_is_uniprot = p1_is_uniprot

    def create_uniprot_links_file(self, output):
        total = Utilities.line_count(self.links)
        self.tell('Creating links file with UniProt accession numbers')
        with open(output, 'w', newline='\n') as out:
            for line in track(open(self.links), total=total, description='Mapping links file...'):
                p1, p2, rest = line.split('\t', maxsplit=2)
                if self.p1_is_uniprot and '|' in p1: # in case we are dealing with a uniprot thing
                    p1 = p1.split('|')[1]
                if self.p1_is_uniprot:
                    a1 = [p1]
                else:
                    a1 = self.mapping.get(p1, None)
                a2 = self.mapping.get(p2, None)
                if all([a1, a2]):
                    for acc1 in a1:
                        for acc2 in a2:
                            out.write(f'{acc1}\t{acc2}\t{rest}')

    def _load_mapping(self, mapping_file):
        self.tell('Loading UniProt to STRING mapping file')
        names = "species   uniprot_ac|uniprot_id   string_id   identity   bit_score".split()
        acc_idx = names.index('uniprot_ac|uniprot_id')
        str_idx = names.index('string_id')
        for line in open(mapping_file):
            if line[0] == '#':
                continue
            fields = line.split()
            accession = fields[acc_idx].split('|')[0]
            string = fields[str_idx]
            if fields[str_idx] not in self.mapping:
                self.mapping[string] = []
            self.mapping[string].append(accession)