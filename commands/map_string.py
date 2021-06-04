from Utils import FancyApp
from tools.string2uniprot import STRING2UniProt

class MapSTRING(FancyApp.FancyApp):

    def __init__(self, args):
        super(MapSTRING, self).__init__()
        self.links_file = args.string_links
        self.mapping_file = args.string_mapping
        self.output = args.output
        self.valid_string = args.valid_string
        if self.valid_string:
            self.valid_string = [i.strip() for i in open(self.valid_string)]
        self.p1 = args.p1_uniprot

    def run(self):
        mapper = STRING2UniProt(self.links_file, self.mapping_file,
                                valid_string=self.valid_string,
                                p1_is_uniprot=self.p1)
        mapper.create_uniprot_links_file(self.output)