from Utils import FancyApp
from GOTool.GeneOntology import GeneOntology


class FilterGAF(FancyApp.FancyApp):

    def __init__(self, args):
        super(FilterGAF, self).__init__()
        self.gaf = args.gaf
        self.allowed_terms = [i.strip() for i in open(args.allowed_terms)]
        self.obo = args.obo
        self.go = None
        self.output = args.output

    def run(self):
        self.go = GeneOntology(self.obo, verbose=True)
        self.go.build_structure()
        self.go.load_annotation_file(self.gaf, 'GOA')
        self.tell('Up-propagating annotations')
        self.go.up_propagate_annotations('GOA')
        self.tell('Extracting annotations')
        annotations = self.go.get_annotations('GOA')
        condition = annotations['GO ID'].isin(self.allowed_terms)
        self.tell('Writing file')
        annotations[condition][['Protein', 'GO ID']].to_csv(self.output,
                                                            sep='\t',
                                                            header=False,
                                                            index=False)
