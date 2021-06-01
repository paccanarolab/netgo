from Utils import FancyApp
from tools.STRINGFilter import STRINGFilter

class FilterSTRING(FancyApp.FancyApp):

    def __init__(self, args):
        super(FilterSTRING, self).__init__()
        self.links = args.string_links
        self.mapping = args.string_mapping
        self.networks = args.networks
        self.output = args.output
        self.proteins = [l.strip() for l in open(args.proteins)]

    def run(self):
        filter = STRINGFilter(self.links, self.mapping)
        filter.filter_string(self.proteins, self.networks, self.output)