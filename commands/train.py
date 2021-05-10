from Utils import ColourClass, FancyApp, Utilities

class Train(FancyApp.FancyApp):

    def __init__(self, args):
        super(Train, self).__init__()
        self.fasta_components = args.fasta_components
        self.fasta_ltr = args.fasta_ltr
        self.goa_components = args.goa_components
        self.goa_ltr = args.goa_ltr
        self.homologs = args.homologs
        self.output_directory = args.output_directory
        self.interpro = args.interpro_output
        self.profet = args.profet_output
        self.blast = args.profet_output
        # Note: the difference between blast and homologs is that
        # blast is to train BLAST-kNN. For training, that means
        # this is a blast between SwissProt and proteins found in
        # GOA_components

    def run(self):
        pass