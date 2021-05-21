import pandas as pd
from Utils import ColourClass, FancyApp

class GAFParser(FancyApp.FancyApp):

    gaf_columns = [
        'DB', 'DB Object ID', 'DB Object Symbol', 'Qualifier', 'GO ID',
        'DB:Reference (IDB:Reference)', 'Evidence Code', 'With (or) From',
        'Aspect', 'DB Object Name', 'DB Object Synonym (ISynonym)', 'DB Object Type',
        'Taxon(Itaxon)', 'Date', 'Assigned By', 'Annoation Extension',
        'Gene Product Form ID'
    ]

    EXPERIMENTAL_EVIDENCE_CODES = ['EXP', 'IDA', 'IPI', 'IMP',
                                   'IGI', 'IEP', 'TAS', 'IC']
    ALL_EVIDENCE_CODES = [
        # experimental
        'EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP',
        # High Throughput
        'HTP', 'HDA', 'HMP', 'HGI', 'HEP',
        # Computational Analysis
        'ISS', 'ISO', 'ISA', 'ISM', 'IGC', 'IBA', 'IBD', 'IKR', 'IRD', 'RCA'
        # Author statement
                                                                       'TAS', 'NAS',
        # Curator statement
        'IC', 'ND',
        # Electronic Annotation
        'IEA'
    ]

    def __init__(self, gaf_file):
        """
        A pandas bases GAF parser.

        This is *not* optimized for memory, make sure you have enough
        memory to read the entire gaf file, or reduce the size of it.
        Parameters
        ----------
        gaf_file
            path to the gaf file
        """
        super(GAFParser, self).__init__()
        self.gaf_file = gaf_file
        self.tell(f'Loading GAF file: {self.gaf_file}')
        self.goa = pd.read_csv(self.gaf_file,
                          sep='\t',
                          names=self.gaf_columns,
                          header=None)
        self.tell(f'Processing GOA tax IDs')
        self.goa['datetime'] = pd.to_datetime(self.goa['Date'],
                                              format='%Y%m%d')
        self.goa['taxa'] = self.goa['Taxon(Itaxon)'].str.split('|')
        self.goa = self.goa.explode('taxa')
        self.goa['Tax ID'] = self.goa['taxa'].str.split(':').str[1]

    def get_assignment(self,
                       db='UniProtKB',
                       evidence_codes='experimental',
                       exclusion_list = None):
        """
        Get a list of protein and GO assignments from the GAF file

        Parameters
        ----------
        db : str, list of str, default 'UniProtKB'
            which
        evidence_codes : str, list of str, default 'experimental'
            a list of valid evidence codes, only annotations with evidence codes
            in the list will be added to the assignment list. Defaults to
            `EXPERIMENTAL_EVIDENCE_CODES`
            (EXP, IDA, IPI, IMP, IGI, IEP, TAS, IC)
        exclusion_list : list, optional
            list of taxonomy IDs to exclude. Annotations of proteins in such
            taxons will be excluded from the result. By default, no taxon is
            excluded

        Returns
        -------
        pandas DataFrame
            functional assignments with columns 'protein' and 'goterm'
        """
        if evidence_codes == 'experimental':
            evidence_codes = GAFParser.EXPERIMENTAL_EVIDENCE_CODES
        if isinstance(db, str):
            db = [db]

        condition = (self.goa['DB'].isin(db) &
                     self.goa['Evidence Code'].isin(evidence_codes))
        if exclusion_list is not None:
            condition &= ~self.goa['Tax ID'].isin(exclusion_list)

        return (self.goa[condition][['DB Object ID', 'GO ID']].copy()
                .rename(columns={'DB Object ID':'protein', 'GO ID':'goterm'}))

