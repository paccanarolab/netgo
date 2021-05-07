#!/bin/python

"""
A very basic uniprot fasta parser.
Just here to parse plain fasta files. Nothing fancier. If the file
does not EXACTLY comply with the format, this will crash.

=======
License
=======

Copyright (c) 2012 Horacio Caniza <h.j.canizavierci@cs.rhul.ac.uk>

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

__author__ = "Horacio Caniza"
__email__ = "h.j.canizavierci@cs.rhul.ac.uk"
__copyright__ = "Copyright (c) 2012, Horacio Caniza"
__license__ = "MIT"
__version__ = "0.1"

__all__ = ["FastaEntry", "FastaFile", "ParseError"]


import re

class ParseError(Exception):
    """Exception thrown when a parsing error occurred"""

    def __init__(self, msg, lineno = 1):
        Exception.__init__("%s near line %d" % (msg, lineno))
        self.lineno = lineno


class FastaEntry(object):
    """FASTA format:
    >db|UniqueIdentifier|EntryName ProteinName OS=OrganismName OX=OrganismIdentifier [GN=GeneName ]PE=ProteinExistence SV=SequenceVersion
    Sequence
    - ``db``: refers to the database.
    - ``UniqueIdentifier:`` Primary accession number for the database
    -``EntryName``: Name for the db entry
    -``ProteinName``:Recommended name of the protein accroding to uniprot.
    -``OrganismName``: Scientific Name of the organism
    -``OrganismIdentifier``: Unique identifier of the source organism, assigned by the NCBI
    -``GeneName``: First gene name. If no gene name, locus or ORF name, not listed.
    -``ProteinExistence: Evidence code for the existence of the protein
    -``SequenceVersion``: version number of the sequence.
    -``Sequence``: Aminoacid sequence
    """

    __slots__ = ["db", "UniqueIdentifier", "EntryName",
                 "ProteinName", "OrganismName", "OrganismIdentifier", "GeneName",
                 "ProteinExistence", "SequenceVersion", "Sequence"]

    def __init__(self, *args, **kwds):
        super(FastaEntry, self).__init__()
        for (name, value) in zip(self.__slots__, args):
            setattr(self, name, value)
        for name, value in kwds.items():
            setattr(self, name, kwds[value])
        for name in self.__slots__:
            if not hasattr(self, name):
                setattr(self, name, "")


class FastaFile(object):
    """A parser class that processes FASTA sequence files"""

    def __init__(self, fp, is_cafa=False, custom_header=None):
        super(FastaFile, self).__init__()
        self.information = {}
        self.already_built = False
        self.lineno = 0
        self.is_cafa = is_cafa
        self.custom_header = custom_header
        if custom_header is not None:
            self.custom_header = re.compile(custom_header)
        
        if isinstance(fp, (str)):
            if fp[-3:] == ".gz":
                from gzip import GzipFile
                self.fp = GzipFile(fp)
            else:
                self.fp = open(fp)
        else:
            self.fp = fp
            self.lineno = 0

    def buildBrowsableDict(self):
        if not self.already_built:
            self.already_built = True
            self.information['proteins'] = {}
            self.information['sequences'] = {}
            for annotation in self.annotations():
                self.information['proteins'][annotation.UniqueIdentifier] = annotation.Sequence
                self.information['sequences'][annotation.Sequence] = annotation.UniqueIdentifier

    def annotations(self):
        """Iterates over the annotations in this annotation file,
        yielding a `FastaEntry` object for each annotation."""

        db = str()
        UniqueIdentifier = str()
        EntryName = str()
        ProteinName = str()
        GeneName = str()
        ProteinExistence = str()
        SequenceVersion = str()
        Sequence = str()
        OrganismName = str()
        OrganismIdentifier = str()

        for line in self.fp.readlines():
            if line[0] == '>':
                if Sequence != "":
                    self.lineno += 1
                    try:
                        yield FastaEntry(db, UniqueIdentifier, EntryName, ProteinName, 
                                         OrganismName, OrganismIdentifier, GeneName,
                                         ProteinExistence, SequenceVersion, Sequence)
                    except TypeError as ex:
                        raise ParseError("Cannot parse annotation", self.lineno)
                    Sequence = ""
                if self.custom_header is not None:
                    match = self.custom_header.match(line)
                    db = match.groupdict().get('db', '')
                    UniqueIdentifier = match.groupdict().get('UniqueIdentifier', '')
                    EntryName = match.groupdict().get('EntryName', '')
                    ProteinName = match.groupdict().get('ProteinName', '')
                    GeneName = match.groupdict().get('GeneName', '')
                    ProteinExistence = match.groupdict().get('ProteinExistence', '')
                    SequenceVersion = match.groupdict().get('SequenceVersion', '')
                    Sequence = match.groupdict().get('Sequence', '')
                    OrganismName = match.groupdict().get('OrganismName', '')
                    OrganismIdentifier = match.groupdict().get('OrganismIdentifier', '')
                else:
                    identifier = line[1:].split('|')
                    if len(identifier) > 1:
                        # 'db'|'UniqueIdentifier'|'EntryName ProteinName OS=OrganismName[ GN=GeneName] PE=ProteinExistence SV=SequenceVersion'
                        # Filter the name of the database
                        # db
                        db = identifier[0]
                        # Filter the Unique Identifier
                        # UniqueIdentifier
                        UniqueIdentifier = identifier[1]
                        # Filter the Entry Name
                        EntryName = identifier[2].split(' ')[0]
                        # 'EntryName ProteinName OS=OrganismName[ GN=GeneName] PE=ProteinExistence SV=SequenceVersion'
                        # Get the ProteinName, ie, the stuff up to the substring OS=
                        # EntryName
                        ProteinName = (identifier[2][len(EntryName):identifier[2].find("OS=")].lstrip(' ')).rstrip(' ')
                        # get the GeneName. Since it's optional, we need to filter it and check it..awesome format..
                        if identifier[2].find("GN=") != -1:
                            # OS=ORganismName GN=GeneName PE=ProteinExistence SV=SequenceVersion'
                            remaining_string = identifier[2].split(' ')[-3:]
                            remaining_string.insert(0, identifier[2][
                                                    identifier[2].find("OS="):identifier[2].find("GN=")].rstrip(' '))
                            OrganismName = remaining_string[0][3:]
                            GeneName = remaining_string[1][3:]
                            ProteinExistence = remaining_string[2][3:]
                            SequenceVersion = remaining_string[3][3:]
                        else:
                            # OS=ORganismName PE=ProteinExistence SV=SequenceVersion'
                            remaining_string = identifier[2].split(' ')[-2:]
                            remaining_string.insert(0, identifier[2][
                                                    identifier[2].find("OS="):identifier[2].find("PE=")].rstrip(' '))
                            GeneName = ""
                            OrganismName = remaining_string[0][3:]
                            ProteinExistence = remaining_string[1][3:]
                            SequenceVersion = remaining_string[2][3:]
                    else:
                        if self.is_cafa:
                            UniqueIdentifier = identifier[0].split(' ')[1].strip()
                        else:
                            UniqueIdentifier = identifier[0].split(' ')[0]
            else:
                Sequence += line.rstrip("\n")
        if Sequence != "":
            self.lineno += 1
            try:
                yield FastaEntry(db, UniqueIdentifier, EntryName, ProteinName, 
                                 OrganismName, OrganismIdentifier, GeneName, ProteinExistence,
                                 SequenceVersion, Sequence)
            except TypeError as ex:
                raise ParseError("Cannot parse annotation", self.lineno)

    def __iter__(self):
        return self.annotations()


