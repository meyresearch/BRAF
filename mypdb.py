import mdtraj as md
import os
import re
import numpy as np
import pandas as pd
from itertools import groupby
from tqdm import tqdm
from time import time as t
from pprint import pprint as pp
from Bio.SeqUtils import seq1
from IPython import embed as e
from traitlets.config import get_config
c = get_config()
c.InteractiveShellEmbed.colors = "Linux"
#e(config=c)

""" References:

https://www.wwpdb.org/documentation/file-format-content/format33/v3.3.html
http://www.bmsc.washington.edu/CrystaLinks/man/pdb/

General Layout of a PDB file:
~~~~~~~~
Title   |HEADER
	    |OBSLTE - Obsolete records
        |TITLE
		|SPLIT  - For describing macromolecular structures of sev. entries
	    |CAVEAT - Severe erros in file
        |COMPND - compound info
        |SOURCE - Biological & Chemical source of molecules
        |KEYWDS - keywords describing model
        |EXPDTA - Experimental method used 
		|NUMMDL - Number of models this entry compresies of
		|MDLTYP - General features of entry (specific info)
	    |AUTHOR
	    |REVDAT - Revision data
	    |SPRSDE - PDB entries superseeded by this model
        |JRNL   - Paper it's referenced in
        |REMARK - Remarks on the study - More to this to implemet
        |\REMARK 465 - Missing resiudes!
		|\REMARK 470 - Missing atoms
~~~~~~~~|
Prim.   |DBREF  - Link to database on sequence
Struct.	|DBREF1 - Alternate form of above for things that don't fit
Section	|DBREF2 - ""
        |SEQADV - Differences between above and below
        |SEQRES - Fasta sequnce of PDB given in three letter codes
	    |MODRES - Modified Residues
~~~~~~~~|
Het.    |HET    - Non-protein residues included
Atom    |HETNAM - Names of het atoms   (non-bio)
Section |HETSYN - Synonyms for the het atoms
	    |FORMUL - Formula of het atoms (non-bio)
~~~~~~~~|
Second. |HELIX  - Position of alpha helices
Struct. |SHEET  - Position of beta sheets
Section |TURN   - Short loops (seems depreceated in v3)
~~~~~~~~|
Connec. |SSBOND - Disulfide bond (links between thiols in CYS)
Annot.  |LINK   - Residue connectivity not implied by primary sequence
Section |CISPEP - cis-Conformation peptides
~~~~~~~~|
Misc.	|SITE   - Specifies residues which constitute essential sites
Feats.  |
~~~~~~~~|
Crystal |CRYST1
Section |ORIGXn
(TODO)  |SCALEn
		|MTRIXn
~~~~~~~~|
Coords  |MODEL  - For multi-model PDBs
Section |ATOM   - See below for more
		|ANISOU - Anistropic temp factors 
        |HETATM - Het atom info, presented same as ATOM records
        |TER    - End of sequence
        |ENDMDL - End of model
~~~~~~~~|
Connect.|CONECT - Connections not implied by the primary structure (see below)
Section |
~~~~~~~~|
Book    |MASTER - Global info about the file  (use for testing?)
Keeping |END    - END FILE
~~~~~~~~|
IMPORTANT STRUCTURES:

ATOM RECORDS:
COLUMNS        DATA  TYPE    FIELD        DEFINITION
-------------------------------------------------------------------------------------
 1 -  6        Record name   "ATOM  "
 7 - 11        Integer       serial       Atom  serial number.
13 - 16        Atom          name         Atom name.
17             Character     altLoc       Alternate location indicator.
18 - 20        Residue name  resName      Residue name.
22             Character     chainID      Chain identifier.
23 - 26        Integer       resSeq       Residue sequence number.
27             AChar         iCode        Code for insertion of residues.
31 - 38        Real(8.3)     x            Orthogonal coordinates for X in Angstroms.
39 - 46        Real(8.3)     y            Orthogonal coordinates for Y in Angstroms.
47 - 54        Real(8.3)     z            Orthogonal coordinates for Z in Angstroms.
55 - 60        Real(6.2)     occupancy    Occupancy.
61 - 66        Real(6.2)     tempFactor   Temperature  factor.
77 - 78        LString(2)    element      Element symbol, right-justified.
79 - 80        LString(2)    charge       Charge  on the atom.

HET ATOM RECORDS:
COLUMNS       DATA  TYPE     FIELD         DEFINITION
-----------------------------------------------------------------------
 1 - 6        Record name    "HETATM"
 7 - 11       Integer        serial        Atom serial number.
13 - 16       Atom           name          Atom name.
17            Character      altLoc        Alternate location indicator.
18 - 20       Residue name   resName       Residue name.
22            Character      chainID       Chain identifier.
23 - 26       Integer        resSeq        Residue sequence number.
27            AChar          iCode         Code for insertion of residues.
31 - 38       Real(8.3)      x             Orthogonal coordinates for X.
39 - 46       Real(8.3)      y             Orthogonal coordinates for Y.
47 - 54       Real(8.3)      z             Orthogonal coordinates for Z.
55 - 60       Real(6.2)      occupancy     Occupancy.
61 - 66       Real(6.2)      tempFactor    Temperature factor.
77 - 78       LString(2)     element       Element symbol; right-justified.
79 - 80       LString(2)     charge        Charge on the atom.

CONNECT RECORDS:
COLUMNS       DATA  TYPE      FIELD        DEFINITION
-------------------------------------------------------------------------
 1 -  6        Record name    "CONECT"
 7 - 11       Integer        serial       Atom  serial number
12 - 16        Integer        serial       Serial number of bonded atom
17 - 21        Integer        serial       Serial  number of bonded atom
22 - 26        Integer        serial       Serial number of bonded atom
27 - 31        Integer        serial       Serial number of bonded atom
		
        """
class PDB_file: # Currently don't initialize things for speed
	# Static Elements
    @staticmethod
    def _index_by(ll):
        idx = []
        count = 0
        cur = None
        for item in ll:
            if cur is None:
                cur = item
            elif cur != item:
                cur = item
                count += 1
            idx.append(count)
        return idx

    @staticmethod
    def _process_int(num):
        num = list(num)
        a = "".join(x for x in num)
        if a.isdigit():
            return int(a)
        else:
            for i,char in enumerate(num):
                if not char.isdigit():
                    num.pop(i)
            return int("".join(x for x in num))



    @classmethod
    def _fasta(cls,res_list):
        """ TODO : Terrible Method! Please change it urgently! """
        res_list = np.array(list(groupby(res_list)))[:,0]
        return "".join(cls._res_dict[code] for code in res_list)

	# List of first elements in a pdb
    _res_dict = {  # values = 3letter, keys = 1letter
                 "ALA"  : "A" ,
                 "CYS"  : "C" ,
                 "ASP"  : "D" ,
                 "GLU"  : "E" ,
                 "PHE"  : "F" ,
                 "GLY"  : "G" ,
                 "HIS"  : "H" ,
                 "ILE"  : "I" ,
                 "LYS"  : "K" ,
                 "LEU"  : "L" ,
                 "MET"  : "M" ,
                 "ASN"  : "N" ,
                 "PRO"  : "P" ,
                 "GLN"  : "Q" ,
                 "ARG"  : "R" ,
                 "SER"  : "S" ,
                 "THR"  : "T" ,
                 "VAL"  : "V" ,
                 "TRP"  : "W" ,
                 "TYR"  : "Y" }
########## CLASS METHODS ######################
    def __init__(self, pdb_file_path,
                verbose=None,debug=False):
        if verbose:
            print(f"PDB Path : {pdb_file_path}")
        else:
            self.verbose = True # TESTING
        if debug is True:
            self.debug = True
        else:
            self.debug = False
        self.file_path = pdb_file_path
        with open(pdb_file_path,"r") as f:
            text = f.readlines()
        self._rawtext = [line[:78] for line in text]
        self._sections = {  "title" : {"HEADER" : [],
                              "OBSLTE" : [],
   				              "TITLE"  : [],
                              "SPLIT"  : [],
   				              "CAVEAT" : [],
   				              "COMPND" : [],
   				              "SOURCE" : [],
   				              "KEYWDS" : [],
   				              "EXPDTA" : [],
                              "NUMMDL" : [],
                              "MDLTYP" : [],
   				              "AUTHOR" : [],
   				              "REVDAT" : [],
   				              "SPRSDE" : [],
   				              "JRNL"   : [],
   				              "REMARK" : []},
                 "primary" : {"DBREF"  : [],
					          "DBREF1" : [],
                              "DBREF2" : [],
                              "SEQRES" : [],
                              "MODRES" : []}, 
                     "het" : {"HET"    : [],        
                              "HETNAM" : [],        
                              "HETSYN" : [],        
                              "FORMUL" : []},        
               "secondary" : {"HELIX"  : [],       
                              "SHEET"  : [],       
                              "TURN"   : []},         
              "connection" : {"SSBOND" : [],       
                              "LINK"   : [],       
                              "CISPEP" : []},       
                    "site" : {"SITE"   : []},
                 "crystal" : {"CRYST1" : [],
                              "ORIGXn" : [],
                              "SCALEn" : [],
                              "MTRIXn" : []},
                    "data" : {"MODEL"  : [],
                              "ATOM"   : [],
                              "ANISOU" : [],
                              "HETATM" : [],
                              "TER"    : [],
                              "ENDMDL" : []},
                    "misc" : {"MASTER" : [],
                              "END"    : []}}
        self._atom_cols = {"serial" :     [slice( 6,11),int],
                  "name" :       [slice(13,16),str],
                  "altLoc" :     [slice(16,17),str],
                  "resName" :    [slice(17,20),str],
                  "chainID" :    [slice(21,22),str],
                  "resSeq" :     [slice(23,27),int],
                  "iCode" :      [slice(26,27),str],
                  "x" :          [slice(31,38),float],
                  "y" :          [slice(39,46),float],
                  "z" :          [slice(47,53),float],
                  "occupancy" :  [slice(55,60),float],
                  "betaFactor" : [slice(61,66),float],
                  "element" :    [slice(76,79),str],
                  "charge"  :    [slice(79,80),str],
                  "segment" :    [slice(72,76),str]}
        self._full_init()

    def __getitem__(self,key):
        for k,v in self._sections.items():
            if k == key:
                return v
            elif key in v:
                return v[key]
        if key in self._atom_cols.keys():
            return self._atom_cols[key]
        print(f"Item : {key} not found")


    def __contains__(self,key):
        for k,v in self._sections.items():
            if k == key:
                return v
            elif key in v:
                return v[key]
        if key in self._atom_cols.keys():
            return self._atom_cols[key]
        print(f"Item : {key} not found")
        raise ItemError

################## SORT THE DATA #####################
    def _check_length(self):            # Make sure raw text is all of same size
        lengths = [len(line) for line in self._rawtext] # Can either pad or strip?



    def _parse_pdb(self):
        t1 = t()
        for line in self._rawtext:
            sect = line.split()[0]
            for section in self._sections.values():
                if sect in section:
                    section[sect].append(line.strip())
        t2 = t()
        self._parse_atoms()
        if self.verbose:
            self._non_empty = self.check_data()

    def _parse_atoms(self):
        if self["ATOM"]:
            records = np.array([list(x) for x in self["ATOM"]])
            try: #TESTING
                for key,(columns,dtype) in self._atom_cols.items():
                    data = records[:,columns]
                    if dtype is int and len(data) > 99999:
                        # Convert to hexadecimal when this many atoms
                        data = [dtype("".join(x)) for x in data[:99999]]
                        [data.append(dtype("0x"+("".join(x)))) for x in data[99999:]]
                    elif dtype is int:  # some pdbs have int+char seqID (rare)
                        data = [self._process_int(x) for x in data]
                    else:
                        data = [dtype("".join(x)) for x in data]
                        if dtype is str:
                            data = [d.strip() for d in data]
                    self._atom_cols[key] = data
                # Custom data
                self._atom_cols["resID"] = self._index_by(self["resSeq"])
                self._atom_cols["segID"] = self._index_by(self["segment"])
                self._atom_cols["chainNum"] = self._index_by(self["chainID"])
                self.atoms = pd.DataFrame.from_dict(self._atom_cols)
                self.description = self.atoms.describe()
            except Exception as error: # For debugging 
                print(f"Error with {self.file_path}")
                print(type(error), error)
                if self.debug:
                    e(config=c)

        else:
            print("No ATOM data!")
            
    # Useful getters
    def get_chain(self,chainID):
        if hasattr(self,"atoms"):
            location = self.atoms["chainID"] == chainID
            return self.atoms.loc[location]
        else:
            print("Error: No atom data")
            raise Exception


    def get_seg(self,segID):
        if hasattr(self,"atoms"):
            location = self.atoms["segID"] == segID
            return self.atoms.loc[location]
        else:
            print("Error: No atom data")
            raise Exception


    def get_residue(self,residue_id):
        if hasattr(self,"atoms"):
            location = self.atoms["resID"] == residue_id
            return self.atoms.loc[location]
        else:
            print("Error: No atom data")
            raise Exception

    def get_residues(self,l_res):
        if hasattr(self,atoms):
            l_res = [self.get_residue(x) for x in l_res]
            return pd.concat(l_res)
        else:
            print("Error: No atom data")
            raise Exception


    def get_fasta(self,sort_by=None):
        if sort_by is None:
            return self._fasta(self["resName"])
        elif sort_by == "chainID" or sort_by == "segID":
            chains = self.atoms[sort_by].unique()
            n_chains = len(chains)
            chain_fasta = dict(zip(chains, [None]*n_chains))
            if n_chains > 1:
                for chain_name in chains:
                    location = (self.atoms[sort_by] == chain_name)
                    residues = self.atoms.loc[location]["resName"]
                    chain_fasta[chain_name] = self._fasta(residues)
                return chain_fasta
            else:
                print(f"Only one of {sort_by}")
                return self.get_fasta()
        elif sort_by == "SEQRES":
            chain_dict = {}
            chain_slice = slice(11,12)
            for line in self["SEQRES"]:
                cur_chain = line[chain_slice]
                chain_dict.setdefault(cur_chain,"")
                res_slice = slice(19,len(line))
                residues = line[res_slice].split()
                residues = [seq1(x) for x in residues]
                for x in residues:
                    chain_dict[cur_chain] += x
            return chain_dict

    def _non_standards(self):
        residues = self["resName"]
        self._uniqueRes = set(residues)
        non_standards = {} # Non-standard residues and their indices
        for indx,code in enumerate(residues):
            if code not in self._res_dict:
                non_standards.setdefault(code,[])
                non_standards[code].append(indx)
        if self.verbose:
            for code, indices in non_standards.items():
                print(f"Non-Standard residue : {code} consisting of "\
                      f"{len(indices)} atoms.")
        self.non_standard_res = non_standards


    def check_data(self):
        none_empty = []
        for k,v in self._sections.items():
            for sd in v.values():
                if sd:
                    none_empty.append(k)
        if none_empty:
            return none_empty
        else:
            print(f"No data found for PDB file: {self.file_path}")
            return None


    def to_dataframe(self):
        return pd.DataFrame.from_dict(self._atom_cols)

    def find_missing(self):
        """
        I think a better method is to look at this by comparing the RESSEQ
        against other methods as the missing residues section does not 
        always follow the same format because PDBs suck. 
        """
        missing_res = []
        missing_atoms = []
        remarks = self["REMARK"]
        for line in remarks:
            split = line.split()
            remark_num = split[1]
            if remark_num == 465:
                missing_res.append(" ".join(x for x in split[2:]))
            elif remark_num == 470:
                missing_atoms.append(" ".join(x for x in split[2:]))
        if missing_res:
            self.missing_residues = missing_res
        else: 
            self.missing_residues = None
        if missing_atoms:
            self.missing_atoms = missing_atoms
        else:
            self.missing_atoms = None

######## Main Init function #########################################
    def _full_init(self):
        self._check_length()
        self._parse_pdb()
        self._non_standards()
        self.find_missing()
        self.chain_fastas = self.get_fasta(sort_by="SEQRES")
        if not self.chain_fastas:
            self.chain_fastas = self.get_fasta()
        
"""
~~~~~~ END PDB CLASS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

def regex_filter(regexp):
    def r_f(thing):
        if regexp.match(thing):
            return True
        else:
            return False
    return r_f

def finder(path,filt=None):
    file_list = []
    for root,folder,files in os.walk(path):
        if filt:
            test = filter(filt,files)
        if list(test):
            files = sorted(files)
            files = filter(filt,files)
            files = [os.path.join(root,f) for f in files]
            [file_list.append(f) for f in files]
    return file_list


def get_fasta(traj,protein=True): # Check "is_protein" is valid
    if isinstance(traj,str):
        traj = md.load(traj)
    residues = [res.name for res in traj.top._residues if res.is_protein]
    seq = "".join(seq1(code) for code in residues) 
    return seq

class residue: # currently unused
    def __init__(self,
                resName,
                resID,
                resSeq,
                resStart,
                resEnd,
                 chainID):
        self.resName = resName
        self.id = resID
        self.resSeq = resSeq
        self.start = resStart
        self.end = resEnd
        self.chain = chainID

    def length(self):
        if hasattr(self,"_length"):
            return self.length
        else:
            self.length = self.end - self.start


if __name__ == '__main__':
    # Testing run
    #directory_path = "/mnt/hdd/work/data"
    directory_path = "/mnt/hdd/work/braf-craf/actual_work/results/blast_searches/PDBs/braf"
    pdb_regexp = re.compile(r".*\.pdb$")
    pdb_filter = regex_filter(pdb_regexp)
    pdb_paths = finder(directory_path,filt=pdb_filter)
    pp(pdb_paths)

    print(f"Processing {len(pdb_paths)} PDBS")
    t1 = t()
    pdbs = [PDB_file(pdb,verbose=False) for pdb in tqdm(pdb_paths)]
    t2 = t()
    time_taken = t2 - t1
    time_per_pdb = round(time_taken/len(pdbs),4)

    print(f"Time taken : {time_taken}")
    print(f"Time per PDB : {time_per_pdb}")
    e(config=c)

