#!/usr/bin/env python
from PyQuante import Molecule 
from PyQuante import MINDO3
from unittools import ang2bohr
def initializeMindo3(mol):
    """
    Initialize PyQuante MINDO3 calculation
    """
    MINDO3.initialize(mol)
    return mol
    
def getMol(mol):
    """
    Return PyQuante molecule object
    """
    return Molecule(Molecule)

def xyz2PyQuanteMol(xyz):
    """
    Convert xyz data to PyQuante molecule object
    """
    N=len(xyz)
    atoms = [('',(0,0,0)) for i in xrange(N)]
    for i in xrange(N):
        atoms[i] = (xyz[i][0],
                    (xyz[i][1] * ang2bohr,
                     xyz[i][2] * ang2bohr,
                     xyz[i][3] * ang2bohr)
                    )
    return Molecule(str(N),atoms,units='Bohr') 

def xyzFile2PyQuanteMol(xyzfile):
    """
    Reads xyz file and creates a PyQuante Molecule object
    xyz should be given in Angstroms
    Note that PyQuante.IO.XYZ.read_xyz(xyzfile)[0] assumes bohrs
    """
    import PyQuante.Molecule 
    with open(xyzfile) as f:
        line = f.readline()
        N=int(line.split()[0])
        title = f.readline()
        atoms = [('',(0,0,0)) for i in xrange(N)]
        for i in xrange(N):
            line = f.readline() 
            chunks = line.split()
            x,y,z = map(float,chunks[1:])
            atoms[i] = (chunks[0],(x*const.ang2bohr,y*const.ang2bohr,z*const.ang2bohr))
    return Molecule(title,atoms,units='Bohr')           
