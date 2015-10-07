#!/usr/bin/env python
import logging
import numpy as np

def initializeLog(debug=False,warning=False,silent=False):
    import sys
    logging.basicConfig(format='%(levelname)s: %(message)s')
    if debug: logLevel = logging.DEBUG
    elif warning: logLevel = logging.WARNING
    elif silent: logLevel = logging.ERROR
    else: logLevel = logging.INFO
    logger = logging.getLogger()
    logger.setLevel(logLevel)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logLevel)
    logging.debug("Debugging messages on...") 
    logging.warning("Only warning messages will be printed ...") 
    return 0

def getArgs():
    import argparse
    parser = argparse.ArgumentParser(description=
    """
    September 18, 2015
    Murat Keceli
    Tools for molecular geometry in xyz format.
    """)
    parser.add_argument('input', metavar='FILE', type=str, nargs='*', help='input arguments')
    parser.add_argument('-d', '--debug', action='store_true', help='Print debug information.')
    parser.add_argument('-w', '--warning', action='store_true', help='Print warnings and errors.') 
    parser.add_argument('-s', '--silent', action='store_true', help='Print only errors.')
    parser.add_argument("-f", dest="filename", required=False, help=" input file", metavar="FILE")
    return parser.parse_args() 

def getCenter(xyz):
    """
    Returns the center point of given xyz.
    """
    npxyz=np.array([i[1:] for i in xyz])
    return np.sum(npxyz,axis=0)/len(npxyz)

def writeXYZ(xyz,xyzfile=None,comment='no comment'):
    """
    Writes an xyz formatted file with the given xyz.
    """
    if not xyzfile: xyzfile='tmp.xyz'
    N=len(xyz)
    with open(xyzfile,'w') as f:
        f.write("{0}\n".format(N))
        f.write(comment+"\n")
        for i in xrange(N):
            f.write("{0:s} {1:.14f} {2:.14f} {3:.14f} \n".format(xyz[i][0],xyz[i][1],xyz[i][2],xyz[i][3]))
    return xyzfile   

def sortXYZ(xyz,pivot=[0.,0.,0.]): 
    """
    Sorts the coordinates of atoms based on their distance from a pivot point.
    Thanks to Marco Verdicchio for the one liner.
    """
    return sorted(xyz,key=lambda x: (x[1]-pivot[0])**2+(x[2]-pivot[1])**2+(x[3]-pivot[2])**2)

def readXYZ(xyzfile):
    """
    Reads an xyz formatted file and returns the coordinates in a list.
    """
    with open(xyzfile) as f:
        line = f.readline()
        N=int(line.split()[0])
        line = f.readline()
        xyz = [['',0,0,0] for i in range((N))]
        for i in range(N):         
            line = f.readline()
            if not line:
                print "corrupt file at line:", i 
                break
            tmp=line.split()
            xyz[i]=[tmp[0],float(tmp[1]),float(tmp[2]),float(tmp[3])]
    return xyz
def xyz2PyQuanteMol(xyz):
    import PyQuante.Molecule 
    N=len(xyz)
    atoms = [('',(0,0,0)) for i in xrange(N)]
    for i in xrange(N):
        atoms[i] = (xyz[i][0],(xyz[i][1],xyz[i][2],xyz[i][3]))
    return PyQuante.Molecule(str(N),atoms,units='Angstrom') 

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
            atoms[i] = (chunks[0],(x,y,z))
    return PyQuante.Molecule(title,atoms,units='Angstrom')           
def main():
    import os.path
    args = getArgs()
    initializeLog(debug=args.debug,warning=args.warning,silent=args.silent)
    if os.path.isfile(args.filename):
        xyz = readXYZ(args.filename)
        center = getCenter(xyz)
        sortedxyz = sortXYZ(xyz)#,pivot=center)
        writeXYZ(sortedxyz)
    else:
        logging.debug('No input arguments given')
    return 0

if __name__ == "__main__":
    main()