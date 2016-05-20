#!/usr/bin/env python
import logging
import numpy as np
import unittools as const

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

def getChainMol(N=8,Z=1,d=1.):
    import PyQuante.Molecule
    """
    Generates the coordinates of a simple chain of $N$ atoms (with an atomic number $Z$)  separated by the given distance $d$ in Angstroms.
    Creates a PyQuante Molecule object with the generated coordinates.
    """
    mol=[[Z,(0.,0.,i*d)] for i in range(N)]
    return PyQuante.Molecule('chain',mol,units='Bohr')

def getCenter(xyz):
    """
    Returns the center point of given xyz.
    """
    npxyz=np.array([i[1:] for i in xyz])
    return np.sum(npxyz,axis=0)/len(npxyz)

def getExtendedArray(ext,a):
    """
    Returns an array with redundant rows based on 
    redundancy counts given in a 1D array.
    ext and a should have the same length.
    The new array has a length equal to sum(ext)
    Parameters
    ----------
    
    Returns
    -------
    b :
    TODO: 
    There can be a more general algorithm for all 
    shapes of a.
    """
    assert len(ext) == len(a), "getExtendedArray requires given arrays to have the same length"
    newlen = np.sum(ext)
    andim = a.ndim
    k = 0 
    if andim ==1:
        nrow = a.shape[0]
        b = np.empty(newlen)
        for i in range(nrow):
            for j in range(ext[i]):
                b[k] = a[i]
                k += 1
    elif andim == 2:
        nrow,ncol = a.shape
        b = np.empty((newlen,ncol))
        for i in range(nrow):
            for j in range(ext[i]):
                b[k,:] = a[i]
                k += 1
    else:
        print 'getExtendedArray requires ndim = 1 or 2, ndim= {0}'.format(andim)
    return b

def getNBFFromS(s):
    """
    Returns the total number of basis functions for 
    a given array of atomic symbols.
    """
    nbf = 0
    for i in s:
        if i.capitalize() == 'H':
            nbf += 1
        else:
            nbf += 4
    return nbf

def getNBFsFromS(s):
    """
    Returns an array of number of basis functions 
    for a given array of atomic symbols.
    """
    nat = len(s)
    nbfs = np.zeros(nat,dtype=np.int)
    for i in range(nat):
        if s[i].capitalize() == 'H':
            nbfs[i] = 1
        else:
            nbfs[i] = 4
    return nbfs

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

def sortXYZlist(xyz,pivot=[0.,0.,0.]): 
    """
    Sorts the coordinates of atoms based on their distance from a pivot point.
    Thanks to Marco Verdicchio for the one liner.
    """
    return sorted(xyz,key=lambda x: (x[1]-pivot[0])**2+(x[2]-pivot[1])**2+(x[3]-pivot[2])**2)

def readXYZlist(xyzfile):
    """
    Reads an xyz formatted file (in Angstroms) and returns the coordinates in a list.
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

def readSXYZ(xyzfile):
    """
    Reads an xyz formatted file (in Angstroms) and returns 
    two numpy arrays:
    Returns
    -------
    s   : N-dim array of strings
    xyz : (N,3)-dim array of floats
    """
    with open(xyzfile) as f:
        lines = f.readlines()
    N = int(lines[0])
    s = np.empty(N,dtype='S2')
    xyz = np.empty((N,3),dtype='d')
    for i in range(N):
        tmp     = lines[i+2].split()
        s[i],xyz[i,:] = tmp[0],tmp[1:4]
    return s, xyz

def readXYZ(xyzfile):
    """
    Reads an xyz formatted file (in Angstroms) and returns the coordinates in a list.
    """
    with open(xyzfile) as f:
        lines = f.readlines()
    N = int(lines[0])
    xyz = [('',(0.,0.,0.))]*N 
    for i in range(N):
        tmp     = lines[i+2].split()
        xyz[i] = (tmp[0],(float(tmp[1]),float(tmp[2]),float(tmp[3])))
    return xyz

def sortXYZ(xyz,pivot=[0.,0.,0.]): 
    """
    Sorts the coordinates of atoms based on their distance from a pivot point.
    Thanks to Marco Verdicchio for the one liner.
    """
    return sorted(xyz,key=lambda x: (x[1][0]-pivot[0])**2+(x[1][1]-pivot[1])**2+(x[1][2]-pivot[2])**2)

def getSortingIndices(xyz,pivot = 0):
    """
    Returns the indices of xyz, sorted by the distance
    from a pivot point.
    """
    if pivot != 0.:
        dist = np.linalg.norm(xyz,axis=1)
    else:
        dist = np.linalg.norm(xyz-pivot,axis=1)
    return np.argsort(dist)

def getSortedXYZ(xyz,pivot = None):
    """
    Returns sorted xyz, sorted by the distance
    from a pivot point.
    """
    idx = getSortingIndices(xyz, pivot)
    return xyz[idx]

def write_XYZ(symbols, positions,filename='tmp.xyz',comment='generated by xyztools.py'):
    natoms = len(symbols)
    with open(filename,'w') as f:
        f.write("{0}\n".format(natoms))
        f.write(comment+"\n")
        for i in xrange(natoms):
            f.write("{0:s} {1:.14f} {2:.14f} {3:.14f} \n".format(symbols[i],positions[i,0],positions[i,1],positions[i,1]))
    return  
        
def main():
    import os.path
    args = getArgs()
    initializeLog(debug=args.debug,warning=args.warning,silent=args.silent)
    if os.path.isfile(args.filename):
        xyz = readXYZ(args.filename)
#        center = getCenter(xyz)
        sortedxyz = sortXYZ(xyz)#,pivot=center)
        writeXYZ(sortedxyz)
    else:
        logging.debug('No input arguments given')
    return 0

if __name__ == "__main__":
    main()