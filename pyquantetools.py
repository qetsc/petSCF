#!/usr/bin/env python
import numpy as np
import PyQuante.Molecule as Molecule
import PyQuante.MINDO3 as MINDO3
import unittools as ut
from PyQuante.CGBF import CGBF
from Bunch import Bunch # Generic object to hold basis functions
from PyQuante.cints import overlap
from numba import jit, float32, void
# PyQuante functions/files are included here for convenience.
# Below is MINDO3_Parameters.py from PyQuante
# TODO: Check if this violates any license agreements.
###################################################################################
"""\
 MINDO3.py: Dewar's MINDO/3 Semiempirical Method

 This program is part of the PyQuante quantum chemistry program suite.

 Copyright (c) 2004, Richard P. Muller. All Rights Reserved. 

 PyQuante version 1.2 and later is covered by the modified BSD
 license. Please see the file LICENSE that is part of this
 distribution. 
"""

#MINDO/3 Parameters: Thru Ar
# in eV
Uss = [ None, -12.505, None,
        None, None, -33.61, -51.79, -66.06,
        -91.73, -129.86, None,
        None, None, None, -39.82, -56.23, -73.39, -98.99, None] 
Upp = [ None, None, None,
        None, None, -25.11, -39.18, -56.40, -78.80, -105.93, None,
        None, None, None, -29.15, -42.31, -57.25, -76.43, None]
gss = [ None, 12.848, None,
        None, None, 10.59, 12.23, 13.59, 15.42, 16.92, None,
        None, None, None, 9.82, 11.56, 12.88, 15.03, None]
gpp = [ None, None, None,
        None, None, 8.86, 11.08, 12.98, 14.52, 16.71, None,
        None, None, None, 7.31, 8.64, 9.90, 11.30, None]
gsp = [ None, None, None,
        None, None, 9.56, 11.47, 12.66, 14.48, 17.25, None,
        None, None, None, 8.36, 10.08, 11.26, 13.16, None]
gppp = [ None, None, None,
         None, None, 7.86, 9.84, 11.59, 12.98, 14.91, None,
         None, None, None, 6.54, 7.68, 8.83, 9.97, None]
hsp = [ None, None, None,
        None, None, 1.81, 2.43, 3.14, 3.94, 4.83, None,
        None, None, None, 1.32, 1.92, 2.26, 2.42, None]
hppp = [ None, None, None,
         None, None, 0.50, 0.62, 0.70, 0.77, 0.90, None,
         None, None, None, 0.38, 0.48, 0.54, 0.67, None]

f03 = [ None, 12.848, 10.0, #averaged repulsion integral for use in gamma
        10.0, 0.0, 8.958, 10.833, 12.377, 13.985, 16.250,
        10.000, 10.000, 0.000, 0.000,7.57 ,  9.00 ,10.20 , 11.73]

IPs = [ None, -13.605, None,
        None, None, -15.160, -21.340, -27.510, -35.300, -43.700, -17.820,
        None, None, None, None, -21.100, -23.840, -25.260, None]
IPp = [ None, None, None,
        None, None, -8.520, -11.540, -14.340, -17.910, -20.890, -8.510,
        None, None, None, None, -10.290, -12.410, -15.090, None]

# slater exponents
zetas = [ None, 1.30, None,
          None, None, 1.211156, 1.739391, 2.704546, 3.640575, 3.111270, None,
          None, None, None, 1.629173, 1.926108, 1.719480, 3.430887, None]
zetap = [ None, None, None,
          None, None, 0.972826, 1.709645, 1.870839, 2.168448, 1.419860, None,
          None, None, None, 1.381721, 1.590665, 1.403205, 1.627017, None]
# Bxy resonance coefficients
#MK: added symmetric ones
Bxy = {
    (1,1)  : 0.244770, 
    (1,5)  : 0.185347, 
    (1,6)  : 0.315011, 
    (1,7)  : 0.360776,
    (1,8)  : 0.417759, 
    (1,9)  : 0.195242, 
    (1,14) : 0.289647, 
    (1,15) : 0.320118,
    (1,16) : 0.220654, 
    (1,17) : 0.231653,
    (5,1)  : 0.185347,
    (5,5)  : 0.151324, 
    (5,6)  : 0.250031, 
    (5,7)  : 0.310959, 
    (5,8)  : 0.349745,
    (5,9)  : 0.219591,
    (6,1)  : 0.315011, 
    (6,5)  : 0.250031, 
    (6,6)  : 0.419907, 
    (6,7)  : 0.410886, 
    (6,8)  : 0.464514, 
    (6,9)  : 0.247494,
    (6,14) : 0.411377, 
    (6,15) : 0.457816, 
    (6,16) : 0.284620, 
    (6,17) : 0.315480,
    (7,1)  : 0.360776, 
    (7,5)  : 0.310959, 
    (7,6)  : 0.410886,
    (7,7)  : 0.377342, 
    (7,8)  : 0.458110, 
    (7,9)  : 0.205347,
    (7,15) : 0.457816, # Rick hacked this to be the same as 6,15
    (8,1)  : 0.417759, 
    (8,5)  : 0.349745, 
    (8,6)  : 0.464514,
    (8,7)  : 0.458110, 
    (8,8)  : 0.659407, 
    (8,9)  : 0.334044, 
    (8,15) : 0.457816, # Rick hacked this to be the same as 6,15
    (9,1)  : 0.195242, 
    (9,5)  : 0.219591, 
    (9,6)  : 0.247494,
    (9,7)  : 0.205347, 
    (9,8)  : 0.334044, 
    (9,9)  : 0.197464,
    (14,14): 0.291703, 
    (15,15): 0.311790, 
    (16,16): 0.202489,
    (17,17): 0.258969,
    }

# axy Core repulsion function terms
axy = {     
    (1,1) : 1.489450, (1,5) : 2.090352, (1,6) : 1.475836, (1,7) : 0.589380,
    (1,8) : 0.478901, (1,9) : 3.771362, (1,14) : 0.940789, (1,15) : 0.923170,
    (1,16) : 1.700689, (1,17) : 2.089404,
    (5,5) : 2.280544, (5,6) : 2.138291, (5,7) : 1.909763, (5,8) : 2.484827,
    (5,9) : 2.862183,
    (6,6) : 1.371208, (6,7) : 1.635259, (6,8) : 1.820975, (6,9) : 2.725913,
    (6,14) : 1.101382, (6,15) : 1.029693, (6,16) : 1.761370, (6,17) : 1.676222,
    (7,7) : 2.209618, (7,8) : 1.873859, (7,9) : 2.861667,
    (8,8) : 1.537190, (8,9) : 2.266949, (9,9) : 3.864997,
    (14,14) : 0.918432, (15,15) : 1.186652, (16,16) : 1.751617,
    (17,17) : 1.792125,
    (7,15) : 1.029693, # Rick hacked this to be the same as 6,15
    (8,15) : 1.029693, # Rick hacked this to be the same as 6,15
    }

# Atomic heat of formations: Mopac got from CRC
Hfat = [ None, 52.102, None,
         None, None, 135.7, 170.89, 113.0, 59.559, 18.86, None,
         None, None, None, 106.0, 79.8, 65.65, 28.95, None]

# Default isolated atomic energy values from Mopac:EISOL3
Eat = [None, -12.505, None,
       None ,None,-61.70,-119.47,-187.51,-307.07,-475.00,None,
       None,None,None,-90.98,-150.81,-229.15,-345.93,None]

nbfat = [ None, 1, None,
          None, None, 4, 4, 4, 4, 4, None,
          None, None, None, 4, 4, 4, 4, None]

CoreQ = [ None, 1, None,
          None, None, 3, 4, 5, 6, 7, None,
          None, None, None, 4, 5, 6, 7, None]

NQN = [ None, 1, 1, # principle quantum number N
        2, 2, 2, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3, 3, 3]
###################################################################################
"""\
 Slater.py: Coefficients for fitting Gaussian functions to Slater
   Functions.These functions are STO-6G fits to Slater exponents
   with exponents of 1. To fit to exponents of \zeta, one need only
   multiply each exponent by \zeta^2. For STO-1G, STO-2G and other
   levels of fit, see the paper.

 Reference: RF Stewart, JCP 52, 431 (1970)

 This program is part of the PyQuante quantum chemistry program suite.

 Copyright (c) 2004, Richard P. Muller. All Rights Reserved. 

 PyQuante version 1.2 and later is covered by the modified BSD
 license. Please see the file LICENSE that is part of this
 distribution. 
"""
gauss_powers = [(0,0,0),(1,0,0),(0,1,0),(0,0,1)]
# Gaussian functions for fitting to Slaters. These functions are
# STO-6G fits to slater exponents with exponents of 1. To fit
# to exponents of \zeta, you need only multiply each
# exponent by \zeta^2
# The rest of these functions can be obtained from Stewart,
#  JCP 52, 431 (1970)

gexps_1s = [2.310303149e01,4.235915534e00,1.185056519e00,
            4.070988982e-01,1.580884151e-01,6.510953954e-02]
gcoefs_1s = [9.163596280e-03,4.936149294e-02,1.685383049e-01,
             3.705627997e-01,4.164915298e-01,1.303340841e-01]

gexps_2s = [2.768496241e01,5.077140627e00,1.426786050e00,
            2.040335729e-01,9.260298399e-02,4.416183978e-02]
gcoefs_2s = [-4.151277819e-03,-2.067024148e-02,-5.150303337e-02,
             3.346271174e-01,5.621061301e-01,1.712994697e-01]

gexps_2p = [5.868285913e00,1.530329631e00,5.475665231e-01,
            2.288932733e-01,1.046655969e-01,4.948220127e-02]
gcoefs_2p = [7.924233646e-03,5.144104825e-02,1.898400060e-01,
             4.049863191e-01,4.012362861e-01,1.051855189e-01]
gexps_3s = [3.273031938e00,9.200611311e-01,3.593349765e-01,
            8.636686991e-02,4.797373812e-02,2.724741144e-02]
gcoefs_3s = [-6.775596947e-03,-5.639325779e-02,-1.587856086e-01,
             5.534527651e-01,5.015351020e-01,7.223633674e-02]

gexps_3p = [5.077973607e00,1.340786940e00,2.248434849e-01,
            1.131741848e-01,6.076408893e-02,3.315424265e-02]
gcoefs_3p = [-3.329929840e-03,-1.419488340e-02,1.639395770e-01,
             4.485358256e-01,3.908813050e-01,7.411456232e-02]

gexps = { # indexed by N,s_or_p:
    (1,0) : gexps_1s,
    (2,0) : gexps_2s,
    (2,1) : gexps_2p,
    (3,0) : gexps_3s,
    (3,1) : gexps_3p
    }

gcoefs = {  # indexed by N,s_or_p:
    (1,0) : gcoefs_1s,
    (2,0) : gcoefs_2s,
    (2,1) : gcoefs_2p,
    (3,0) : gcoefs_3s,
    (3,1) : gcoefs_3p
    }
s_or_p = [0,1,1,1] # whether the func is s or p type, based on the L QN
###################################################################################

def initialize(atoms):
    "PyQuante: Assign parameters for the rest of the calculation"
    ibf = 0 # Counter to overall basis function count
    for atom in atoms:
        xyz = atom.pos()
        atom.Z = CoreQ[atom.atno]
        atom.basis = []
        atom.rho = ut.e2/f03[atom.atno]
        atom.nbf = nbfat[atom.atno]
        atom.Eref = Eat[atom.atno]
        atom.Hf = Hfat[atom.atno]
        atom.gss = gss[atom.atno]
        atom.gsp = gsp[atom.atno]
        atom.gpp = gpp[atom.atno]
        atom.gppp = gppp[atom.atno]
        atom.hsp = hsp[atom.atno]
        atom.hppp = hppp[atom.atno]
        for i in xrange(atom.nbf):
            bfunc = Bunch()
            atom.basis.append(bfunc)
            bfunc.index = ibf # pointer to overall basis function index
            ibf += 1
            bfunc.type = i # s,x,y,z
            bfunc.atom = atom # pointer to parent atom
            bfunc.cgbf = CGBF(xyz,gauss_powers[i])
            zi = gexps[(NQN[atom.atno],s_or_p[i])]
            ci = gcoefs[(NQN[atom.atno],s_or_p[i])]
            if i:
                zeta = zetap[atom.atno]
                bfunc.u = Upp[atom.atno]
                bfunc.ip = IPp[atom.atno]
            else:
                zeta = zetas[atom.atno]
                bfunc.u = Uss[atom.atno]
                bfunc.ip = IPs[atom.atno]
            for j in xrange(len(zi)):
                bfunc.cgbf.add_primitive(zi[j]*zeta*zeta,ci[j])
            bfunc.cgbf.normalize()
    return atoms

def initializeMindo3(atoms):
    """
    Initialize PyQuante MINDO3 calculation
    This takes more than 5 seconds on vesta,
    TODO: 
    Profile this function.
    """
    MINDO3.initialize(atoms)
    return atoms

def getOverlap(a,b):
        "Overlap matrix element with another CGBF"
        Sij = 0.
        for ipbf in a.prims:
            for jpbf in b.prims:
                Sij += (ipbf.coef*
                             jpbf.coef*
                             ipbf.norm*
                             jpbf.norm*
                             overlap(ipbf.exp,ipbf.powers,ipbf.origin,
                                     jpbf.exp,jpbf.powers,jpbf.origin) )
        return a.norm*b.norm*Sij
    
def getH2O():
    return Molecule('H2O',
                   [(8,  ( 0.00000000,     0.00000000,     0.04851804)),
                    (1,  ( 0.75300223,     0.00000000,    -0.51923377)),
                    (1,  (-0.75300223,     0.00000000,    -0.51923377))],
                   units='Angstrom')

def xyz2PyQuanteMolOld(xyz):
    """
    Convert xyz data to PyQuante molecule object
    """
    N=len(xyz)
    atoms = [('',(0,0,0)) for i in xrange(N)]
    for i in xrange(N):
        atoms[i] = (xyz[i][0],
                    (xyz[i][1] * ut.ang2bohr,
                     xyz[i][2] * ut.ang2bohr,
                     xyz[i][3] * ut.ang2bohr)
                    )
    return Molecule(str(N),atoms,units='Bohr') 

def xyz2PyQuanteMol(xyz):
    """
    Convert xyz data to PyQuante molecule object
    """
    return Molecule('PSCFmol',xyz,units='angs') 

def sxyz2PyQuanteMol(s,xyz):
    """
    Convert s , xyz to PyQuante molecule object
    Parameters
    ----------
    s   : N- numpy array of strings
    xyz : (N,3)- numpy array of floats
    """
    nat = len(s)
    sxyz = [(str(s[i]),(xyz[i,0],xyz[i,1],xyz[i,2])) for i in range(nat)]
    return Molecule('PSCFmol',sxyz,units='angs') 

def xyzFile2PyQuanteMol(xyzfile):
    """
    Reads xyz file and creates a PyQuante Molecule object
    xyz should be given in Angstroms
    Note that PyQuante.IO.XYZ.read_xyz(xyzfile)[0] assumes bohrs
    """
    with open(xyzfile) as f:
        line = f.readline()
        N=int(line.split()[0])
        title = f.readline()
        atoms = [('',(0,0,0)) for i in xrange(N)]
        for i in xrange(N):
            line = f.readline() 
            chunks = line.split()
            x,y,z = map(float,chunks[1:])
            atoms[i] = (chunks[0],(x*ut.ang2bohr,y*ut.ang2bohr,z*ut.ang2bohr))
    return Molecule(title,atoms,units='Bohr') 


def getGij(bfi,bfj):
    "PyQuante: Coulomb-like term for orbitals on the same atom"
    i,j = bfi.type,bfj.type
    assert bfi.atom is bfj.atom, "Incorrect call to get_g"
    if i==0 and j==0:
        return bfi.atom.gss
    elif i==0 or j==0:
        return bfi.atom.gsp
    elif i==j:
        return bfi.atom.gpp
    return bfi.atom.gppp

def getHij(bfi,bfj):
    "PyQuante: Exchange-like term for orbitals on the same atom"
    i,j = bfi.type,bfj.type
    assert bfi.atom is bfj.atom, "Incorrect call to get_h"
    if i==0 or j==0:
        return bfi.atom.hsp
    return bfi.atom.hppp

def getAlphaij(atnoi,atnoj):
    "PyQuante: Part of the scale factor for the nuclear repulsion"
    return axy[(min(atnoi,atnoj),max(atnoi,atnoj))]

def getBeta0ij(atnoi,atnoj):
    "PyQuante: Resonanace integral for coupling between different atoms"
    return Bxy[(min(atnoi,atnoj),max(atnoi,atnoj))]

def getScaleij(atnoi,atnoj,R):
    "PyQuante: Prefactor from the nuclear repulsion term"
    alpha = getAlphaij(atnoi,atnoj)
    if atnoi == 1:
        if atnoj == 7 or atnoj == 8:
            return alpha*np.exp(-R)
    elif atnoj == 1:
        if atnoi == 7 or atnoi == 8:
            return alpha*np.exp(-R)
    return np.exp(-alpha*R)

def getGammaij(rhoi,rhoj,rij2):
    """
    PyQuante: Returns Coulomb repulsion that goes to the proper limit at R=0"
    """
    return ut.e2 / np.sqrt(rij2 + 0.25 * (rhoi + rhoj)**2.)

def getEnukeij(atomi, atomj,Rij2=0):
    """
    Returns the nuclear repulsion energy between two atoms.
    Rij2 is the square of the distance between the atoms given in bohr^2.
    """
    Zi      = atomi.Z
    atnoi   = atomi.atno
    rhoi    = atomi.rho #e2/f03[atom.atno]
    Zj      = atomj.Z
    atnoj   = atomj.atno
    rhoj    = atomj.rho
    if not Rij2:
        Rij2 = atomi.dist2(atomj)
        if not Rij2:
            return 0.
    Rij2 = Rij2 * ut.bohr2ang * ut.bohr2ang   
    Rij = np.sqrt(Rij2)
    gammaij = getGammaij(rhoi, rhoj, Rij2)    
    scaleij = getScaleij(atnoi,atnoj,Rij)
    return ( Zi * Zj * gammaij +  abs(Zi * Zj * (ut.e2 / Rij - gammaij) * scaleij) )

def getGuessD(atoms,basis=None, opt=0):
    "Average occupation density matrix"
    if basis is None:
        basis = getBasis(atoms) 
    nbf = len(basis)
    D = np.zeros((nbf,nbf))
    if opt == 0 : 
        for i in xrange(nbf):
            atomi=basis[i].atom
            if atomi.atno == 1: 
                D[i,i] = atomi.Z/1.
            else:               
                D[i,i] = atomi.Z/4.
    else:
        pass            
    return D

def getF(atoms,D,F0=None,basis=None):
    if basis is None:
        basis = getBasis(atoms)
    if F0 is None:    
        F0 = getF0(atoms,basis)
    F1 = getF1(atoms,D,basis)
    F2 = getF2(atoms,D,basis)
    return F0 + F1 + F2

def getD(cevecs):
    """
    Returns the density matrix using the matrix
    of eigenvectors
    """
    return 2. * np.dot(cevecs, cevecs.T)

def runSCF(atoms,basis=None,F0=None,D=None,scfthresh=1.E-3,maxiter=40):
    """
    MINDO3 SCF driver for closed shell systems.
    Returns:
    Electronic energy in ev
    All eigenvalues
    All eigenvectors
    Final density matrix
    """
    if basis is None:
        basis = getBasis(atoms)
    if F0 is None:    
        F0 = getF0(atoms,basis)
    if D is None:
        D = getGuessD(atoms)
    nclosed, nopen = getNClosedNOpen(atoms)
    if nopen:
        print "Not implemented: Open shell system"
        return    
    eold = 0.            
    for i in range(maxiter):
        F  = getF(atoms,D,F0,basis)
        enew = np.sum(D*(F0+F))
        if abs(enew-eold) < scfthresh :
            break
        evals, evecs = np.linalg.eigh(F)
        D = 2. * np.dot(evecs[:,0:nclosed], (evecs[:,0:nclosed]).T)
        eold = enew
    return 0.5 * enew,evals,evecs,D        
   
def getNVE(atoms):
    "PyQuante: Return number of valence electrons for given atoms."
    nel = 0
    for atom in atoms: 
        nel += atom.Z
    return nel

def getNBF(atoms):
    "PyQuante: Return number of basis functions for given atoms."
    nbf = 0
    for atom in atoms: 
        nbf += atom.nbf
    return nbf

def getNClosedNOpen(atoms,nel=None,mult=None):
    "PyQuante: Get the number of open/closed orbitals based on nel & multiplicity"
    if nel is None:
        nel = getNVE(atoms)
    nclosed,nopen = divmod(nel,2)
    if mult: #test the multiplicity
        nopen = mult-1
        nclosed,ntest = divmod(nel-nopen,2)
        if ntest:
            raise Exception("Impossible nel, multiplicity %d %d " % (nel,mult))
    return nclosed,nopen

def getEref(atoms):
    """
    Pyquante: Returns reference energy in ev.
    Note: PyQuante returns it in kcal/mol.
    Eref = heat of formation - energy of atomization
    """
    eat = 0
    hfat = 0
    for atom in atoms:
        eat += atom.Eref
        hfat += atom.Hf
    return hfat * ut.kcal2ev - eat

def getBasis(atoms,nbf=None):
    """
    Returns the basis set in the order of atoms.
    """
    if nbf is None:
        nbf = getNBF(atoms)
    basis = [atoms[0].basis[0]]*nbf
    i=0
    for atom in atoms:
        for bf in atom.basis:
            basis[i] = bf
            i += 1
    return basis 

def getAtomIDs(basis):
    """
    Returns atom IDs for a given basis.
    Atom IDs are simply integer labels ([0,N) for N atoms) based on 
    how atoms are sorted in the basis.
    TODO:
    Probably not needed
    """
    nbf=len(basis)
    tmp = np.zeros(nbf,dtype=np.int)
    for i in xrange(nbf):
        tmp[i]=basis[i].atom.atid
    return tmp

def getF0(atoms,basis=None):
    "Form the zero-iteration (density matrix independent) Fock matrix"
    if basis is None:
        basis = getBasis(atoms)
    nbf = len(basis)
    nat = len(atoms)
    F0 = np.zeros((nbf,nbf))
    # U term 
    for i in xrange(nbf):
        F0[i,i] = basis[i].u

    # Nuclear attraction
    ibf = 0 # bf number of the first bfn on iat
    for iat in xrange(nat):
        atomi = atoms[iat]
        for jat in xrange(nat):
            atomj = atoms[jat]
            if iat == jat: continue
            gammaij = getPQGammaij(atomi,atomj)
            for i in xrange(atomi.nbf):
                F0[ibf+i,ibf+i] -= gammaij*atomj.Z
        ibf += atomi.nbf

    # Off-diagonal 
    for ibf in xrange(nbf):
        bfi = basis[ibf]
        ati = bfi.atom
        atnoi = ati.atno
        for jbf in xrange(ibf):
            bfj = basis[jbf]
            atj = bfj.atom
            atnoj = atj.atno
            betaij = getBeta0ij(atnoi,atnoj)
            Sij = bfi.cgbf.overlap(bfj.cgbf)
            IPij = bfi.ip + bfj.ip
            F0[ibf,jbf] = F0[jbf,ibf] = betaij*IPij*Sij
    return F0

def getF1(atoms,D,basis=None):
    "One-center corrections to the core fock matrix"
    if basis is None:
        basis = getBasis(atoms)
    nbf = len(basis)
    nat = len(atoms)
    F1 = np.zeros((nbf,nbf))

    ibf = 0 # bf number of the first bfn on iat
    for iat in xrange(nat):
        atomi = atoms[iat]
        for i in xrange(atomi.nbf):
            bfi = atomi.basis[i]
            gii = getGij(bfi,bfi)
            qi =  D[ibf+i,ibf+i]
            F1[ibf+i,ibf+i] = 0.5*qi*gii
            
            for j in xrange(atomi.nbf):  # ij on same atom
                if j != i:
                    bfj = atomi.basis[j]
                    qj = D[ibf+j,ibf+j]
                    gij = getGij(bfi,bfj)
                    pij = D[ibf+i,ibf+j]
                    hij = getHij(bfi,bfj)
                    # the following 0.5 is something of a kludge to match
                    #  the mopac results.
                    F1[ibf+i,ibf+i] += qj*gij - 0.5*qj*hij
                    F1[ibf+i,ibf+j] += 0.5*pij*(3*hij-gij) #MK off
        ibf += atomi.nbf
    return F1

def getF2(atoms,D,basis=None):
    "Two-center corrections to the core fock matrix"
    if basis is None:
        basis = getBasis(atoms)
    nbf = len(basis)
    nat = len(atoms)
    F2 = np.zeros((nbf,nbf))

    ibf = 0 # bf number of the first bfn on iat
    for iat in xrange(nat):
        atomi = atoms[iat]
        jbf = 0
        for jat in xrange(nat):
            atomj = atoms[jat]
            if iat != jat:
                gammaij = getPQGammaij(atomi,atomj)
                for i in xrange(atomi.nbf):
                    qi = D[ibf+i,ibf+i]
                    qj = 0
                    for j in xrange(atomj.nbf):
                        pij = D[ibf+i,jbf+j]
                        F2[ibf+i,jbf+j] -= 0.25*pij*gammaij #MK off
                        F2[jbf+j,ibf+i] = F2[ibf+i,jbf+j]
                        qj += D[jbf+j,jbf+j]
                        F2[jbf+j,jbf+j] += 0.5*qi*gammaij
                    F2[ibf+i,ibf+i] += 0.5*qj*gammaij
            jbf += atomj.nbf
        ibf += atomi.nbf
    return F2

def getPQNuclearEnergy(atoms):
    """
    Returns PyQuante Nuclear energy in ev.
    """
    return MINDO3.get_enuke(atoms)

def getPQReferenceEnergy(atoms):
    """
    Returns PyQuante referene energy in ev.
    """
    return MINDO3.get_reference_energy(atoms) * ut.kcal2ev

def getPQMINDO3Energy(atoms):
    """
    Returns PyQuante MINDO3 heat of formation energy in ev.
    SCF threshold in PyQuante is 0.001 ev.
    """
    return MINDO3.scf(atoms) * ut.kcal2ev

def getPQGammaij(atomi,atomj):
    "Coulomb repulsion that goes to the proper limit at R=0"
    return MINDO3.get_gamma(atomi, atomj)

def getXYZFromAtoms(atoms):
    """
    Returns xyz array for given atoms
    """
    nat = len(atoms)
    xyz = np.zeros((nat,3))
    for i,atom in enumerate(atoms):
        xyz[i,:] = atom.pos()
    return xyz

def getXYZFromBasis(atoms,basis=None):
    if basis is None:
        basis = getBasis(atoms)
    nbf = len(basis)
    xyz = np.zeros((nbf,3))
    for i,bas in enumerate(basis):
        xyz[i,:] = bas.atom.pos()
    return xyz

def testgetF0(atoms=None,basis=None):
    if atoms is None:
        atoms = Molecule('H2O',
                   [(8,  ( 0.00000000,     0.00000000,     0.04851804)),
                    (1,  ( 0.75300223,     0.00000000,    -0.51923377)),
                    (1,  (-0.75300223,     0.00000000,    -0.51923377))],
                   units='Angstrom')
        MINDO3.initialize(atoms)
    if basis is None:
        basis = getBasis(atoms)     
    pqF0 = MINDO3.get_F0(atoms)
    F0   = getF0(atoms,basis)
    assert np.allclose(F0,pqF0), "getF0 does not match PyQuante values"
    return True
    
def testgetF1(atoms=None,D=None,basis=None):
    if atoms is None:
        atoms = Molecule('H2O',
                   [(8,  ( 0.00000000,     0.00000000,     0.04851804)),
                    (1,  ( 0.75300223,     0.00000000,    -0.51923377)),
                    (1,  (-0.75300223,     0.00000000,    -0.51923377))],
                   units='Angstrom')
        MINDO3.initialize(atoms)
    if basis is None:
        basis = getBasis(atoms)
    if D is None:
        nbf = len(basis)
        D =  np.random.rand(nbf,nbf)        
    pqF1 = MINDO3.get_F1(atoms,D)
    F1   = getF1(atoms,D,basis)
    assert np.allclose(F1,pqF1), "getF1 does not match PyQuante values"
    return True
    
def testgetF2(atoms=None,D=None,basis=None):
    if atoms is None:
        atoms = Molecule('H2O',
                   [(8,  ( 0.00000000,     0.00000000,     0.04851804)),
                    (1,  ( 0.75300223,     0.00000000,    -0.51923377)),
                    (1,  (-0.75300223,     0.00000000,    -0.51923377))],
                   units='Angstrom')
        MINDO3.initialize(atoms)
    if basis is None:
        basis = getBasis(atoms)
    if D is None:
        nbf = len(basis)
        D =  np.random.rand(nbf,nbf)        
    pqF2 = MINDO3.get_F2(atoms,D)
    F2   = getF2(atoms,D,basis)
    assert np.allclose(F2,pqF2), "getF1 does not match PyQuante values"
    return True

def testrunSCF(atoms=None):
    if atoms is None:
        atoms = Molecule('H2O',
                   [(8,  ( 0.00000000,     0.00000000,     0.04851804)),
                    (1,  ( 0.75300223,     0.00000000,    -0.51923377)),
                    (1,  (-0.75300223,     0.00000000,    -0.51923377))],
                   units='Angstrom')
        MINDO3.initialize(atoms)
    nclosed = getNClosedNOpen(atoms)[0]
    basis = getBasis(atoms)
    F0 = getF0(atoms,basis)
    escf = runSCF(atoms,basis,F0)[0]
    epqscf = MINDO3.scfclosed(atoms,F0,nclosed)
    assert abs(escf-epqscf) < 0.01, "Escf,Epqscf={0},{1}".format(escf,epqscf)
    return True    