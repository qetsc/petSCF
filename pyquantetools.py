#!/usr/bin/env python
import PyQuante.Molecule as Molecule
import PyQuante.MINDO3 as MINDO3
import unittools as ut
import numpy as np

# Some PyQuante functions/files are included here for convenience.
# Below is MINDO3_Parameters.py from PyQuante
# TODO: Check if this violates any license aggreements.
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

def initializeMindo3(atoms):
    """
    Initialize PyQuante MINDO3 calculation
    This takes more than 5 seconds on vesta,
    TODO: 
    Profile this function.
    """
    MINDO3.initialize(atoms)
    return atoms

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