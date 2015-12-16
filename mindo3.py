import sys, petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np
import petsctools as pt
import slepctools as st
#import PyQuante
from mpi4py import MPI
import constants as const
#from PyQuante.MINDO3_Parameters import axy,Bxy,f03
from PyQuante.MINDO3 import initialize

Print = PETSc.Sys.Print
"""
MINDO/3 parameters taken from PyQuante
"""
#MINDO/3 Parameters: Thru Ar
# in eV
f03 = [ None, 12.848, 10.0, #averaged repulsion integral for use in gamma
        10.0, 0.0, 8.958, 10.833, 12.377, 13.985, 16.250,
        10.000, 10.000, 0.000, 0.000,7.57 ,  9.00 ,10.20 , 11.73]
# Bxy resonance coefficients
Bxy = {
    (1,1) : 0.244770, (1,5) : 0.185347, (1,6) : 0.315011, (1,7) : 0.360776,
    (1,8) : 0.417759, (1,9) : 0.195242, (1,14) : 0.289647, (1,15) : 0.320118,
    (1,16) : 0.220654, (1,17) : 0.231653,
    (5,5) : 0.151324, (5,6) : 0.250031, (5,7) : 0.310959, (5,8) : 0.349745,
    (5,9) : 0.219591,
    (6,6) : 0.419907, (6,7) : 0.410886, (6,8) : 0.464514, (6,9) : 0.247494,
    (6,14) : 0.411377, (6,15) : 0.457816, (6,16) : 0.284620, (6,17) : 0.315480,
    (7,7) : 0.377342, (7,8) : 0.458110, (7,9) : 0.205347,
    (8,8) : 0.659407, (8,9) : 0.334044, (9,9) : 0.197464,
    (14,14) : 0.291703, (15,15) : 0.311790, (16,16) : 0.202489,
    (17,17) : 0.258969,
    (7,15) : 0.457816, # Rick hacked this to be the same as 6,15
    (8,15) : 0.457816, # Rick hacked this to be the same as 6,15
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

def getAlphaij(atnoi,atnoj):
    "PyQuante: Part of the scale factor for the nuclear repulsion"
    return axy[(min(atnoi,atnoj),max(atnoi,atnoj))]

def getBeta0ij(atnoi,atnoj):
    "PyQuante: Resonanace integral for coupling between different atoms"
    return Bxy[(min(atnoi,atnoj),max(atnoi,atnoj))]

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
    return const.e2 / np.sqrt(rij2 + 0.25 * (rhoi + rhoj)**2.)

def getNelectrons(atoms,charge=0):
    "PyQuante: Number of electrons in an atoms. Can be dependent on the charge"
    nel = 0
    for atom in atoms: nel += atom.Z
    return nel-charge

def getNbasis(atoms):
    "Number of basis functions in an atom list"
    nbf = 0
    for atom in atoms: nbf += atom.nbf
    return nbf

def getEref(atoms):
    "Ref = heat of formation - energy of atomization"
    eat = 0
    hfat = 0
    for atom in atoms:
        eat += atom.Eref
        hfat += atom.Hf
    return hfat-eat * const.ev2kcal

def getEnukeij(atomi, atomj,Rij2=0):
    """
    Returns the nuclear repulsion energy between two atoms.
    Rij2 is the square of the distance between the atoms given in bohr^2.
    """
    Zi      = atomi.Z
    atnoi   = atomi.atno
    rhoi    = atomi.rho
    Zj      = atomj.Z
    atnoj   = atomj.atno
    rhoj    = atomj.rho
    if not Rij2:
        Rij2 = atomi.dist2(atomj)
        if not Rij2:
            return 0.
    Rij2 = Rij2 * const.bohr2ang * const.bohr2ang   
    Rij = np.sqrt(Rij2)
    gammaij = getGammaij(rhoi, rhoj, Rij2)    
    scaleij = getScaleij(atnoi,atnoj,Rij)
    return ( Zi * Zj * gammaij +  abs(Zi * Zj * (const.e2 / Rij - gammaij) * scaleij) )

def getBasis(atoms,nbf):
    """
    Returns the basis set in the order of atoms.
    """
    basis = [atoms[0].basis[0]]*nbf
    i=0
    for atom in atoms:
        for bf in atom.basis:
            basis[i] = bf
            i += 1
    return basis 

def getAtomIDs(basis):
    nbf=len(basis)
    tmp = np.zeros(nbf)
    for i in xrange(nbf):
        tmp[i]=basis[i].atom.atid
    return tmp

def getNuclearEnergySerial(nat,atoms,maxdist):
    maxdist2 = maxdist * maxdist * const.ang2bohr * const.ang2bohr
    enuke=0.0
    for i in xrange(nat):
        atomi=atoms[i]
    #    for j in xrange(i+1,nat): # same as below
        for j in xrange(i):
            atomj=atoms[j]
            distij2 = atomi.dist2(atomj) # (in bohr squared) * bohr2ang2
            if distij2 < maxdist2:
                enuke += getEnukeij(atomi, atomj, distij2)           
    return enuke

def getNuclearEnergy(Na,atoms,maxdist):
    Np=MPI.COMM_WORLD.size
    rank=MPI.COMM_WORLD.rank
    Nc=Na/Np
    remainder=Na%Np
    maxdist2 = maxdist * maxdist * const.ang2bohr * const.ang2bohr
    bohr2ang2 = const.bohr2ang * const.bohr2ang
    e2        = const.e2
    enuke = 0
    for i in xrange(Nc):
        atomi = atoms[rank*Nc+i]
        for j in xrange(rank*Nc+i):
            atomj = atoms[j]
            distij2 = atomi.dist2(atomj) # (in bohr squared) * bohr2ang2
            if distij2 < maxdist2:
                enuke += getEnukeij(atomi, atomj, distij2)   
    if remainder - rank > 0:
        atomi = atoms[Na-rank-1]
        for j in xrange(Na-rank-1):
            atomj = atoms[j]
            distij2 = atomi.dist2(atomj) # (in bohr squared) * bohr2ang2
            if distij2 < maxdist2:
                enuke += getEnukeij(atomi, atomj, distij2)   

    return MPI.COMM_WORLD.allreduce(enuke)   
        
def getT(basis,maxdist,maxnnz=[0],bandwidth=[0],comm=PETSc.COMM_SELF):
    """
    Computes a matrix for the two-center two-electron integrals.
    Assumes spherical symmetry, no dependence on basis function, only atom types.
    Parametrized for pairs of atoms. (Two-atom parameters)
    maxnnz: max number of nonzeros per row. If it is given performance might improve    
    This matrix also determines the nonzero structure of the Fock matrix.
    TODO:
    Optimize preallocation based on diagonal and offdiagonal nonzeros.
    Values are indeed based on atoms, not basis functions, so possible to improve performance by nbf/natom.
    Cythonize
    """

    nbf      = len(basis)
    maxdist2 = maxdist * maxdist * const.ang2bohr * const.ang2bohr
    enuke=0.0
    Vdiag = PETSc.Vec().create(comm=comm)
    A        = PETSc.Mat().create(comm=comm)
    A.setType('aij') #'sbaij'
    A.setSizes([nbf,nbf]) 
    if any(maxnnz): 
        A.setPreallocationNNZ(maxnnz) 
    else:
        A.setPreallocationNNZ([nbf,nbf])
    A.setUp()
    A.setOption(A.Option.NEW_NONZERO_ALLOCATION_ERR,True)
    rstart, rend = A.getOwnershipRange()
    localsize = rend-rstart
    Vdiag.setSizes((localsize,nbf))
    Vdiag.setUp()
    nnz = 0
    bohr2ang2 = const.bohr2ang**2.
    e2        = const.e2
    if any(bandwidth):
        if len(bandwidth)==1: bandwidth=np.array([bandwidth]*nbf)
    else:
        bandwidth=np.array([nbf]*nbf)    
    for i in xrange(rstart,rend):
        atomi   = basis[i].atom
        atnoi   = atomi.atno
        rhoi    = atomi.rho
        gammaii = f03[atnoi]
        Vdiag[i] = gammaii
        for j in xrange(i+1,min(i+bandwidth[i],nbf)):
            atomj = basis[j].atom
            if atomi.atid == atomj.atid:
                A[i,j] = gammaii
                nnz += 1
            else:                        
                distij2 = atomi.dist2(atomj) # (in bohr squared) * bohr2ang2
                if distij2 < maxdist2:
                    rhoj    = atomj.rho 
                    distij2 = distij2 * bohr2ang2
                    gammaij = e2 / np.sqrt(distij2 + 0.25 * (rhoi + rhoj)**2.)
                    A[i,j] = gammaij
                    nnz += 1
    A.setDiagonal(Vdiag) 
    A.assemblyBegin()
    nnz =  MPI.COMM_WORLD.allreduce(nnz)  + nbf 
    A.assemblyEnd()
    B = A.duplicate(copy=True)
    B = B + A.transpose() 
    B.setDiagonal(Vdiag) 
    return  nnz, B

def getTold(basis,maxdist,maxnnz=[0],bandwidth=[0],comm=PETSc.COMM_SELF):
    """
    Computes MINDO3 nuclear repulsion energy and gamma matrix
    Nuclear repulsion energy: Based on PyQuante MINDO3 get_enuke(atoms)
    Gamma matrix: Based on PyQuante MINDO3 get_gamma(atomi,atomj)
    "Coulomb repulsion that goes to the proper limit at R=0"
    Corresponds to two-center two-electron integrals
    Assumes spherical symmetry, no dependence on basis function, only atom types.
    Parametrized for pairs of atoms. (Two-atom parameters)
    maxnnz: max number of nonzeros per row. If it is given performance might improve    
    Gamma matrix also determines the nonzero structure of the Fock matrix.
    TODO:
    Values are indeed based on atoms, not basis functions, so possible to improve performance by nbf/natom.
    Better to preallocate based on diagonal and offdiagonal nonzeros.
    Cythonize
    """

    nbf      = len(basis)
    maxdist2 = maxdist * maxdist * const.ang2bohr * const.ang2bohr
    enuke=0.0
    Vdiag = PETSc.Vec().create(comm=comm)
    A        = PETSc.Mat().create(comm=comm)
    A.setType('aij') #'sbaij'
    A.setSizes([nbf,nbf]) 
    if any(maxnnz): 
        A.setPreallocationNNZ(maxnnz) 
    else:
        A.setPreallocationNNZ([nbf,nbf])
    A.setUp()
    A.setOption(A.Option.NEW_NONZERO_ALLOCATION_ERR,True)
    rstart, rend = A.getOwnershipRange()
    localsize = rend-rstart
    Vdiag.setSizes((localsize,nbf))
    Vdiag.setUp()
    nnz = 0
    bohr2ang2 = const.bohr2ang**2.
    e2        = const.e2
    if any(bandwidth):
        if len(bandwidth)==1: bandwidth=np.array([bandwidth]*nbf)
    else:
        bandwidth=np.array([nbf]*nbf)    
    for i in xrange(rstart,rend):
        atomi   = basis[i].atom
        Zi      = atomi.Z
        nbfi    = atomi.nbf 
        atnoi   = atomi.atno
        rhoi    = atomi.rho
        gammaii = PyQuante.MINDO3_Parameters.f03[atnoi]
        Vdiag[i] = gammaii
        for j in xrange(i+1,min(i+bandwidth[i],nbf)):
            atomj = basis[j].atom
            if atomi.atid == atomj.atid:
                A[i,j] = gammaii
                nnz += 1
            else:                        
                distij2 = atomi.dist2(atomj) # (in bohr squared) * bohr2ang2
                if distij2 < maxdist2:
                    Zj      = atomj.Z
                    nbfj    = atomj.nbf 
                    atnoj   = atomj.atno
                    rhoj    = atomj.rho 
                    distij2 = distij2 * bohr2ang2
                    gammaij = e2 / np.sqrt(distij2 + 0.25 * (rhoi + rhoj)**2.)
                    R=np.sqrt(distij2)
                    scale = PyQuante.MINDO3.get_scale(atnoi,atnoj,R)
                    enuke += ( Zi * Zj * gammaij +  abs(Zi * Zj * (e2 / R - gammaij) * scale) ) / ( nbfi * nbfj )
                    A[i,j] = gammaij
                    nnz += 1
    A.setDiagonal(Vdiag) 
    A.assemblyBegin()
    enuke =  MPI.COMM_WORLD.allreduce(enuke)        
    nnz =  MPI.COMM_WORLD.allreduce(nnz)  + nbf      
    A.assemblyEnd()
    B = A.duplicate(copy=True)
    B = B + A.transpose() 
    B.setDiagonal(Vdiag) 
    return  nnz,enuke, B

def getGammaold(basis,maxdist,maxnnz=[0],bandwidth=[0],comm=PETSc.COMM_SELF):
    """
    Computes MINDO3 nuclear repulsion energy and gamma matrix
    Nuclear repulsion energy: Based on PyQuante MINDO3 get_enuke(atoms)
    Gamma matrix: Based on PyQuante MINDO3 get_gamma(atomi,atomj)
    "Coulomb repulsion that goes to the proper limit at R=0"
    Corresponds to two-center two-electron integrals
    Assumes spherical symmetry, no dependence on basis function, only atom types.
    Parametrized for pairs of atoms. (Two-atom parameters)
    maxnnz: max number of nonzeros per row. If it is given performance might improve    
    Gamma matrix also determines the nonzero structure of the Fock matrix.
    TODO:
    Values are indeed based on atoms, not basis functions, so possible to improve performance by nbf/natom.
    Better to preallocate based on diagonal and offdiagonal nonzeros.
    Cythonize
    """
    import constants as const

    nbf      = len(basis)
    maxdist2 = maxdist * maxdist
    enuke=0.0
    A        = PETSc.Mat().create(comm=comm)
    A.setType('aij') #'sbaij'
    A.setSizes([nbf,nbf]) 
    if any(maxnnz): 
        A.setPreallocationNNZ(maxnnz) 
    else:
        A.setPreallocationNNZ(nbf)
    A.setUp()
    A.setOption(A.Option.NEW_NONZERO_ALLOCATION_ERR,False)
    rstart, rend = A.getOwnershipRange()
    nnz = 0
    mindist = 1.E-5 # just to see same atom or not
    if any(bandwidth):
        if len(bandwidth)==1: bandwidth=np.array([bandwidth]*nbf)
        for i in xrange(rstart,rend):
            atomi=basis[i].atom
            gammaii= PyQuante.MINDO3_Parameters.f03[atomi.atno]
            A[i,i] = gammaii
            for j in xrange(i+1,min(i+bandwidth[i],nbf)):
                atomj = basis[j].atom
                if atomi == atomj:
                    A[i,j] = gammaii
                    A[j,i] = gammaii
                    nnz += 1
                else:                        
                    distij2 = atomi.dist2(atomj) * const.bohr2ang**2.
                    if distij2 < maxdist2: 
                        gammaij=const.e2/np.sqrt(distij2+0.25*(atomi.rho+atomj.rho)**2.)
                        R=np.sqrt(distij2)
                        scale = PyQuante.MINDO3.get_scale(atomi.atno,atomj.atno,R)
                        enuke += ( atomi.Z*atomj.Z*gammaij +  abs(atomi.Z*atomj.Z*(const.e2/R-gammaij)*scale) ) / ( atomi.nbf *atomj.nbf )
                        A[i,j] = gammaij
                        A[j,i] = gammaij
                        nnz += 1
    else:
        for i in xrange(rstart,rend):
            atomi=basis[i].atom
            gammaii= PyQuante.MINDO3_Parameters.f03[atomi.atno]
            A[i,i] = gammaii
            for j in xrange(i+1,nbf):
                atomj = basis[j].atom
                if atomi == atomj:
                    A[i,j] = gammaii
                    A[j,i] = gammaii
                    nnz += 1
                else:                        
                    distij2 = atomi.dist2(atomj) * const.bohr2ang**2.
                    if distij2 < maxdist2: 
                        gammaij=const.e2/np.sqrt(distij2+0.25*(atomi.rho+atomj.rho)**2.)
                        R=np.sqrt(distij2)
                        scale = PyQuante.MINDO3.get_scale(atomi.atno,atomj.atno,R)
                        enuke += ( atomi.Z*atomj.Z*gammaij +  abs(atomi.Z*atomj.Z*(const.e2/R-gammaij)*scale) ) / ( atomi.nbf *atomj.nbf )
                        A[i,j] = gammaij
                        A[j,i] = gammaij
                        nnz += 1
    A.assemblyBegin()
    enuke =  MPI.COMM_WORLD.allreduce(enuke)        
    nnz =  MPI.COMM_WORLD.allreduce(nnz)  + nbf      
    A.assemblyEnd()
    return  nnz,enuke, A
       
def getF0(atoms,basis,T):
    """
    Form the zero-iteration (density matrix independent) Fock matrix
    Ref 1: DOI:10.1021/ja00839a001
    Ref 2: ISBN:089573754X
    Ref 3: DOI:10.1002/wcms.1141
    TODO: 
    Cythonize
    """
    A = T.duplicate()
    A.setUp()
    rstart, rend = A.getOwnershipRange()
    for i in xrange(rstart,rend):
        basisi=basis[i]
        typei=basisi.type
        ipi=basisi.ip
        atomi=basisi.atom
        cols,vals = T.getRow(i)
        tmp = basisi.u # Ref1, Ref2
        k = 0
        for j in cols:
            basisj=basis[j]
            atomj=basisj.atom
            if atomj != atomi:
 #               tmp -= T[i,j] * atomj.Z / len(atomj.basis) # Ref1, Ref2 adopted sum to be over orbitals rather than atoms
                tmp -= vals[k] * atomj.Z / atomj.nbf # Ref1, Ref2 adopted sum to be over orbitals rather than atoms
          #  if i != j: # According to Ref1, Ref2 and Ref3  atoms should be different, but PyQuante implementation ignores that, however this does not change any results since the overlap of different orbitals on the same atom is very small. (orthogonal) 
           #     if typei == basisj.type:
                betaij = getBeta0ij(atomi.atno,atomj.atno)
                Sij = basisi.cgbf.overlap(basisj.cgbf)
                IPij = ipi + basisj.ip
                tmp2 =  betaij * IPij * Sij     # Ref1, Ref2 
                A[i,j] = tmp2
            k = k + 1    
        A[i,i] = tmp        
    A.assemble()
    return A

def getD0(basis,guess=0,T=None,comm=PETSc.COMM_SELF):
    """
    Returns the guess (initial) density matrix.
    guess = 0 :
        A very simple guess is used which a diagonal matrix containing atomic charge divided by number of basis functions for the atom.
    guess = 1 :
        Same diagoanl with guess 0 but offdiagonal values are set to d. Duplicates T.
    guess = 2 :
        TODO:
        Same diagoanl with guess 0 but offdiagonal values are set to random values. Duplicates T.   
    """
    nbf=len(basis)

    if guess==0: 
        A= PETSc.Mat().create(comm=comm)
        A.setType('aij') 
        A.setSizes([nbf,nbf])        
        A.setPreallocationNNZ(1) 
        A.setUp()
        rstart, rend = A.getOwnershipRange() 
        for i in xrange(rstart,rend):
            atomi=basis[i].atom
            if atomi.atno == 1: A[i,i] = atomi.Z/1.
            else:               A[i,i] = atomi.Z/4.
        A.assemble()    
    elif guess==1:
        d=0.
        if T:
            A=T.duplicate()
            rstart, rend = A.getOwnershipRange() 
            for i in xrange(rstart,rend):
                atomi=basis[i].atom
                cols,vals = T.getRow(i) 
                for j in cols:
                    if i==j:
                        if atomi.atno == 1: A[i,i] = atomi.Z/1.
                        else:               A[i,i] = atomi.Z/4.
                    else:    
                        A[i,j] = d * T[i,j]
            A.assemble()
        else:
            Print("You need to give a template matrix for guess type {0}".format(guess))            
    return  A 

def getG(basis, comm=PETSc.COMM_SELF, T=None):
    """
    Returns the matrix for one-electron Coulomb term, (mu mu | nu nu) where mu and nu orbitals are centered on the same atom.
    Block diagonal matrix with 1x1 (Hydrogens) or 4x4 blocks.
    If T is given, assumes the nonzero pattern of T.
    """
    nbf             = len(basis)
    if T:
        A = T.duplicate()
    else:        
        maxnnzperrow    = 4
        A               = PETSc.Mat().create(comm=comm)
        A.setType('aij') #'sbaij'
        A.setSizes([nbf,nbf]) 
        A.setPreallocationNNZ(maxnnzperrow) 
    k=0
    A.setUp()
    rstart, rend = A.getOwnershipRange()
    for i in xrange(rstart,rend):
        basisi  = basis[i]
        atomi   = basisi.atom
        nbfi    = atomi.nbf
        A[i,i] = getGij(basisi,basisi)
        if atomi.atno > 1:
            minj = max(0,i-nbfi)
            maxj = min(nbf,i+nbfi) 
            for j in xrange(minj,maxj):
                basisj = basis[j]
                atomj   = basisj.atom
                if atomi.atid == atomj.atid:
                    A[i,j] = getGij(basisi,basisj)
    A.assemble()
    return A

def getH(basis, comm=PETSc.COMM_SELF, T=None):
    """
    Returns the matrix for one-electron exchange term, (mu nu | mu nu) where mu and nu orbitals are centered on the same atom. 
    Block diagonal matrix with 1x1 (Hydrogens) or 4x4 blocks.
    If T is given, assumes the nonzero pattern of T.
    """
    nbf             = len(basis)
    if T:
        A = T.duplicate()
    else:        
        maxnnzperrow    = 4
        A               = PETSc.Mat().create(comm=comm)
        A.setType('aij') #'sbaij'
        A.setSizes([nbf,nbf]) 
        A.setPreallocationNNZ(maxnnzperrow) 
    k=0
    A.setUp()
    rstart, rend = A.getOwnershipRange()
    for i in xrange(rstart,rend):
        basisi  = basis[i]
        atomi   = basisi.atom
        nbfi    = atomi.nbf
        A[i,i] = getGij(basisi,basisi)
        if atomi.atno > 1:
            minj = max(0,i-nbfi)
            maxj = min(nbf,i+nbfi) 
            for j in xrange(minj,maxj):
                basisj = basis[j]
                atomj   = basisj.atom
                if atomi.atid == atomj.atid and i != j:
                    A[i,j] = getHij(basisi,basisj)
    A.assemble()
    return A

def getFD( basis, D, T):
    """
    Density matrix dependent terms of the Fock matrix
    """
    diagD = pt.convert2SeqVec(D.getDiagonal()) 
    A = T.duplicate()
    A.setUp()
    rstart, rend = A.getOwnershipRange()
    for i in xrange(rstart,rend):
        basisi=basis[i]
        atomi=basisi.atom
        colsT,valsT = T.getRow(i)
        colsD,valsD = D.getRow(i)
        tmpii = 0.5 * diagD[i] * PyQuante.MINDO3.get_g(basisi,basisi) # Since g[i,i]=h[i,i] 
        k=0
        for j in colsT:
            basisj=basis[j]
            atomj=basisj.atom
            if i != j:
                tmpij=0
                Djj=diagD[j] # D[j,j]
                Dij=valsD[k]
                Tij=valsT[k]
                if atomj == atomi:
                    gij=PyQuante.MINDO3.get_g(basisi,basisj)
                    hij=PyQuante.MINDO3.get_h(basisi,basisj)
                #    tmpii += Djj * gij - 0.5 * Dij * hij # as given in Eq 2 in Ref1 and page 54 in Ref2
                    tmpii += Djj * ( gij - 0.5 * hij )#Ref1 and PyQuante, In Ref2, Ref3, when i==j, g=h
                 #   tmpij  = -0.5 * Dij * PyQuante.MINDO3.get_h(basisi,basisj) # Eq 3 in Ref1 but not in Ref2 and PyQuante
                    tmpij =  0.5 * Dij * ( 3. * hij - gij ) #Ref3, PyQuante, I think this term is an improvement to MINDO3 (it is in MNDO) so not found in Ref1 and Ref2  
                else:
                    tmpii += Tij * Djj     # Ref1, Ref2, Ref3
                    tmpij = -0.5 * Tij * Dij   # Ref1, Ref2, Ref3  
                A[i,j] = tmpij
            k=k+1    
        A[i,i] = tmpii        
    A.assemble()
    return A

def getF(atomIDs, D, F0, T, G, H):
    """
    Density matrix dependent terms of the Fock matrix
    """
    diagG = G.getDiagonal()
    diagD = pt.convert2SeqVec(D.getDiagonal()) 
    A     = T.duplicate()
    A.setUp()
    rstart, rend = A.getOwnershipRange()
    for i in xrange(rstart,rend):
        atomi=atomIDs[i]
        colsD,valsD = D.getRow(i)
        colsG,valsG = G.getRow(i)
        colsH,valsH = H.getRow(i)
        colsT,valsT = T.getRow(i)
        tmpii       = 0.5 * diagD[i] * diagG[i] # Since g[i,i]=h[i,i]
        k=0
        idxG=0
        for j in colsT:
            atomj=atomIDs[j]
            if i != j:
                tmpij = 0
                Djj   = diagD[j] # D[j,j]
                if len(valsD)>1:
                    Dij    = valsD[k]
                else:
                    Dij    = 0
                Tij   = valsT[k]
                if atomj  == atomi:
                    idxG   = np.where(colsG==j)
                    Gij    = valsG[idxG]
                    Hij    = valsH[idxG]
                    tmpii += Djj * ( Gij - 0.5 * Hij ) # Ref1 and PyQuante, In Ref2, Ref3, when i==j, g=h
                    tmpij  =  0.5 * Dij * ( 3. * Hij - Gij ) # Ref3, PyQuante, I think this term is an improvement to MINDO3 (it is in MNDO) so not found in Ref1 and Ref2  
                else:
                    tmpii += Tij * Djj     # Ref1, Ref2, Ref3
                    tmpij = -0.5 * Tij * Dij   # Ref1, Ref2, Ref3  
                A[i,j] = tmpij
            k=k+1    
        A[i,i] = tmpii        
    A.assemble()
    return A

def getFDseq(atomIDs, Dseq, F0, T, G, H):
    """
    Density matrix dependent terms of the Fock matrix
    """
    diagG = G.getDiagonal()
    diagD = pt.convert2SeqVec(D.getDiagonal()) 
    A     = T.duplicate()
    A.setUp()
    rstart, rend = A.getOwnershipRange()
    for i in xrange(rstart,rend):
        atomi=atomIDs[i]
        colsD,valsD = D.getRow(i)
        colsG,valsG = G.getRow(i)
        colsH,valsH = H.getRow(i)
        colsT,valsT = T.getRow(i)
        tmpii       = 0.5 * diagD[i] * diagG[i] # Since g[i,i]=h[i,i]
        k=0
        idxG=0
        for j in colsT:
            atomj=atomIDs[j]
            if i != j:
                tmpij = 0
                Djj   = diagD[j] # D[j,j]
                if len(valsD)>1:
                    Dij    = valsD[k]
                else:
                    Dij    = 0
                Tij   = valsT[k]
                if atomj  == atomi:
                    idxG   = np.where(colsG==j)
                    Gij    = valsG[idxG]
                    Hij    = valsH[idxG]
                    tmpii += Djj * ( Gij - 0.5 * Hij ) # Ref1 and PyQuante, In Ref2, Ref3, when i==j, g=h
                    tmpij  =  0.5 * Dij * ( 3. * Hij - Gij ) # Ref3, PyQuante, I think this term is an improvement to MINDO3 (it is in MNDO) so not found in Ref1 and Ref2  
                else:
                    tmpii += Tij * Djj     # Ref1, Ref2, Ref3
                    tmpij = -0.5 * Tij * Dij   # Ref1, Ref2, Ref3  
                A[i,j] = tmpij
            k=k+1    
        A[i,i] = tmpii        
    A.assemble()
    return A

def scf(opts,nocc,atomIDs,D,F0,T,G,H,stage):
    """
    """
    maxiter     = opts.getInt('maxiter', 30)
    guess       = opts.getInt('guess', 0)
    scfthresh   = opts.getReal('scfthresh',1.e-5)
    interval    = [opts.getReal('a',-500.) , opts.getReal('b', 500.)]
    staticsubint= opts.getInt('staticsubint',0)
    usesips     = opts.getBool('sips',False)
    
    Eel       = 0.
    gap       = 0.
    converged = False 
    eps       = None   
    subint    = [0]
    Print("{0:*^60s}".format("SELF-CONSISTENT-FIELD ITERATIONS"))
    Print("SCF threshold: {0:5.3e}".format(scfthresh))
    Print("Maximum number of SCF iterations: {0}".format(maxiter))
    if staticsubint == 0: 
        Print("Fixed subintervals will be used")
    elif staticsubint == 1: 
        Print("Subintervals will be adjusted at each iteration with fixed interval")
    elif staticsubint == 2: 
        Print("Subintervals will be adjusted at each iteration")
    else:
        Print("Not available")   
    if usesips:
        try:
            import SIPs.sips as sips
        except:
            Print("SIPs not found")
            usesips = False
    for k in xrange(1,maxiter+1):
        Print("{0:*^60s}".format("Iteration "+str(k)))
        t0 = pt.getWallTime()
        stage = pt.getStage(stagename='F',oldstage=stage)
        F    = getF(atomIDs, D, F0, T, G, H)
        F    = F0 + F
        Eold = Eel
        stage = pt.getStage(stagename='Trace',oldstage=stage)
        if k==1:
            if guess==0:
                Eel  = 0.5 * pt.getTraceDiagProduct(D,F0+F)
            else:
                Eel  = 0.5 * pt.getTraceProductAIJ(D, F0+F)
            stage = pt.getStage(stagename='SetupEPS',oldstage=stage)    
            eps = st.setupEPS(F, B=None,interval=interval)  
            stage = pt.getStage(stagename='SolveEPS',oldstage=stage)
            eps, nconv, eigarray = st.solveEPS(eps,returnoption=1,nocc=nocc)
        else:
            Eel  = 0.5 * pt.getTraceProductAIJ(D, F0+F)
            stage = pt.getStage(stagename='UpdateEPS',oldstage=stage)            
            subint =interval
            if staticsubint==1:
                nsubint=st.getNumberOfSubIntervals(eps)
                subint = st.getSubIntervals(eigarray[0:nocc],nsubint,interval=interval) 
            elif staticsubint==2:
                nsubint=st.getNumberOfSubIntervals(eps)
                subint = st.getSubIntervals(eigarray[0:nocc],nsubint)
            eps = st.updateEPS(eps,F,subintervals=subint)
            stage = pt.getStage(stagename='SolveEPS',oldstage=stage)
            eps, nconv, eigarray = st.solveEPS(eps,returnoption=1,nocc=nocc)         
        if (len(eigarray)>nocc+1):
            gap = eigarray[nocc] - eigarray[nocc-1]              
            Print("Gap            = {0:20.10f} kcal/mol = {1:20.10f} ev = {2:20.10f} Hartree".format(gap*const.ev2kcal,gap,gap*const.ev2hartree))  
        Print("Eel            = {0:20.10f} kcal/mol = {1:20.10f} ev = {2:20.10f} Hartree".format(Eel*const.ev2kcal,Eel,Eel*const.ev2hartree))  
        stage = pt.getStage(stagename='Density', oldstage=stage)
        nden = nocc
        if nconv < nocc: 
            nden = nconv
        if usesips:
            D = sips.getDensityMat(eps,0,nden)
        else:    
            D = st.getDensityMatrix(eps,T, nden)
        sizecommD = D.getComm().Get_size()    
        if sizecommD > 1 and sizecommD < F.getComm().Get_size():   
            D = pt.getSeqMat(D)
        pt.getWallTime(t0,str='Iteration completed in')
        if abs(Eel-Eold) < scfthresh and nconv >= nocc:
            Print("Converged at iteration {0}".format(k))
            converged = True
            return converged, Eel, gap
    return converged, Eel, gap

def getEnergy(qmol,opts):

    stage       = pt.getStage(stagename='MINDO3')
    t0          = pt.getWallTime()
    maxdist     = opts.getReal('maxdist', 1.e6)
    solve       = opts.getInt('solve', 0)
    maxnnz      = [opts.getInt('maxnnz', 0)]
    guess       = opts.getInt('guess', 0)
    bandwidth   = [opts.getInt('bw', 0)]
    spfilter    = opts.getReal('spfilter',0.)
    debug       = opts.getBool('debug',False)
    checknnz    = opts.getBool('checknnz',False) 
    pyquante    = opts.getBool('pyquante',False) #TODO get Pyquante energy for testing
    Print("Distance cutoff: {0:5.3f}".format(maxdist))
    qmol  = initialize(qmol)
    pt.getWallTime(t0,str="MINDO3 initialized in")
#    if pyquante:
#        Epyquante = PyQuante.MINDO3.scf(qmol)
    atoms   = qmol.atoms
    Eref  = getEref(qmol)
    nbf   = getNbasis(qmol)    
    nel   = getNelectrons(atoms)
    nocc  = nel/2
    basis = getBasis(qmol, nbf)
    atomIDs = getAtomIDs(basis)
    matcomm = PETSc.COMM_WORLD
    Print("Number of basis functions  : {0} = Matrix size".format(nbf))
    Print("Number of valance electrons: {0}".format(nel))
    Print("Number of occupied orbitals: {0} = Number of required eigenvalues".format(nocc))
    if checknnz:
        stage = pt.getStage(stagename='Nonzero info', oldstage=stage)
        maxnnz,bandwidth = pt.getNnzInfo(basis, maxdist)
    stage = pt.getStage(stagename='T', oldstage=stage)
    nnz, T            = getT(basis, maxdist,maxnnz=maxnnz, bandwidth=bandwidth, comm=matcomm)
    Enuke             = getNuclearEnergy(len(atoms), atoms, maxdist)
    dennnz = nnz / (nbf*(nbf+1)/2.0)  * 100.
    Print("Nonzero density percent : {0}".format(dennnz))
    Print("Eref           = {0:20.10f} kcal/mol = {1:20.10f} ev = {2:20.10f} Hartree".format(Eref, Eref*const.kcal2ev, Eref*const.kcal2hartree))    
    Print("Enuc           = {0:20.10f} kcal/mol = {1:20.10f} ev = {2:20.10f} Hartree".format(Enuke*const.ev2kcal,Enuke,Enuke*const.ev2hartree))
    stage = pt.getStage(stagename='F0', oldstage=stage)
    F0    = getF0(atoms, basis, T)
    stage = pt.getStage(stagename='D0', oldstage=stage)
    D0     = getD0(basis,guess=guess,T=T,comm=matcomm)
    stage = pt.getStage(stagename='G', oldstage=stage)
    G     = getG(basis,comm=matcomm)    
    stage = pt.getStage(stagename='H', oldstage=stage)
    H     = getH(basis,T=G)
    pt.getWallTime(t0,str="Pre-SCF steps finished in")
    t0          = pt.getWallTime()
    converged, Eelec, gap = scf(opts,nocc,atomIDs,D0,F0,T,G,H,stage)
    pt.getWallTime(t0,str="SCF finished in")
    Etot   = Eelec + Enuke
    Efinal = Etot*const.ev2kcal+Eref
    Print("Enuc             = {0:20.10f} kcal/mol = {1:20.10f} ev = {2:20.10f} Hartree".format(Enuke*const.ev2kcal,Enuke,Enuke*const.ev2hartree))
    Print("Eref             = {0:20.10f} kcal/mol = {1:20.10f} ev = {2:20.10f} Hartree".format(Eref, Eref*const.kcal2ev, Eref*const.kcal2hartree))
    Print("Eelec            = {0:20.10f} kcal/mol = {1:20.10f} ev = {2:20.10f} Hartree".format(Eelec*const.ev2kcal,Eelec,Eelec*const.ev2hartree))
    Print("Eelec+Enuc       = {0:20.10f} kcal/mol = {1:20.10f} ev = {2:20.10f} Hartree".format(Etot*const.ev2kcal,Etot,Etot*const.ev2hartree))
    Print("Eelec+Enuc+Eref  = {0:20.10f} kcal/mol = {1:20.10f} ev = {2:20.10f} Hartree".format(Efinal, Efinal*const.kcal2ev,Efinal*const.kcal2hartree))
    Print("Gap              = {0:20.10f} kcal/mol = {1:20.10f} ev = {2:20.10f} Hartree".format(gap*const.ev2kcal,gap,gap*const.ev2hartree))
    return Efinal

def main(qmol,opts):
    import PyQuante.MINDO3_Parameters
    import PyQuante.MINDO3
    import constants as const
 
    stage       = pt.getStage(stagename='Input')
    maxdist     = opts.getReal('maxdist', 1.e6)
    maxiter     = opts.getInt('maxiter', 30)
    analysis    = opts.getInt('analysis', 0)
    solve       = opts.getInt('solve', 0)
    maxnnz      = [opts.getInt('maxnnz', 0)]
    guess       = opts.getInt('guess', 0)
    bandwidth   = [opts.getInt('bw', 0)]
    sort        = opts.getInt('sort', 0)     
    scfthresh   = opts.getReal('scfthresh',1.e-5)
    spfilter    = opts.getReal('spfilter',0.)
    staticsubint= opts.getBool('staticsubint',False)
    debug       = opts.getBool('debug',False)
    usesips     = opts.getBool('sips',False)
    Print("Distance cutoff: {0:5.3f}".format(maxdist))
    Print("SCF threshold: {0:5.3e}".format(scfthresh))
    Print("Maximum number of SCF iterations: {0}".format(maxiter))
    if staticsubint == 2:
        Print("Subintervals will be updated after the first iteration")
    if usesips:
        try:
            import SIPs.sips as sips
        except:
            Print("sips not found")
            usesips = False
    stage = pt.getStage(stagename='Initialize')
    qmol  = PyQuante.MINDO3.initialize(qmol)
    atoms = qmol.atoms
    stage = pt.getStage(stagename='Enuclear', oldstage=stage)
    Enuke = PyQuante.MINDO3.get_enuke(qmol)
    Print("Enuc           = {0:20.10f} kcal/mol = {1:20.10f} ev = {2:20.10f} Hartree".format(Enuke*const.ev2kcal,Enuke,Enuke*const.ev2hartree))
    stage = pt.getStage(stagename='Ereference', oldstage=stage)
    Eref  = PyQuante.MINDO3.get_reference_energy(qmol)
    Print("Eref           = {0:20.10f} kcal/mol = {1:20.10f} ev = {2:20.10f} Hartree".format(Eref, Eref*const.kcal2ev, Eref*const.kcal2hartree))
    stage = pt.getStage(stagename='Basisset', oldstage=stage)
    nbf   = PyQuante.MINDO3.get_nbf(qmol)    
    nel   = PyQuante.MINDO3.get_nel(atoms)
    nocc  = nel/2
    basis = getBasis(qmol, nbf)
    atomIDs = getAtomIDs(basis)
    matcomm=PETSc.COMM_WORLD
    Print("Number of basis functions  : {0} = Matrix size".format(nbf))
    Print("Number of valance electrons: {0}".format(nel))
    Print("Number of occupied orbitals: {0} = Number of required eigenvalues".format(nocc))
    if not (all(maxnnz) or all(bandwidth)):
        stage = pt.getStage(stagename='getNnz', oldstage=stage)
        nnzarray,bwarray = pt.getNnzInfo(basis, maxdist)
        maxnnz = max(nnzarray)
        maxbw = max(bwarray)
        sumnnz = sum(nnzarray)
        avgnnz = sumnnz / float(nbf)
        dennnz = sumnnz / (nbf*(nbf+1)/2.0)  * 100.
        Print("Maximum nonzeros per row: {0}".format(maxnnz))
        Print("Maximum bandwidth       : {0}".format(maxbw))
        Print("Average nonzeros per row: {0}".format(avgnnz))
        Print("Total number of nonzeros: {0}".format(sumnnz))
        Print("Nonzero density percent : {0}".format(dennnz))
        stage = pt.getStage(stagename='getGamma', oldstage=stage)
        nnz,Enuke, T     = getGamma(basis, maxdist,maxnnz=nnzarray, bandwidth=bwarray, comm=matcomm)
        dennnz = nnz / (nbf*(nbf+1)/2.0)  * 100.
        Print("Nonzero density percent : {0}".format(dennnz))
        Print("Enuc2          = {0:20.10f} kcal/mol = {1:20.10f} ev = {2:20.10f} Hartree".format(Enuke*const.ev2kcal,Enuke,Enuke*const.ev2hartree))
    else:
        stage = pt.getStage(stagename='getGamma', oldstage=stage)
        nnz, Enuke, T     = getGamma(basis, maxdist,maxnnz=maxnnz, bandwidth=bandwidth, comm=matcomm)
        dennnz = nnz / (nbf*(nbf+1)/2.0)  * 100.
        Print("Nonzero density percent : {0}".format(dennnz))
        Print("Enuc3          = {0:20.10f} kcal/mol = {1:20.10f} ev = {2:20.10f} Hartree".format(Enuke*const.ev2kcal,Enuke,Enuke*const.ev2hartree))
    stage = pt.getStage(stagename='F0', oldstage=stage)
    F0    = getF0(atoms, basis, T)
    stage = pt.getStage(stagename='D0', oldstage=stage)
    D     = getD0(basis,guess=1,T=T,comm=matcomm)
    stage = pt.getStage(stagename='Ddiag', oldstage=stage)
    Ddiag = pt.convert2SeqVec(D.getDiagonal()) 
        
    Eel   = 0.    
    Print("{0:*^60s}".format("SELF-CONSISTENT-FIELD ITERATIONS"))
    for iter in xrange(1,maxiter):
        Print("{0:*^60s}".format("Iteration "+str(iter)))
        stage = pt.getStage(stagename='FD', oldstage=stage)
        FD    = getFD(basis, D, T)
        F     = F0 + FD
        Eold = Eel

        if solve > 0: 
            stage = pt.getStage(stagename='Trace', oldstage=stage)
            Eel   = 0.5*pt.getTraceProductAIJ(D,F0+F)
            Print("Eel            = {0:20.10f} kcal/mol = {1:20.10f} ev = {2:20.10f} Hartree".format(Eel*const.ev2kcal,Eel,Eel*const.ev2hartree))     
            t0 = pt.getWallTime()
            stage = pt.getStage(stagename='Solve', oldstage=stage)
            if staticsubint or iter<2:
                if solve==2: 
                    eps = st.solveEPS(F,returnoption=-1,nocc=nocc)
                else:  
                    eps, nconv, eigarray = st.solveEPS(F,returnoption=1,nocc=nocc)  
            else:
                if solve==2: 
                    eps = st.solveEPS(F,returnoption=-1,nocc=nocc)  
                else:                  
                    nsubint=st.getNumberOfSubIntervals(eps)
                    subint = st.getSubIntervals(eigarray[0:nocc],nsubint) 
                    eps, nconv, eigarray = st.solveEPS(F,subintervals=subint,returnoption=1,nocc=nocc)   
            pt.getWallTime(t0)    
            stage = pt.getStage(stagename='Density', oldstage=stage)
            t0 = pt.getWallTime()
            if usesips:
                if solve==2: 
                    D = sips.solveDensityMat(eps,0,nocc)
                else: 
                    D = sips.getDensityMat(eps,0,nocc)
            else:    
                D = st.getDensityMatrix(eps,T, nocc)
            t = pt.getWallTime(t0)
        
        if abs(Eel-Eold) < scfthresh:
            Print("Converged at iteration %i" % (iter+1))
            break
        Ddiag=pt.convert2SeqVec(D.getDiagonal()) 


    Etot   = Eel+Enuke
    Efinal = Etot*const.ev2kcal+Eref
    Print("Enuc           = {0:20.10f} kcal/mol = {1:20.10f} ev = {2:20.10f} Hartree".format(Enuke*const.ev2kcal,Enuke,Enuke*const.ev2hartree))
    Print("Eref           = {0:20.10f} kcal/mol = {1:20.10f} ev = {2:20.10f} Hartree".format(Eref, Eref*const.kcal2ev, Eref*const.kcal2hartree))
    Print("Eel            = {0:20.10f} kcal/mol = {1:20.10f} ev = {2:20.10f} Hartree".format(Eel*const.ev2kcal,Eel,Eel*const.ev2hartree))
    Print("Eel+Enuke      = {0:20.10f} kcal/mol = {1:20.10f} ev = {2:20.10f} Hartree".format(Etot*const.ev2kcal,Etot,Etot*const.ev2hartree))
    Print("Eel+Enuke+Eref = {0:20.10f} kcal/mol = {1:20.10f} ev = {2:20.10f} Hartree".format(Efinal, Efinal*const.kcal2ev,Efinal*const.kcal2hartree))
    if debug:
        stage = PETSc.Log.Stage('PyQuante');
        Print('%f' % (PyQuante.MINDO3.scf(qmol)))
    return

       