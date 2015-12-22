import numpy as np
import petsctools as pt
import slepctools as st
import unittools as ut

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

def writeEnergies(en,unit='', enstr=''):
    Ekcal, Eev, Ehart = ut.convertEnergy(en, unit)
    pt.write("{0: <16s} = {1:20.10f} kcal/mol = {2:20.10f} ev = {3:20.10f} Hartree".format(enstr,Ekcal, Eev, Ehart))
    return 0

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
    return ut.e2 / np.sqrt(rij2 + 0.25 * (rhoi + rhoj)**2.)

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
    return hfat-eat * ut.ev2kcal

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
    Rij2 = Rij2 * ut.bohr2ang * ut.bohr2ang   
    Rij = np.sqrt(Rij2)
    gammaij = getGammaij(rhoi, rhoj, Rij2)    
    scaleij = getScaleij(atnoi,atnoj,Rij)
    return ( Zi * Zj * gammaij +  abs(Zi * Zj * (ut.e2 / Rij - gammaij) * scaleij) )

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
    maxdist2 = maxdist * maxdist * ut.ang2bohr * ut.ang2bohr
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

def getNuclearEnergy(comm,Na,atoms,maxdist):
    Np=comm.size
    rank=comm.rank
    Nc=Na/Np
    remainder=Na%Np
    maxdist2 = maxdist * maxdist * ut.ang2bohr * ut.ang2bohr
    bohr2ang2 = ut.bohr2ang * ut.bohr2ang
    e2        = ut.e2
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

    return comm.allreduce(enuke)   

def getLocalNnzPerRowSym(basis,rstart,rend,maxdist2):
    """
    Returns an array containing local number of nonzeros per row based on distance between atoms.
    Size depends on the number of rows per process.
    Locality is based on a temporarily created AIJ matrix. Is there a better way?
    I am not  sure if this is needed, I could do this for all processes since I only need to create a vector of size nbf
    """
    nbf=len(basis)
    dnnz=np.ones(rend-rstart,dtype='int32')
    onnz=np.zeros(rend-rstart,dtype='int32')
    
    for i in xrange(rstart,rend):
        atomi=basis[i].atom
        for j in xrange(i+1,rend):
            distij2=atomi.dist2(basis[j].atom)
            if distij2 < maxdist2: 
                dnnz[i] += 1
        for j in xrange(rend,nbf):
            distij2=atomi.dist2(basis[j].atom)
            if distij2 < maxdist2: 
                onnz[i] += 1
        pt.write(dnnz[i],onnz[i])        
    return dnnz,onnz

def getLocalNnzPerRow(basis,rstart,rend,maxdist2):
    """
    Returns three arrays that contains: 
    dnnz: local numbers of nonzseros per row in diagonal blocks (square) 
    onnz: local numbers of nonzeros per row in off-diagonal blocks (rectangular)
    jmax: max column index that contains a nonzero.
    Nonzeros are based on distance between atoms.
    TODO: Exploit symmetry, not sure how to do that.
    """
   # t0 = pt.getWallTime()
    nbf=len(basis)
    localsize=rend-rstart
    dnnz=np.zeros(localsize,dtype='int32')
    onnz=np.zeros(localsize,dtype='int32')
    jmax=np.zeros(localsize,dtype='int32')
    k=0
    for i in xrange(rstart,rend):
        atomi=basis[i].atom
        for j in xrange(nbf):
            distij2=atomi.dist2(basis[j].atom)
            if distij2 < maxdist2:
                if j >= rstart and j < rend: 
                    dnnz[k] += 1
                else:
                    onnz[k] += 1
                if j > jmax[k]:
                    jmax[k] = j 
        k += 1 
   # pt.getWallTime(t0=t0,str='Local nnz')                
    return dnnz, onnz, jmax

def getT(comm,basis,maxdist,preallocate=False):
    """
    Computes a matrix for the two-center two-electron integrals.
    Assumes spherical symmetry, no dependence on basis function, only atom types.
    Parametrized for pairs of atoms. (Two-atom parameters)
    maxnnz: max number of nonzeros per row. If it is given performance might improve    
    This matrix also determines the nonzero structure of the Fock matrix.
    Nuclear repulsion energy is also computed.
    TODO:
    Values are indeed based on atoms, not basis functions, so possible to improve performance by nbf/natom.
    Cythonize
    """
    t = pt.getWallTime()
    nbf      = len(basis)
    nnz = nbf*nbf
    maxdist2 = maxdist * maxdist * ut.ang2bohr * ut.ang2bohr
    Vdiag = pt.createVec(comm=comm)
    A        = pt.createMat(comm=comm)
    A.setType('aij') #'sbaij'
    A.setSizes([nbf,nbf]) 
    A.setUp()
    A.setOption(A.Option.NEW_NONZERO_ALLOCATION_ERR,True)
    rstart, rend = A.getOwnershipRange()
    localsize = rend - rstart
    Vdiag.setSizes((localsize,nbf))
    Vdiag.setUp()
    bohr2ang2 = ut.bohr2ang**2.
    e2        = ut.e2
    k = 0   
    enuke=0.0
    t = pt.getWallTime(t0=t,str='Initial')
    if preallocate:
        dnnz,onnz,jmax = getLocalNnzPerRow(basis,rstart,rend,maxdist2)
        nnz = sum(dnnz) + sum(onnz)
        t = pt.getWallTime(t0=t,str='Local nnz')
        A.setPreallocationNNZ((dnnz,onnz)) 
    else:
        A.setPreallocationNNZ([nbf,nbf])
        jmax = np.array([localsize]*nbf)
    t = pt.getWallTime(t0=t,str='Prealloc')
    for i in xrange(rstart,rend):
        atomi   = basis[i].atom
        atnoi   = atomi.atno
        rhoi    = atomi.rho
        gammaii = f03[atnoi]
        Vdiag[i] = gammaii
    #    for j in xrange(i+1,min(i+bandwidth[i],nbf)):
        for j in xrange(i+1,jmax[k]+1):
            atomj = basis[j].atom
            if atomi.atid == atomj.atid:
                A[i,j] = gammaii
            else:                        
                distij2 = atomi.dist2(atomj) # (in bohr squared) * bohr2ang2
                if distij2 < maxdist2:
                    rhoj    = atomj.rho 
                    distij2 = distij2 * bohr2ang2
                    gammaij = e2 / np.sqrt(distij2 + 0.25 * (rhoi + rhoj)**2.)
                    R=np.sqrt(distij2)
                    A[i,j] = gammaij
                    atnoj   = atomj.atno
                    enuke += ( atomi.Z*atomj.Z*gammaij +  abs(atomi.Z*atomj.Z*(ut.e2/R-gammaij)*getScaleij(atnoi, atnoj, R)) ) / ( atomi.nbf * atomj.nbf )
        k += 1            
    t = pt.getWallTime(t0=t,str='For loop')
    A.setDiagonal(Vdiag) 
    A.assemblyBegin()
    enuke =  comm.allreduce(enuke)        
    if preallocate:
        nnz =  comm.allreduce(nnz) 
    A.assemblyEnd()
    B = A.duplicate(copy=True)
    B = B + A.transpose() 
    B.setDiagonal(Vdiag) 
    t = pt.getWallTime(t0=t,str='Assembly')
    return  nnz,enuke,B

def getGammaold(comm,basis,maxdist,maxnnz=[0],bandwidth=[0]):
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
    A        = pt.createMat(comm=comm)
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
                    distij2 = atomi.dist2(atomj) * ut.bohr2ang**2.
                    if distij2 < maxdist2: 
                        gammaij=ut.e2/np.sqrt(distij2+0.25*(atomi.rho+atomj.rho)**2.)
                        R=np.sqrt(distij2)
                        scale = PyQuante.MINDO3.get_scale(atomi.atno,atomj.atno,R)
                        enuke += ( atomi.Z*atomj.Z*gammaij +  abs(atomi.Z*atomj.Z*(ut.e2/R-gammaij)*scale) ) / ( atomi.nbf *atomj.nbf )
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
                    distij2 = atomi.dist2(atomj) * ut.bohr2ang**2.
                    if distij2 < maxdist2: 
                        gammaij=ut.e2/np.sqrt(distij2+0.25*(atomi.rho+atomj.rho)**2.)
                        R=np.sqrt(distij2)
                        scale = PyQuante.MINDO3.get_scale(atomi.atno,atomj.atno,R)
                        enuke += ( atomi.Z*atomj.Z*gammaij +  abs(atomi.Z*atomj.Z*(ut.e2/R-gammaij)*scale) ) / ( atomi.nbf *atomj.nbf )
                        A[i,j] = gammaij
                        A[j,i] = gammaij
                        nnz += 1
    A.assemblyBegin()
    enuke =  comm.allreduce(enuke)        
    nnz =  comm.allreduce(nnz)  + nbf      
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

def getD0(comm,basis,guess=0,T=None):
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
        A= pt.createMat(comm=comm)
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
            pt.write("You need to give a template matrix for guess type {0}".format(guess))            
    return  A 

def getG(comm, basis, T=None):
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
        A               = pt.createMat(comm=comm)
        A.setType('aij') #'sbaij'
        A.setSizes([nbf,nbf]) 
        A.setPreallocationNNZ(maxnnzperrow) 
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

def getH(basis, T=None,comm=None):
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
        A               = pt.createMat(comm=comm)
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

def getF(atomids, D, F0, T, G, H):
    """
    Density matrix dependent terms of the Fock matrix
    """
    diagG = G.getDiagonal()
    diagD = pt.convert2SeqVec(D.getDiagonal()) 
    A     = T.duplicate()
    A.setUp()
    rstart, rend = A.getOwnershipRange()
    for i in xrange(rstart,rend):
        atomi=atomids[i]
        colsD,valsD = D.getRow(i)
        colsG,valsG = G.getRow(i)
        colsH,valsH = H.getRow(i)
        colsT,valsT = T.getRow(i)
        tmpii       = 0.5 * diagD[i] * diagG[i] # Since g[i,i]=h[i,i]
        k=0
        idxG=0
        for j in colsT:
            atomj=atomids[j]
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

def getFDseq(atomids, Dseq, F0, T, G, H):
    """
    Density matrix dependent terms of the Fock matrix
    """
    diagG = G.getDiagonal()
    diagD = pt.convert2SeqVec(D.getDiagonal()) 
    A     = T.duplicate()
    A.setUp()
    rstart, rend = A.getOwnershipRange()
    for i in xrange(rstart,rend):
        atomi=atomids[i]
        colsD,valsD = D.getRow(i)
        colsG,valsG = G.getRow(i)
        colsH,valsH = H.getRow(i)
        colsT,valsT = T.getRow(i)
        tmpii       = 0.5 * diagD[i] * diagG[i] # Since g[i,i]=h[i,i]
        k=0
        idxG=0
        for j in colsT:
            atomj=atomids[j]
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

def scf(opts,nocc,atomids,D,F0,T,G,H,stage):
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
    homo      = 0
    lumo      = 0
    converged = False 
    eps       = None   
    subint    = [0]
    pt.write("{0:*^60s}".format("SELF-CONSISTENT-FIELD ITERATIONS"))
    pt.write("SCF threshold: {0:5.3e}".format(scfthresh))
    pt.write("Maximum number of SCF iterations: {0}".format(maxiter))
    if staticsubint == 0: 
        pt.write("Fixed subintervals will be used")
    elif staticsubint == 1: 
        pt.write("Subintervals will be adjusted at each iteration with fixed interval")
    elif staticsubint == 2: 
        pt.write("Subintervals will be adjusted at each iteration")
    else:
        pt.write("Not available")   
    if usesips:
        try:
            import SIPs.sips as sips
        except:
            pt.write("SIPs not found")
            usesips = False
    for k in xrange(1,maxiter+1):
        pt.write("{0:*^60s}".format("Iteration "+str(k)))
        t0 = pt.getWallTime()
        stage = pt.getStage(stagename='F',oldstage=stage)
        F    = getF(atomids, D, F0, T, G, H)
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
        
        if (len(eigarray)>nocc):
            homo = eigarray[nocc-1] 
            lumo = eigarray[nocc]
            gap = lumo - homo             
            writeEnergies(homo,unit='ev',enstr='HOMO')
            writeEnergies(lumo,unit='ev',enstr='LUMO')
            writeEnergies(gap,unit='ev',enstr='Gap')
        writeEnergies(Eel, unit='ev', enstr='Eel')
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
        pt.getWallTime(t0,str='Iteration')
        if abs(Eel-Eold) < scfthresh and nconv >= nocc:
            pt.write("Converged at iteration {0}".format(k))
            converged = True
            return converged, Eel, homo, lumo
    return converged, Eel, homo, lumo

def getEnergy(qmol,opts):
    stage, t0   = pt.getStageTime(newstage='MINDO3')
    maxdist     = opts.getReal('maxdist', 1.e6)
    solve       = opts.getInt('solve', 0)
    maxnnz      = [opts.getInt('maxnnz', 0)]
    guess       = opts.getInt('guess', 0)
    bandwidth   = [opts.getInt('bw', 0)]
    spfilter    = opts.getReal('spfilter',0.)
    debug       = opts.getBool('debug',False)
    checknnz    = opts.getBool('checknnz',False) 
    pyquante    = opts.getBool('pyquante',False) #TODO get Pyquante energy for testing
    from PyQuante.MINDO3 import initialize
    qmol  = initialize(qmol)
    pt.getWallTime(t0,str="MINDO3 initialization")
    if pyquante:
        from PyQuante.MINDO3 import scf as pyquantescf
        Epyquante = pyquantescf(qmol)
    atoms   = qmol.atoms
    Eref  = getEref(qmol)
    nbf   = getNbasis(qmol)    
    nel   = getNelectrons(atoms)
    nocc  = nel/2
    basis = getBasis(qmol, nbf)
    atomids = getAtomIDs(basis)
    matcomm = pt.getComm()
    pt.write("Distance cutoff: {0:5.3f}".format(maxdist))
    pt.write("Number of basis functions  : {0} = Matrix size".format(nbf))
    pt.write("Number of valance electrons: {0}".format(nel))
    pt.write("Number of occupied orbitals: {0} = Number of required eigenvalues".format(nocc))
    if checknnz:
        stage,t = pt.getStageTime(newstage='Nonzero info', oldstage=stage,t0=t0)
        maxnnz,bandwidth = pt.getNnzInfo(basis, maxdist)
        t0=t
    stage, t = pt.getStageTime(newstage='T', oldstage=stage,t0=t0)
    nnz, Enuke, T            = getT(matcomm, basis, maxdist,preallocate=True)
#    Enuke             = getNuclearEnergy(len(atoms), atoms, maxdist)
#    dennnz = nnz / (nbf*(nbf+1)/2.0)  * 100.
    dennnz = (100. * nnz) / (nbf*nbf) 
    pt.write("Nonzero density percent : {0:6.3f}".format(dennnz))
    writeEnergies(Eref, unit='kcal', enstr='Eref')
    writeEnergies(Enuke, unit='ev', enstr='Enuc')
    stage, t = pt.getStageTime(newstage='F0', oldstage=stage, t0=t)
    F0    = getF0(atoms, basis, T)
    stage, t = pt.getStageTime(newstage='D0', oldstage=stage ,t0=t)
    D0     = getD0(matcomm,basis,guess=guess,T=T)
    stage, t = pt.getStageTime(newstage='G', oldstage=stage, t0=t)
    G     = getG(matcomm,basis)    
    stage, t = pt.getStageTime(newstage='H', oldstage=stage, t0=t)
    H     = getH(basis,T=G)
    pt.getStageTime(oldstage=stage, t0=t)
    pt.getWallTime(t0,str="Pre-SCF")
    t0          = pt.getWallTime()
    converged, Eelec, homo, lumo = scf(opts,nocc,atomids,D0,F0,T,G,H,stage)
    gap = lumo - homo
    if converged:
        pt.getWallTime(t0,str="SCF")
    else:    
        pt.getWallTime(t0,str="No convergence")
    Etot   = Eelec + Enuke
    Efinal = Etot*ut.ev2kcal+Eref
    writeEnergies(Eref, unit='kcal', enstr='Eref')
    writeEnergies(Enuke, 'ev', 'Enuc')
    writeEnergies(Eelec, unit='ev', enstr='Eel')
    writeEnergies(homo,unit='ev',enstr='HOMO')
    writeEnergies(lumo,unit='ev',enstr='LUMO')
    writeEnergies(gap,unit='ev',enstr='Gap')
    writeEnergies(Efinal, unit= 'kcal', enstr='Eref+Enuc+Eelec')
    return Efinal
