import petsctools as pt
import slepctools as st
import unittools as ut
import numpy as np
import scftools as ft
import os.path
import pyquantetools as qt
import xyztools as xt
from sys import exit
try:
    import SIPs.sips as sips
    usesips = True
except:
    pt.write("Error: SIPs not found")
    usesips = False
            
def writeEnergies(en,unit='', enstr=''):
    Ekcal, Eev, Ehart = ut.convertEnergy(en, unit)
    pt.write("{0: <24s} = {1:20.10f} kcal/mol = {2:20.10f} ev = {3:20.10f} Hartree".format(enstr,Ekcal, Eev, Ehart))
    return 0

def getNuclearEnergySerial(nat,atoms,maxdist):
    maxdist2 = maxdist * maxdist * ut.ang2bohr * ut.ang2bohr
    Enuc=0.0
    for i in xrange(nat):
        atomi=atoms[i]
    #    for j in xrange(i+1,nat): # same as below
        for j in xrange(i):
            atomj=atoms[j]
            distij2 = atomi.dist2(atomj) # (in bohr squared) * bohr2ang2
            if distij2 < maxdist2:
                Enuc += qt.getEnukeij(atomi, atomj, distij2)           
    return Enuc

def getNuclearEnergy(comm,atoms,maxdist=1.E9):
    """
    Returns total nuclear energy in ev.
    Maxdist is cutoff in Angstrom.
    """
    Np=comm.size
    Na=len(atoms)
    rank=comm.rank
    Nc=Na/Np
    remainder=Na%Np
    maxdist2 = maxdist * maxdist * ut.ang2bohr * ut.ang2bohr
    Enuc = 0
    for i in xrange(Nc):
        atomi = atoms[rank*Nc+i]
        for j in xrange(rank*Nc+i):
            atomj = atoms[j]
            distij2 = atomi.dist2(atomj) # (in bohr squared) * bohr2ang2
            if distij2 < maxdist2:
                Enuc += qt.getEnukeij(atomi, atomj, distij2)   
    if remainder - rank > 0:
        atomi = atoms[Na-rank-1]
        for j in xrange(Na-rank-1):
            atomj = atoms[j]
            distij2 = atomi.dist2(atomj) # (in bohr squared) * bohr2ang2
            if distij2 < maxdist2:
                Enuc += qt.getEnukeij(atomi, atomj, distij2)   

    return pt.getCommSum(comm,Enuc)

        
def getNuclearEnergyFull(comm,atoms):
    """
    Computes nuclear energy as defined in MINDO/3 method.
    Calls getEnukeij
    TODO:
    Rewrite with Cython at least getEnukeij function
    There should be very efficient tecniques for this calculation, since
    it is standard in all atomic simulations.
    """
    Np=comm.size
    Na=len(atoms)
    rank=comm.rank
    Nc=Na/Np
    remainder=Na%Np
    Enuc = 0
    for i in xrange(Nc):
        atomi = atoms[rank*Nc+i]
        for j in xrange(rank*Nc+i):
            atomj = atoms[j]
            distij2 = atomi.dist2(atomj) # (in bohr squared) * bohr2ang2
            Enuc += qt.getEnukeij(atomi, atomj, distij2)   
    if remainder - rank > 0:
        atomi = atoms[Na-rank-1]
        for j in xrange(Na-rank-1):
            atomj = atoms[j]
            distij2 = atomi.dist2(atomj) # (in bohr squared) * bohr2ang2
            Enuc += qt.getEnukeij(atomi, atomj, distij2)   

    return pt.getCommSum(comm,Enuc)   

def getT(comm,pos,basis,maxdist,nnzinfo=None,rrange=None):
    """
    Computes a matrix for the two-center two-electron integrals.
    Assumes spherical symmetry, no dependence on basis function, only atom types.
    Parametrized for pairs of atoms. (Two-atom parameters)
    This matrix also determines the nonzero structure of the Fock matrix.
    Nuclear repulsion energy is also computed.
    TODO:
    Nuclear energy better be removed from here.    
    Values are indeed based on atoms, not basis functions, so computations can be
    reduced by nbf/natom, but not sure how to vectorize.
    Use SBAIJ instead of AIJ
    Cythonize
    """
    t            = pt.getWallTime()
    nbf          = len(basis)
    maxdist2     = maxdist * maxdist    
    e2           = ut.e2
    if rrange is None:
        rrange   = pt.distributeN(comm, nbf)
    if nnzinfo is None:
        nnzinfo  = pt.getLocalNnzInfo(pos,maxdist2,rrange)
        t        = pt.getWallTime(t0=t,str='Count nnz')
    rstart, rend = rrange    
    localsize    = rend - rstart
    Enuc         = 0.
    A            = pt.createMat(comm=comm)
    t = pt.getWallTime(t0=t,str='Create Mat')
    A.setType('aij')
    A.setSizes([(localsize,nbf),(localsize,nbf)])
    dnnz, onnz  = nnzinfo
    nnz            = sum(dnnz) + sum(onnz)
    if localsize > 0 :
        A.setPreallocationNNZ((dnnz,onnz))
    else: 
        A.setPreallocationNNZ((0,0))
    t = pt.getWallTime(t0=t,str='Preallocate')
    baseidx = np.arange(nbf,dtype='int32')
    for i in range(rstart,rend):
        atomi   = basis[i].atom
        atnoi   = atomi.atno
        rhoi    = atomi.rho
        gammaii = qt.f03[atnoi]
        disti   = pos - pos[i]
        dists2  = np.sum(disti * disti,axis=1)
        cols    = baseidx[dists2 < maxdist2]
        vals    = np.zeros(len(cols))
        for n, j in enumerate(cols):
            atomj = basis[j].atom
            if atomi.atid == atomj.atid:
                vals[n] = gammaii
            else:                        
                distij2 = dists2[j]
                rhoj    = atomj.rho 
                gammaij = e2 / np.sqrt(distij2 + 0.25 * (rhoi + rhoj)*(rhoi + rhoj))
                R       = np.sqrt(distij2)
                vals[n] = gammaij
                atnoj   = atomj.atno
                Enuc  +=  ( ( atomi.Z * atomj.Z * gammaij 
                              + abs(atomi.Z * atomj.Z * (e2/R-gammaij) * qt.getScaleij(atnoi, atnoj, R)) ) 
                           / (atomi.nbf * atomj.nbf) )
        A.setValues(i,cols,vals,addv=pt.INSERT)
    t = pt.getWallTime(t0=t,str='For loop')
    A.assemblyBegin()
    A.assemblyEnd()
    t = pt.getWallTime(t0=t,str='Assemble mat')
    Enuc = 0.5 * pt.getCommSum(comm, Enuc)
    nnz =  pt.getCommSum(comm, nnz, integer=True) 
    t = pt.getWallTime(t0=t,str='Reductions')
    return  nnz,Enuc, A

def getTold(comm,basis,maxdist,nnzinfo=None,rrange=None):
    """
    Computes a matrix for the two-center two-electron integrals.
    Matrix is symmetric, and upper-triangular is computed.
    Diagonals assumes half of the real value since
    at the end transpose of the matrix is added to make it symmetric.
    Assumes spherical symmetry, no dependence on basis function, only atom types.
    Parametrized for pairs of atoms. (Two-atom parameters)
    This matrix also determines the nonzero structure of the Fock matrix.
    Nuclear repulsion energy is also computed.
    TODO:
    Values are indeed based on atoms, not basis functions, so possible to improve performance by nbf/natom.
    Use SBAIJ instead of AIJ
    Cythonize
    """
    t            = pt.getWallTime()
    nbf          = len(basis)
    maxdist2     = maxdist * maxdist    
    maxdist2b    = maxdist2 * ut.ang2bohr * ut.ang2bohr    
    bohr2ang2    = ut.bohr2ang**2.
    e2           = ut.e2
    if rrange is None:
        rstart, rend = pt.distributeN(comm, nbf)
    else:
        rstart, rend = rrange    
    localsize    = rend - rstart
    Enuc         = 0.
    k            = 0   
    A            = pt.createMat(comm=comm)
    t = pt.getWallTime(t0=t,str='Create Mat')
    A.setType('aij')
    A.setSizes([(localsize,nbf),(localsize,nbf)])
    if nnzinfo is None:
        nnzinfo = pt.getLocalNnzInfoPQ(basis,rstart,rend,maxdist2b)
        t       = pt.getWallTime(t0=t,str='Count nnz')
    dnnz,onnz,jmax = nnzinfo
    nnz            = sum(dnnz) + sum(onnz)
    if localsize > 0 :
        A.setPreallocationNNZ((dnnz,onnz))
    else: 
        A.setPreallocationNNZ((0,0))
    t = pt.getWallTime(t0=t,str='Preallocate')
    for i in xrange(rstart,rend):
        atomi   = basis[i].atom
        atnoi   = atomi.atno
        rhoi    = atomi.rho
        gammaii = qt.f03[atnoi]
        nnzrow  = dnnz[k] + onnz[k]
        cols    = np.zeros((nnzrow,), dtype=np.int32)
        vals    = np.zeros(nnzrow)
        #A[i,i]  = gammaii / 2.
        cols[0] = i
        vals[0] = gammaii / 2.
        n       = 1
        for j in xrange(i+1,jmax[k]+1):
            atomj = basis[j].atom
            if atomi.atid == atomj.atid:
                #A[i,j] = gammaii
                cols[n] = j
                vals[n] = gammaii
                n += 1 
            else:                        
                distij2 = atomi.dist2(atomj) 
                if distij2 < maxdist2b:
                    rhoj    = atomj.rho 
                    distij2 = distij2 * bohr2ang2
                    gammaij = e2 / np.sqrt(distij2 + 0.25 * (rhoi + rhoj)**2.)
                    R       = np.sqrt(distij2)
                    #A[i,j]  = gammaij
                    cols[n] = j
                    vals[n] = gammaij
                    atnoj   = atomj.atno
                    Enuc  +=  ( ( atomi.Z * atomj.Z * gammaij 
                                  + abs(atomi.Z * atomj.Z * (ut.e2/R-gammaij) * qt.getScaleij(atnoi, atnoj, R)) ) 
                               / (atomi.nbf * atomj.nbf) )
                    n += 1
        A.setValues(i,cols[0:n],vals[0:n],addv=pt.INSERT)
        k += 1            
    t = pt.getWallTime(t0=t,str='For loop')
    A.assemblyBegin()
    A.assemblyEnd()
    t = pt.getWallTime(t0=t,str='Assemble mat')
    Enuc = pt.getCommSum(comm, Enuc)
    nnz =  pt.getCommSum(comm, nnz, integer=True) 
    t = pt.getWallTime(t0=t,str='Reductions')
    B = A.duplicate(copy=True)
    B = B + A.transpose() 
    t = pt.getWallTime(t0=t,str='Add transpose')
    return  nnz,Enuc, B

def getTDense(comm,basis):
    """
    Computes a matrix for the two-center two-electron integrals.
    Matrix is symmetric, and upper-triangular is computed.
    Diagonals assumes half of the real value since
    at the end transpose of the matrix is added to make it symmetric.
    Assumes spherical symmetry, no dependence on basis function, only atom types.
    Parametrized for pairs of atoms. (Two-atom parameters)
    This matrix also determines the nonzero structure of the Fock matrix.
    Nuclear repulsion energy is also computed.
    TODO:
    Values are indeed based on atoms, not basis functions, so possible to improve performance by nbf/natom.
    Use SBAIJ instead of AIJ
    Cythonize
    """
    t            = pt.getWallTime()
    nbf          = len(basis)
    bohr2ang2    = ut.bohr2ang**2.
    e2           = ut.e2
    rstart, rend = pt.distributeN(comm, nbf)
    localsize    = rend - rstart
    Enuc         = 0.
    k            = 0   
    t            = pt.getWallTime(t0=t,str='Initialize')
    pt.sync()
    t            = pt.getWallTime(t0=t,str='Barrier - distribute')
    A            = pt.createDenseMat([(localsize,nbf),(localsize,nbf)],comm=comm)
    t = pt.getWallTime(t0=t,str='Create Dense Mat')
    for i in xrange(rstart,rend):
        atomi   = basis[i].atom
        atnoi   = atomi.atno
        rhoi    = atomi.rho
        gammaii = qt.f03[atnoi]
        cols    = np.zeros(nbf, dtype=np.int32)
        vals    = np.zeros(nbf, dtype=np.int32)
        cols[0] = i
        vals[0] = gammaii / 2.
        n       = 1
        for j in xrange(i):
            atomj = basis[j].atom
            if atomi.atid == atomj.atid:
                cols[n] = j
                vals[n] = gammaii
            else:                        
                distij2 = atomi.dist2(atomj) # (in bohr squared) * bohr2ang2
                rhoj    = atomj.rho 
                distij2 = distij2 * bohr2ang2
                gammaij = e2 / np.sqrt(distij2 + 0.25 * (rhoi + rhoj)**2.)
                R       = np.sqrt(distij2)
                cols[n] = j
                vals[n] = gammaij
                atnoj   = atomj.atno
                Enuc  +=  ( ( atomi.Z * atomj.Z * gammaij 
                              + abs(atomi.Z * atomj.Z * 
                                    (ut.e2/R-gammaij) * 
                                    qt.getScaleij(atnoi, atnoj, R) ) ) 
                           / (atomi.nbf * atomj.nbf) )
            n += 1
        A.setValues(i,cols[0:n],vals[0:n],addv=pt.INSERT)
        k += 1            
    t = pt.getWallTime(t0=t,str='For loop')
    A.assemblyBegin()
    A.assemblyEnd()
    t = pt.getWallTime(t0=t,str='Assemble mat')
    Enuc = pt.getCommSum(comm, Enuc)
    t = pt.getWallTime(t0=t,str='Reductions')
    B = A.duplicate(copy=True)
    B = B + A.transpose() 
    t = pt.getWallTime(t0=t,str='Add transpose')
    A.destroy()
    return  Enuc, B

def getTFromGuess(comm,guessmat,basis):
    """
    Returns a matrix that stores gammaij.
    The matrix has the same nonzero pattern of guesmat.
    Computes nnz info, and nuclear energy.
    TODO:
    Allow to use a subset of the nonzeros of the guessmat.
    Make use of symmetry. 
    Try sbaij.
    """
    t            = pt.getWallTime()
    bohr2ang2    = ut.bohr2ang**2.
    e2           = ut.e2
    A = guessmat.duplicate()
    A.setUp()
    rstart, rend = A.getOwnershipRange()
    nnz = 0
    Enuc         = 0.
    for i in xrange(rstart,rend):
        cols, vals    = guessmat.getRow(i)
        nnz    += len(cols)
        atomi   = basis[i].atom
        atnoi   = atomi.atno
        rhoi    = atomi.rho
        gammaii = qt.f03[atnoi]
        n       = 0
        for j in cols:
            atomj = basis[j].atom
            if atomi.atid == atomj.atid:
                vals[n] = gammaii
            else:                        
                distij2 = atomi.dist2(atomj) # (in bohr squared) * bohr2ang2
                rhoj    = atomj.rho 
                distij2 = distij2 * bohr2ang2
                gammaij = e2 / np.sqrt(distij2 + 0.25 * (rhoi + rhoj)**2.)
                R       = np.sqrt(distij2)
                vals[n] = gammaij
                atnoj   = atomj.atno
                Enuc   += ( (atomi.Z*atomj.Z*gammaij 
                           +  abs(atomi.Z*atomj.Z*(ut.e2/R-gammaij)*qt.getScaleij(atnoi, atnoj, R)) ) 
                           / (atomi.nbf * atomj.nbf) )
            n += 1
        A.setValues(i,cols,vals,addv=pt.INSERT)
    t = pt.getWallTime(t0=t,str='For loop')
    A.assemblyBegin()
    A.assemblyEnd()
    t = pt.getWallTime(t0=t,str='Assemble mat')
    Enuc = .5 * pt.getCommSum(comm, Enuc) # Since we disregard symmetry.
    nnz =  pt.getCommSum(comm, nnz, integer=True) 
    t = pt.getWallTime(t0=t,str='Reductions')
    return  nnz,Enuc, A

def getF0(atoms,basis,T):
    """
    Form the zero-iteration (density matrix independent) Fock matrix
    Ref 1: DOI:10.1021/ja00839a001
    Ref 2: ISBN:089573754X
    Ref 3: DOI:10.1002/wcms.1141
    ToDo:
    Vectorize 
    Cythonize
    """
    A = T.duplicate( )
    A.setUp()
    rstart, rend = A.getOwnershipRange()
    for i in xrange(rstart,rend):
        basisi      = basis[i]
        cgbfi       = basisi.cgbf
        ipi         = basisi.ip
        atomi       = basisi.atom
        cols, valsT = T.getRow(i)
        vals        = np.zeros_like(valsT)
        tmp = basisi.u # Ref1, Ref2
        for k,j in enumerate(cols):
            basisj=basis[j]
            atomj=basisj.atom
            cgbfj      = basisj.cgbf
            if i==j:
                kdiag = k
            elif atomj.atid != atomi.atid:
                tmp -= valsT[k] * atomj.Z / atomj.nbf # Ref1, Ref2 adopted sum to be over orbitals rather than atoms
                betaij = qt.getBeta0ij(atomi.atno,atomj.atno)
#                Sij = basisi.cgbf.overlap(basisj.cgbf) #bottleneck
                Sij = qt.getOverlap(cgbfi,cgbfj) #bottleneck
                IPij = ipi + basisj.ip
                vals[k] = betaij * IPij * Sij     # Ref1, Ref2 
        vals[kdiag] = tmp
        A.setValues(i,cols,vals)        
    A.assemble()
    return A

def getD0Diagonal(comm,basis):
    """
    Returns the guess (initial) density matrix.
    Guesfile is the path for a stored density matrix
    If guess file is not found, a simple guess for the density 
    matrix is returned. 
    Default guess is a diagonal matrix based on valance electrons of atoms.   
    """
    
    nbf=len(basis) 
    A= pt.createMat(comm=comm)
    A.setType('aij') 
    A.setSizes([nbf,nbf])        
    A.setPreallocationNNZ(1) 
    A.setUp()
    rstart, rend = A.getOwnershipRange() 
    for i in xrange(rstart,rend):
        atomi=basis[i].atom
        if atomi.atno == 1: 
            A[i,i] = atomi.Z/1.
        else:               
            A[i,i] = atomi.Z/4.
    A.assemble()    
    return  A 

def getDiagonalD0(T,basis,nbfs):
    """
    Returns the guess (initial) density matrix
    based on the number of valance electrons
    and the number of basis functions
    to satisfy Tr(D) = sum(nves)
    Notes
    -----
    ToDo:
    Do not duplicate T, 
    simply create a diagonal matrix, or
    just use a vec.
    Use a list of number of valance e's instead
    of using pyquante atom object attribute.   
    """
    A = T.duplicate()
    diag = A.getVecLeft()  
    diag.array_w = [1.]*len(basis)
    for i,nbf in enumerate(nbfs):
        if nbf != 1:             
            diag.array_w[i] = basis[i].atom.Z / 4. 
    A.setDiagonal(diag)
    A.assemble()            
    return A

def getD0ScaledT(T,basis,nbfs,tscale=0.001):
    """
    Returns the guess (initial) density matrix
    with a diagonal based on the number of valance electrons
    and the number of basis functions
    to satisfy Tr(D) = sum(nves)
    Off-diagonal nonzeros are nonzeros of T scaled
    by a given scale.
    Notes
    -----
  
    """
    A = T.duplicate(copy=True)
    A.scale(tscale)
    diag = A.getVecLeft()  
    diag.array_w = [1.]*len(basis)
    for i,nbf in enumerate(nbfs):
        if nbf != 1:             
            diag.array_w[i] = basis[i].atom.Z / 4. 
    A.setDiagonal(diag)
    A.assemble()            
    return A

def getD0FromFile(comm,guessfile=''):
    """
    Returns the guess (initial) density matrix.
    Guesfile is the path for a stored density matrix
    If guess file is not found, a simple guess for the density 
    matrix is returned. 
    Default guess is a diagonal matrix based on valance electrons of atoms.   
    """

    if os.path.isfile(guessfile):
        pt.write("Read density matrix file: {0}".format(guessfile))
        A = pt.getMatFromFile(guessfile, comm) 
    else:
        pt.write("Can't find density matrix file: {0}".format(guessfile))
        A = None
    return  A 

def getD0BlockedFast(qmol,bstart,bend,napb,nbfpb,T):
    """
    Returns a density matrix with nonzeros on block diagonals
    based on the scf solutions for subsystems.
    """
    A = T.duplicate()
    for b in range(bstart,bend):
        astart  = b * napb
        aend    = astart + napb
        rstart  = b * nbfpb
        rend    = rstart  + nbfpb
        cqmols  = qmol.subsystem(str(b),indices=(range(astart,aend)))
        Dc      = qt.runSCF(cqmols)[3]    
        try:
            A.setValuesBlocked(range(rstart,rend),range(rstart,rend),Dc[:,:])
        except:
            pt.write("New nonzero in rows {0}:{1}".format(rstart,rend))
            import sys
            sys.exit('Error in getD0BlockedFast')    
    A.assemble()      
    return A

def getD0Blocked(qmol,bstart,bend,napb,nbfpb,T):
    """
    Returns a density matrix with nonzeros on block diagonals
    based on the scf solutions for subsystems.
    """
    A = T.duplicate()
    for b in range(bstart,bend):
        astart  = b * napb
        aend    = astart + napb
        rstart  = b * nbfpb
        rend    = rstart  + nbfpb
        cqmols  = qmol.subsystem(str(b),indices=(range(astart,aend)))
        Dc      = qt.runSCF(cqmols)[3]
        for i in range(rstart,rend):
            cols = T.getRow(i)[0]
            cols = cols[(cols>=rstart) & (cols<rend)]
            A.setValues(i,cols,Dc[i-rstart,cols-rstart])    
    A.assemble()      
    return A

def getG(comm, basis, T=None):
    """
    Returns the matrix for one-electron Coulomb term, (mu mu | nu nu)
     where mu and nu orbitals are centered on the same atom.
    Block diagonal matrix with 1x1 (Hydrogens) or 4x4 blocks.
    If T is given, assumes the nonzero pattern of T.
    """
    nbf             = len(basis)
    if T:
        A = T.duplicate( )
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
        A[i,i] = qt.getGij(basisi,basisi)
        if atomi.atno > 1:
            minj = max(0,i-nbfi)
            maxj = min(nbf,i+nbfi) 
            for j in xrange(minj,maxj):
                basisj = basis[j]
                atomj   = basisj.atom
                if atomi.atid == atomj.atid:
                    A[i,j] = qt.getGij(basisi,basisj)
    A.assemble()
    return A

def getH(basis, G=None,comm=None):
    """
    Returns the matrix for one-electron exchange term, (mu nu | mu nu) 
    where mu and nu orbitals are centered on the same atom. 
    Block diagonal matrix with 1x1 (Hydrogens) or 4x4 blocks.
    If T is given, assumes the nonzero pattern of T.
    """
    nbf             = len(basis)
    if G:
        A = G.duplicate(copy=True)
        rstart, rend = A.getOwnershipRange()
        for i in xrange(rstart,rend):
            basisi  = basis[i]
            atomi   = basisi.atom
            nbfi    = atomi.nbf
            if atomi.atno > 1:
                minj = max(0,i-nbfi)
                maxj = min(nbf,i+nbfi) 
                for j in xrange(minj,maxj):
                    basisj = basis[j]
                    atomj   = basisj.atom
                    if atomi.atid == atomj.atid and i != j:
                        A[i,j] = qt.getHij(basisi,basisj)
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
            A[i,i] = qt.getGij(basisi,basisi)
            if atomi.atno > 1:
                minj = max(0,i-nbfi)
                maxj = min(nbf,i+nbfi) 
                for j in xrange(minj,maxj):
                    basisj = basis[j]
                    atomj   = basisj.atom
                    if atomi.atid == atomj.atid and i != j:
                        A[i,j] = qt.getHij(basisi,basisj)
    A.assemble()
    return A

def getF(atomids,T,D,GH1,GH2):
    """
    Returns density matrix dependent terms of the Fock matrix.
    Parameters
    ----------
    atomids: 1D int array
             Length of array should be equal to size 
             of the matrix.
    T,D,
    GH1,GH2: Petsc mat of the same nnz pattern.
    GH1 =        G - 0.5 * H
    GH2 = -0.5 * G + 1.5 * H
    TODO:
    cython
    """
    t       = pt.getWallTime()
    diagD   = D.getDiagonal()
    diagGH1 = GH1.getDiagonal()
    diag    = diagGH1.array_r * diagD.array_r
    diagD   = pt.convert2SeqVec(diagD) 
    t       = pt.getWallTime(t0=t,str='AllGather Diag')
    A       = T.duplicate( )
    rstart, rend = A.getOwnershipRange()
    for i in xrange(rstart,rend):
        atomidi      = atomids[i]
        cols, valsT  = T.getRow(i)
        valsD        = D.getRow(i)[1] 
        valsGH1      = GH1.getRow(i)[1]
        valsGH2      = GH2.getRow(i)[1] 
        valsF        = np.zeros_like(valsT)
        tmpii        = diag[i-rstart] # 0.5 * diagD[i] * G[i,i] # Note G[i,i]=H[i,i]
        for k,j in enumerate(cols):
            if i != j:
                Djj   = diagD[j] # D[j,j]
                Dij   = valsD[k]
                Tij   = valsT[k]
                if atomids[j]  == atomidi:
                    tmpii += Djj * valsGH1[k] # Ref1 and PyQuante, In Ref2, Ref3, when i==j, g=h
                    tmpij  = Dij * valsGH2[k]  # Ref3, PyQuante, I think this term is an improvement to MINDO3 (it is in MNDO) so not found in Ref1 and Ref2  
                else:
                    tmpii += Tij * Djj     # Ref1, Ref2, Ref3
                    tmpij  = -0.5 * Tij * Dij   # Ref1, Ref2, Ref3  
                valsF[k] = tmpij
            else:
                kdiag = k    
        valsF[kdiag] = tmpii
        A.setValues(i,cols,valsF,addv=pt.INSERT)        
    t = pt.getWallTime(t0=t,str='For loop')        
    A.assemble()
    t = pt.getWallTime(t0=t,str='Mat assemble')    
    return A

def getFold(atomids,T,D,G,H):
    """
    Density matrix dependent terms of the Fock matrix
    """
    t            = pt.getWallTime()
    diagD = D.getDiagonal()
    diagD = pt.convert2SeqVec(diagD) 
    t = pt.getWallTime(t0=t,str='AllGather Diag')
    A     = T.duplicate( )
    A.setUp()
    rstart, rend = A.getOwnershipRange()
    t = pt.getWallTime(t0=t,str='Mat ops')    
    for i in xrange(rstart,rend):
        atomi        = atomids[i]
        cols, valsT  = T.getRow(i)
        valsG        = G.getRow(i)[1]
        valsD        = D.getRow(i)[1] # cols same as T
        valsH        = H.getRow(i)[1] # cols same as G
        valsF        = np.zeros_like(valsT)
        tmpii = 0.5 * diagD[i] * G[i,i] # Since g[i,i]=h[i,i]
        for k,j in enumerate(cols):
            atomj=atomids[j]
            if i != j:
                Djj   = diagD[j] # D[j,j]
                Dij   = valsD[k]
                Tij   = valsT[k]
                if atomj  == atomi:
                    Gij    = valsG[k]
                    Hij    = valsH[k]
                    tmpii += Djj * ( Gij - 0.5 * Hij ) # Ref1 and PyQuante, In Ref2, Ref3, when i==j, g=h
                    tmpij  =  0.5 * Dij * ( 3. * Hij - Gij ) # Ref3, PyQuante, I think this term is an improvement to MINDO3 (it is in MNDO) so not found in Ref1 and Ref2  
                else:
                    tmpii += Tij * Djj     # Ref1, Ref2, Ref3
                    tmpij = -0.5 * Tij * Dij   # Ref1, Ref2, Ref3  
                valsF[k] = tmpij
            else:
                kdiag = k    
        valsF[kdiag] = tmpii
        A.setValues(i,cols,valsF,addv=pt.INSERT)        
    t = pt.getWallTime(t0=t,str='For loop')        
    A.assemble()
    t = pt.getWallTime(t0=t,str='Mat assemble')    
    return A

def getFolder(atomids, D, F0, T, G, H):
    """
    Density matrix dependent terms of the Fock matrix
    """
    t            = pt.getWallTime()
    diagD = D.getDiagonal()
#    diagD = pt.getSeqArr(diagD) 
    diagD = pt.convert2SeqVec(diagD) 
    t = pt.getWallTime(t0=t,str='AllGather Diag')
    A     = T.duplicate( )
    A.setUp()
    rstart, rend = A.getOwnershipRange()
    t = pt.getWallTime(t0=t,str='Mat ops')    
    for i in xrange(rstart,rend):
        atomi        = atomids[i]
        colsT, valsT = T.getRow(i)
        colsG, valsG = G.getRow(i)
        valsD        = D.getRow(i)[1] # cols same as T
        valsH        = H.getRow(i)[1] # cols same as G
        valsF        = np.zeros(len(valsT))
        tmpii = 0.5 * diagD[i] * G[i,i] # Since g[i,i]=h[i,i]
        k=0
        idxG=0
        for j in colsT:
            atomj=atomids[j]
            if i != j:
                #tmpij = 0.
                Djj   = diagD[j] # D[j,j]
                if len(valsD)>1:
                    Dij    = valsD[k]
                else:
                    Dij    = 0.
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
                #A[i,j] = tmpij
                valsF[k] = tmpij
            else:
                kdiag = k    
            k=k+1
        #A[i,i] = tmpii
        valsF[kdiag] = tmpii
        A.setValues(i,colsT,valsF,addv=pt.INSERT)        
    t = pt.getWallTime(t0=t,str='For loop')        
    A.assemble()
    t = pt.getWallTime(t0=t,str='Mat assemble')    
    return A

def scf(nocc,atomids,D,F0,T,G,H):
    """
    Performs, self-consistent field  iterations until convergence, or max
    number of iterations reached.
    """
    t = pt.getWallTime()
    opts          = pt.options
    maxiter       = opts.getInt('maxiter', 30)
    scfthresh     = opts.getReal('scfthresh',1.e-3)
    a, b          = opts.getReal('a',-50.) , opts.getReal('b', -10.)
    bintype       = opts.getInt('bintype',1)
    rangebuffer   = opts.getReal('rangebuffer',0.25)
    eigsfile      = opts.getString('eigsfile','eigs.txt')
    local         = opts.getBool('local',True)
    nbin          = opts.getInt('eps_krylovschur_partitions',1)
    sync          = opts.getBool('sync',False)
    solve         = opts.getBool('solve',True)
    saveall       = opts.getBool('saveall',False)
    wcomm = pt.worldcomm
    nrank    = wcomm.size # total number of ranks
    npmat = nrank / nbin # number of ranks for each slice       
    Eel       = 0.
    gap       = 0.
    homo      = 0
    lumo      = 0
    converged = False 
    eps       = None   
    pt.write("{0:*^60s}".format("SELF-CONSISTENT-FIELD ITERATIONS"))
    pt.write("SCF threshold: {0:5.3e}".format(scfthresh))
    pt.write("Maximum number of SCF iterations: {0}".format(maxiter))
    pt.write("Number of bins: {0}".format(nbin))
    pt.write("Number of ranks per bin: {0}".format(npmat))
    if sync:
        pt.sync()
        t            = pt.getWallTime(t0=t,str='Barrier - SCF options')           

    eigs  = []
    if os.path.isfile(eigsfile) and bintype > 0: 
        if not pt.rank:
            eigs = np.loadtxt(eigsfile)
            print("{0} eigenvalues read from file {1}".format(len(eigs),eigsfile))
        eigs = wcomm.bcast(eigs,root=0)    
    binedges = st.getBinEdges(eigs, nbin,bintype=bintype,rangebuffer=rangebuffer,interval=[a,b])
    F = None
    F0loc = None
    Floc  = None
    Tloc  = None
    Gloc  = None
    Hloc  = None

    for k in xrange(1,maxiter+1):
        pt.write("{0:*^60s}".format("Iteration "+str(k)))
        t0 = pt.getWallTime()
        if k==1:
            stage, t = pt.getStageTime(newstage='F', t0=t0)
            Ftmp = getF(atomids,T,D,G,H)
            Ftmp = F0 + Ftmp
            F = Ftmp.copy(F,None)
            Eold = Eel
            stage, t = pt.getStageTime(newstage='Trace',oldstage=stage, t0=t)
            Eel  = 0.5 * pt.getTraceProductAIJ(D, F0+F)
            if not solve:
                break
            stage, t = pt.getStageTime(newstage='SetupEPS',oldstage=stage, t0=t)  
            eps = st.setupEPS(F, B=None,binedges=binedges)  
        else:
            Eold = Eel
            stage, t = pt.getStageTime(newstage='F',oldstage=stage, t0=t)
            if local:
                Ftmp = getF(atomids,Tloc,D,Gloc,Hloc)
                Ftmp = F0loc + Ftmp
                Floc = Ftmp.copy(Floc,None)
                stage, t = pt.getStageTime(newstage='Trace',oldstage=stage, t0=t)            
                Eel  = 0.5 * pt.getTraceProductAIJ(D, F0loc+Floc)
            else:
                Ftmp = getF(atomids,T,D,G,H)
                Ftmp = F0 + Ftmp
                F = Ftmp.copy(F,None)
                stage, t = pt.getStageTime(newstage='Trace',oldstage=stage, t0=t)            
                Eel  = 0.5 * pt.getTraceProductAIJ(D, F0+F)
            writeEnergies(Eel, unit='ev', enstr='Eel')
            stage, t = pt.getStageTime(newstage='UpdateEPS',oldstage=stage, t0=t) 
            binedges = st.getBinEdges(eigs, nbin,bintype=bintype,rangebuffer=rangebuffer,interval=[a,b])           
            if local:
                eps = st.updateEPS(eps,Floc,binedges=binedges,local=local)
            else:
                eps = st.updateEPS(eps,F,binedges=binedges,local=local)                
        stage,t = pt.getStageTime(newstage='SolveEPS'+str(k),oldstage=stage, t0=t)
        eps = st.solveEPS(eps)
        t1 = pt.getWallTime(t0=t, str='Solve')
        nconv = st.getNumberOfConvergedEigenvalues(eps)
        t1 = pt.getWallTime(t0=t1, str='Get no of eigs')
        pt.write("Number of converged eigenvalues: {0}".format(nconv))
        if nconv < nocc: 
            pt.write("Error! Missing eigenvalues.")
            pt.write("Number of required eigenvalues: {0}".format(nocc))
            break
        eigs = st.getNEigenvalues(eps,nocc)
        pt.write("Eigenvalue range: {0:5.3f}, {1:5.3f}".format(min(eigs),max(eigs)))
        t1 = pt.getWallTime(t0=t1, str='Get eigs')
        if (len(eigs)>nocc):
            homo = eigs[nocc-1] 
            lumo = eigs[nocc]
            gap = lumo - homo             
            writeEnergies(homo,unit='ev',enstr='HOMO')
            writeEnergies(lumo,unit='ev',enstr='LUMO')
            writeEnergies(gap,unit='ev',enstr='Gap')
        stage, t = pt.getStageTime(newstage='Density', oldstage=stage, t0=t)
        if usesips:
            D = sips.getDensityMat(eps,0,nocc)
        else:    
            D = st.getDensityMatrix(eps,T,nocc)
        if k==1 and local: 
            stage, t = pt.getStageTime(newstage='Redundant mat', oldstage=stage, t0=t)
            matcomm = D.getComm()
            F0loc = pt.getRedundantMat(F0, nbin, matcomm, out=F0loc)
            Floc  = pt.getRedundantMat( F, nbin, matcomm, out=Floc)
            Tloc  = pt.getRedundantMat( T, nbin, matcomm, out=Tloc)
            Gloc  = pt.getRedundantMat( G, nbin, matcomm, out=Gloc)
            Hloc  = pt.getRedundantMat( H, nbin, matcomm, out=Hloc)
        if saveall:
            ft.saveall(opts,k,Floc,D,eigs)
        t = pt.getWallTime(t0,str='Iteration')
        if abs(Eel-Eold) < scfthresh and nconv >= nocc:
            pt.write("Converged at iteration {0}".format(k))
            converged = True
            break
    return converged, Eel, homo, lumo, D

def scfwithaccelerators(opts,nocc,atomids,D,F0,T,G,H,stage):
    """
    Performs, self-consistent field  iterations until convergence, or max
    number of iterations reached. 
    Used Pulay's DIIS or Aitken's 3-point extrapolation for the acceleration
    of convergence.
    """
    maxiter     = opts.getInt('maxiter', 30)
    scfacc      = opts.getInt('scfacc', 0)
    guess       = opts.getInt('guess', 0)
    sizediis    = opts.getInt('diissize', 4)
    errdiis     = opts.getReal('diiserr', 1.e-2)
    scfthresh   = opts.getReal('scfthresh',1.e-5)
    interval    = [opts.getReal('a',-50.) , opts.getReal('b', -10.)]
    slicing     = opts.getInt('slicing',0)
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
    if slicing == 0: 
        pt.write("Fixed subintervals will be used")
    elif slicing == 1: 
        pt.write("Subintervals will be adjusted at each iteration with fixed interval")
    elif slicing == 2: 
        pt.write("Subintervals will be adjusted at each iteration")
    else:
        pt.write("Not available")   
    if  scfacc == 1:
        pt.write("Pulay's DIIS for F with sizediis: {0}",format(sizediis))
        pt.write("                   with errdiis: {0}".format(errdiis))
        Flist  = [F0] * sizediis
        Elist  = [F0] * sizediis
        idiis  = 0
    elif scfacc == 2:    
        pt.write("Pulay's DIIS for D with sizediis: {0}",format(sizediis))
        pt.write("                   with errdiis: {0}".format(errdiis))
        Dlist  = [D] * sizediis 
        Elist  = [F0] * sizediis
        idiis  = 0
    elif scfacc == 3:
        pt.write("Aitken 3-point extrapolation for F")
        Flist = [F0] * 3
    elif scfacc == 4:
        pt.write("Aitken 3-point extrapolation for D")
        Dlist = [D] * 3
        
    for k in xrange(1,maxiter+1):
        pt.write("{0:*^60s}".format("Iteration "+str(k)))
        t0 = pt.getWallTime()
        stage = pt.getStage(stagename='F',oldstage=stage)
        F    = getF(atomids,T,D,G,H)
        F    = F0 + F 
#        F.axpy(1.0,F0,structure=F.Structure.SAME_NONZERO_PATTERN) # same as above, addidtional symbolic factorizations
        if scfacc == 3:
            kmod3 = (k-1)%3
            Flist[kmod3] = F
            if kmod3==0  and k > 4:
                F = ft.extrapolate3(Flist[0], Flist[1], Flist[2])
                pt.write('3-point extrapolation for F applied')  
        elif scfacc == 1:
            FDcommutator = pt.getCommutator(F, D)
            maxerr = pt.getMaxAbsAIJ(FDcommutator)
            pt.write('max(abs([F,D])) = {0}'.format(maxerr))           
            if maxerr < errdiis:
                idiis += 1
                imod  = idiis%sizediis
                Flist[imod] = F
                Elist[imod] = FDcommutator
                if idiis > sizediis: 
                    c = ft.getDIISSolution(sizediis, Elist)
                    F = 0 
                    for i in range(sizediis):
                        F += c[i] * Flist[i]
                    pt.write('DIIS extrapolation for F applied')
        Eold = Eel
        stage = pt.getStage(stagename='Trace',oldstage=stage)
        if k==1:
            if guess==0:
                Eel  = 0.5 * pt.getTraceProductDiag(D,F0+F)
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
            if slicing == 1:
                nsubint=st.getNumberOfSubIntervals(eps)
                subint = st.getBinEdges1(eigarray[0:nocc],nsubint,interval=interval) 
            elif slicing == 2:
                nsubint=st.getNumberOfSubIntervals(eps)
                subint = st.getBinEdges1(eigarray[0:nocc],nsubint)
            eps = st.updateEPS(eps,F,binedges=subint)
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
        npmat = D.getComm().getSize()
        if scfacc == 4:
            kmod3 = (k-1)%3
            Dlist[kmod3] = D
            if kmod3==0  and k > 4:
                D = ft.extrapolate3(Dlist[0], Dlist[1], Dlist[2])
                pt.write('3-point extrapolation')  
        elif scfacc == 2:
            FDcommutator = pt.getCommutator(F, D)
            maxerr = pt.getMaxAbsAIJ(FDcommutator)
            pt.write('max(abs([F,D])) = {0}'.format(maxerr))           
            if maxerr < errdiis:
                idiis += 1
                imod  = idiis%sizediis
                Dlist[imod] = D
                Elist[imod] = FDcommutator
                if idiis > sizediis: 
                    c = ft.getDIISSolution(sizediis, Elist)
                    D = 0 
                    for i in range(sizediis):
                        D += c[i] * Dlist[i]
                    pt.write('DIIS extrapolation for D applied')
        if npmat > 1 and npmat < F.getComm().getSize():   
            D = pt.getSeqMat(D)
        pt.getWallTime(t0,str='Iteration')
        if abs(Eel-Eold) < scfthresh and nconv >= nocc:
            pt.write("Converged at iteration {0}".format(k))
            converged = True
            return converged, Eel, homo, lumo
    return converged, Eel, homo, lumo

def runMINDO3(qmol,s=None,xyz=None,opts=None):
    """
    Returns MINDO3 nuclear energy, total electronic energy and heat of formation 
    in kcal/mol for a given PyQuante molecule.
    """
    stage, t0   = pt.getStageTime(newstage='MINDO3')
    if opts is None:
        opts    = pt.options
    maxiter     = opts.getInt('maxiter', 30)
    scfthresh   = opts.getReal('scfthresh',1.e-3)
    maxdist     = opts.getReal('maxdist', 10.)
    guess       = opts.getInt('guess', 0)
    tscale      = opts.getReal('tscale', 0.001)
    guessfile   = opts.getString('guessfile', 'dens.bin')
    nuke        = opts.getBool('nuke',False)
    sync        = opts.getBool('sync',False)
    serial      = opts.getBool('serial',False)
    napc        = opts.getInt('napc', 3) # number of atoms per cluster
    napb        = opts.getInt('napb', 3) # number of atoms per block
    qmol        = qt.initialize(qmol) #bottleneck
    t           = pt.getWallTime(t0=t0,str='PyQuante initialization')  
    if sync: 
        pt.sync()
        t = pt.getWallTime(t0=t,str='Barrier - init')
    if serial:
        qt.runSCF(qmol, scfthresh=scfthresh, maxiter=maxiter)
        return 
    Eref      = qt.getEref(qmol)
    nat       = len(qmol)
    ncluster  = nat / napc
    nblock    = nat / napb
    nbf       = qt.getNBF(qmol)    
    nel       = qt.getNVE(qmol)
    nocc      = nel/2
    basis     = qt.getBasis(qmol, nbf)
    maxdist2  = maxdist * maxdist
    atomids   = qt.getAtomIDs(basis)
    worldcomm = pt.getComm()
    nrank     = worldcomm.size
    if sync:
        pt.sync()
        t            = pt.getWallTime(t0=t,str='Barrier - options')
    pt.write("Distance cutoff: {0:5.3f}".format(maxdist))
    pt.write("Number of basis functions  : {0} = Matrix size".format(nbf))
    pt.write("Number of valance electrons: {0}".format(nel))
    pt.write("Number of clusters: {0}".format(ncluster))
    pt.write("Number of occupied orbitals: {0} = Number of required eigenvalues".format(nocc))
    t           = pt.getWallTime(t0=t,str='Basis set')
    if xyz is None:
        bxyz = qt.getXYZFromBasis(qmol,basis)
    else:
        nbfs = xt.getNBFsFromS(s)
        bxyz = xt.getExtendedArray(nbfs,xyz)
    pt.write("Guess density matrix option: {0}".format(guess))    
    if guess < 2:   
        stage, t = pt.getStageTime(newstage='T', oldstage=stage,t0=t0)
        rstart, rend = pt.distributeN(worldcomm, nbf)
        nnzinfo = pt.getLocalNnzInfo(bxyz,maxdist2,(rstart,rend))
        t = pt.getWallTime(t0=t,str='Count nnz')
        nnz, Enuc, T            = getT(worldcomm,bxyz,basis, maxdist,nnzinfo,rrange=(rstart,rend))
        stage, t = pt.getStageTime(newstage='D0', oldstage=stage ,t0=t)
        pt.write("Generating diagonal density matrix")
        bnbfs = xt.getExtendedArray(nbfs,nbfs)
        if  guess == 0:      
            D0     = getDiagonalD0(T,basis[rstart:rend],bnbfs[rstart:rend]) 
        elif guess == 1:
            D0     = getD0ScaledT(T,basis[rstart:rend],bnbfs[rstart:rend],tscale)
        t = pt.getWallTime(t0=t,str='D0 generated in')          
    elif guess == 2 or guess == 3:
        pt.write("Number of blocks, {0}".format(nblock))
        pt.write("Use -sort 3")
        if nat % napb != 0 :
            pt.write("Incompetible number of atoms per block: {0}".format(napc))
            return
        elif nblock < nrank:
            pt.write("Number of blocks, {0}, is less than number of ranks!".format(ncluster))   
        else:
            stage, t = pt.getStageTime(newstage='T', oldstage=stage,t0=t0)
            nbfpb    = nbf / nblock 
            bstart, bend = pt.distributeN(worldcomm,nblock)
            rstart, rend = bstart * nbfpb, bend * nbfpb
            nnzinfo = pt.getLocalNnzInfo(bxyz,maxdist2,(rstart,rend))
            t = pt.getWallTime(t0=t,str='Count nnz')
            nnz, Enuc, T            = getT(worldcomm,bxyz,basis,maxdist,nnzinfo,rrange=(rstart,rend))
            stage, t = pt.getStageTime(newstage='D0', oldstage=stage ,t0=t)
            pt.write("Generating block-diagonal density matrix")
            if guess == 3:
                D0 = getD0Blocked(qmol,bstart,bend, napb, nbfpb, T)
            elif guess == 2:
                D0 = getD0BlockedFast(qmol,bstart,bend, napb, nbfpb, T)     
            t = pt.getWallTime(t0=t,str='D0 generated in')        
    elif guess == 4:
        pt.write("Reading density matrix from file")
        stage, t = pt.getStageTime(newstage='D0', oldstage=stage ,t0=t)
        D0 = getD0FromFile(worldcomm, guessfile)
        t = pt.getWallTime(t0=t,str='D0 generated in')
        if D0:
            stage, t = pt.getStageTime(newstage='T', oldstage=stage,t0=t0)
            nnz, Enuc, T            = getTFromGuess(worldcomm, D0, basis)
    else:
        pt.write("Not implemented guess option".format(guess))
        return        
    dennnz = (100. * nnz) / (nbf*nbf) 
    pt.write("Nonzero density percent : {0:6.3f}".format(dennnz))
    if nuke:
        stage, t = pt.getStageTime(newstage='Nuclear', oldstage=stage,t0=t0)
        Enukefull                = getNuclearEnergyFull(worldcomm, qmol)   
        writeEnergies(Enukefull, unit='ev', enstr='Enucfull')
    writeEnergies(Eref, unit='ev', enstr='Eref')
    writeEnergies(Enuc, unit='ev', enstr='Enuc')
    stage, t = pt.getStageTime(newstage='F0', oldstage=stage, t0=t)
    F0    = getF0(qmol,basis,T)
    stage, t = pt.getStageTime(newstage='G & H', oldstage=stage, t0=t)
    G     = getG(worldcomm,basis,T)    
    H     = getH(basis,G)
    GH1   =        G - 0.5 * H
    GH2   = -0.5 * G + 1.5 * H 
    pt.getStageTime(oldstage=stage, t0=t)
    pt.getWallTime(t0,str="Pre-SCF")
    t0          = pt.getWallTime()
    converged, Eelec, homo, lumo, D = scf(nocc,atomids,D0,F0,T,GH1,GH2)
    if nuke:
        Etotfull   = Eelec + Enukefull
        Efinalfull = Etotfull*ut.ev2kcal+Eref
        writeEnergies(Enukefull, 'ev', 'Enucfull')
        writeEnergies(Etotfull,unit='ev',enstr='Enucfull+Eelec')
        writeEnergies(Efinalfull, unit= 'kcal', enstr='Eref+Enucfull+Eelec')
    gap = lumo - homo
    Etot   = Eelec + Enuc
    Efinal = Etot  + Eref
    if converged:
        pt.getWallTime(t0,str="SCF achieved")
        writeEnergies(Eref, unit='ev', enstr='Eref')
        writeEnergies(Enuc, 'ev', 'Enuc')
        writeEnergies(Eelec, unit='ev', enstr='Eel')
        writeEnergies(homo,unit='ev',enstr='HOMO')
        writeEnergies(lumo,unit='ev',enstr='LUMO')
        writeEnergies(gap,unit='ev',enstr='Gap')
        writeEnergies(Etot,unit='ev',enstr='Enuc+Eelec')
        writeEnergies(Efinal, unit= 'ev', enstr='Eref+Enuc+Eelec')
    else:    
        pt.getWallTime(t0,str="SCF FAILED!!!")
    return Enuc,Etot,Efinal,D

def testNuclearEnergy(atoms=None):
    """
    Tests getNuclearEnergy with PyQuante result.
    """
    if atoms == None:
        atoms = qt.getH2O()
        qt.initializeMindo3(atoms)
    comm = pt.getComm()
    thresh = 1.E-5
    Epscf = getNuclearEnergy(comm, atoms)
    if comm.rank == 0:
        Epq = qt.getPQNuclearEnergy(atoms)   
        assert (abs(Epscf-Epq) < thresh), "Nuclear energies differ more than 0.01 mev"
    return True
    
def testMINDO3Energy(atoms=None,s=None,xyz=None):
    """
    Tests PSCF MINDO3 energy with PyQuante result.
    Threshold is 0.001 ev since this is the threshold
    of PyQuante SCF iterations.
    """
    if atoms == None:
        atoms = qt.getH2O()
        qt.initializeMindo3(atoms)
    thresh = 0.001 # ev
    comm = pt.getComm()
    Epscf = runMINDO3(atoms,s=s,xyz=xyz)[2]
    if comm.rank == 0:
        Epq = qt.getPQMINDO3Energy(atoms)
        assert (abs(Epscf-Epq) < thresh), "MINDO3 energies differ more than 0.001 ev"
        pt.write("PSCF MINDO3 energy matches PyQuante value...")
    return True    