"""
This module implements Hartree-Fock method.
F0[mu,nu] = T[mu,nu] + N[mu,nu]
T[mu,nu]  = < mu | -0.5 del^2 | nu > = -0.5 * int dr mu(r) del^2 nu(r)

"""

import sys, petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np
import petsctools as pt
import slepctools as st
#import PyQuante
from mpi4py import MPI
import constants as const
import unittools as ut

Print = PETSc.Sys.Print
from PyQuante.Ints import coulomb

def getNuclearAttractionij(mol,basisi,basisj):
    tmp = 0.
    for atom in mol:
        tmp += atom.Z * basisi.nuclear(basisj,atom.pos())
    return tmp

def getNuclearRepulsionij(Zi,Zj,rij_au):
    """
    Zi: Nuclear charge of ith atom
    rij: Distance between atom i and j in Bohr (au)
    Returns the nuclear repulsion energy
    """
    return Zi*Zj/rij_au

def getNuclearEnergy(Na,mol,maxdist_au):
    Np=MPI.COMM_WORLD.size
    rank=MPI.COMM_WORLD.rank
    Nc=Na/Np
    remainder=Na%Np
    enuke = 0
    for i in xrange(Nc):
        atomi = mol[rank*Nc+i]
        Zi    = atomi.get_nuke_chg()
        for j in xrange(rank*Nc+i):
            atomj = mol[j]
            rij_au = atomi.dist(atomj) 
            if rij_au < maxdist_au:
                Zj = atomj.get_nuke_chg()
                enuke += getNuclearRepulsionij(Zi,Zj,rij_au)   
    if remainder - rank > 0:
        atomi = mol[Na-rank-1]
        for j in xrange(Na-rank-1):
            atomj = mol[j]
            rij_au = atomi.dist(atomj) # (in bohr squared) * bohr2ang2
            if rij_au < maxdist_au:
                Zj = atomj.get_nuke_chg()
                enuke += getNuclearRepulsionij(Zi,Zj,rij_au)   

    return MPI.COMM_WORLD.allreduce(enuke) 

def getBasis(mol,basis='sto-3g'):
    """\
    bfs = getbasis(atoms,basis_data=None)
    
    Given a Molecule object and a basis library, form a basis set
    constructed as a list of CGBF basis functions objects.
    """
    from PyQuante.Basis.basis import BasisSet
    from PyQuante.Basis.Tools import get_basis_data
    basis_data = get_basis_data(basis)
    return BasisSet(mol, basis_data)

def get2eInts(a,b,c,d):
    return coulomb(a, b, c, d) - 0.5 * coulomb(a, c, b, d)

def getDterm(basisi,basisj,basis,D):
    """
    Density matrix dependent terms of the Fock matrix
    """
    rstart, rend = D.getOwnershipRange()
    tmp=0.
    for m in xrange(rstart,rend):
        basism=basis[m]
        colsD,valsD = D.getRow(m)
        for n in colsD: 
            basisn=basis[n]
            tmp += D[m,n] * get2eInts(basisi, basisj, basism, basisn)
    return MPI.COMM_WORLD.allreduce(tmp)

def getS(mol,basis,maxdist,maxnnz=[0],bandwidth=[0],comm=PETSc.COMM_SELF):
    """
    Computes overlap matrix.
    Sparsity induced by distance cutoff
    TODO:
    Better to preallocate based on diagonal and offdiagonal nonzeros.
    Cythonize
    """
    import constants as const

    nbf      = len(basis)
    maxdist2 = maxdist * maxdist * const.ang2bohr * const.ang2bohr
    Vdiag = PETSc.Vec().create(comm=comm)
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
    localsize = rend-rstart
    Vdiag.setSizes((localsize,nbf))
    Vdiag.setUp()
    nnz = 0
    if any(maxnnz): 
        A.setPreallocationNNZ(maxnnz) 
    else:
        A.setPreallocationNNZ([nbf,nbf]) 
        
    if any(bandwidth):
        if len(bandwidth)==1: bandwidth=np.array([bandwidth]*nbf)
    else:
        bandwidth=np.array([nbf]*nbf)      
    for i in xrange(rstart,rend):
        basisi  = basis[i]
        atidi   = basisi.atid
        atomi   = mol[atidi]
        Vdiag[i] = 1.0
        for j in xrange(i+1,min(i+bandwidth[i],nbf)):
            basisj = basis[j]
            atidj  = basisj.atid
            if atidi == atidj:
                A[i,j] = basis[i].overlap(basisj)
                nnz += 1
            else:                        
                atomj = mol[basisj.atid]
                distij2 = atomi.dist2(atomj) 
                if distij2 < maxdist2:
                    A[i,j] = basisi.overlap(basisj)
                    nnz += 1
    A.setDiagonal(Vdiag) 
    A.assemblyBegin()
    nnz =  MPI.COMM_WORLD.allreduce(nnz)  + nbf      
    A.assemblyEnd()
    B = A.duplicate(copy=True)
    B = B + A.transpose() 
    B.setDiagonal(Vdiag) 
    return  nnz, B

def getF0(mol,basis,T):
    """
    Form the zero-iteration (density matrix independent) Fock matrix, 
    also known as core Hamiltonian

    TODO: 
    Cythonize
    """
    A = T.duplicate()
    A.setUp()
    rstart, rend = A.getOwnershipRange()
    for i in xrange(rstart,rend):
        basisi=basis[i]
        cols,vals = T.getRow(i)
        for j in cols:
            basisj  = basis[j]
            tmp  = basisi.kinetic(basisj) + getNuclearAttractionij(mol, basisi, basisj)
            A[i,j] = tmp
    A.assemble()
    return A

def getF(basis, D, T=None):
    """
    Density matrix dependent terms of the Fock matrix
    """
    if not T:
        T = D
    A = T.duplicate()
    A.setUp()
    rstart, rend = A.getOwnershipRange()
    for i in xrange(rstart,rend):
        basisi=basis[i]
        cols,vals = T.getRow(i)
        for j in cols: 
            basisj=basis[j]
            A[i,j] = getDterm(basisi,basisj,basis,D)
    A.assemble()
    return A

def rhf(opts,nocc,basis,S,F0):
    maxiter     = opts.getInt('maxiter', 30)
    guess       = opts.getInt('guess', 0)
    scfthresh   = opts.getReal('scfthresh',1.e-5)
    interval    = [opts.getReal('a',-500.) , opts.getReal('b', 500.)]
    staticsubint= opts.getInt('staticsubint',0)
    usesips     = opts.getBool('sips',False)
    Eel       = 0.
    gap       = 0.
    D         = 0.
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
        Eold = Eel
        if k==1:
            F = F0
            stage = pt.getStage(stagename='SetupEPS')    
            eps = st.setupEPS(F, B=S,interval=interval)  
            stage = pt.getStage(stagename='SolveEPS')
            eps, nconv, eigarray = st.solveEPS(eps,returnoption=1,nocc=nocc)
        else:
            F = getF(basis,D,S) + F0
            stage = pt.getStage(stagename='UpdateEPS',oldstage=stage)            
            subint =interval
            if staticsubint==1:
                nsubint=st.getNumberOfSubIntervals(eps)
                subint = st.getSubIntervals(eigarray[0:nocc],nsubint,interval=interval) 
            elif staticsubint==2:
                nsubint=st.getNumberOfSubIntervals(eps)
                subint = st.getSubIntervals(eigarray[0:nocc],nsubint)
            eps = st.updateEPS(eps,F,B=S,subintervals=subint)
            stage = pt.getStage(stagename='SolveEPS',oldstage=stage)
            eps, nconv, eigarray = st.solveEPS(eps,returnoption=1,nocc=nocc)         
        if (len(eigarray)>nocc):
            gap = eigarray[nocc] - eigarray[nocc-1]              
            Print("Gap            = {0:20.10f} kcal/mol = {1:20.10f} ev = {2:20.10f} Hartree".format(gap*const.ev2kcal,gap,gap*const.ev2hartree))  
        stage = pt.getStage(stagename='Density', oldstage=stage)
        nden = nocc
        if nconv < nocc: 
            nden = nconv
        if usesips:
            D = sips.getDensityMat(eps,0,nden)
        else:    
            D = st.getDensityMatrix(eps,S, nden)
        sizecommD = D.getComm().Get_size()    
        if sizecommD > 1 and sizecommD < F.getComm().Get_size():   
            D = pt.getSeqMat(D)
        Eel  = 0.5 * pt.getTraceProductAIJ(D, F0+F)
        Eel1 = pt.getTraceProductAIJ(D, F0)
        Eel2 = 0.5 * pt.getTraceProductAIJ(D, F) - Eel1
        Print("Eel            = {0:20.10f} kcal/mol = {1:20.10f} ev = {2:20.10f} Hartree".format(*ut.convertEnergy(Eel, 'kcal')))  
        Print("Eel1           = {0:20.10f} kcal/mol = {1:20.10f} ev = {2:20.10f} Hartree".format(*ut.convertEnergy(Eel1, 'kcal')))  
        Print("Eel2           = {0:20.10f} kcal/mol = {1:20.10f} ev = {2:20.10f} Hartree".format(*ut.convertEnergy(Eel2, 'kcal')))  
        pt.getWallTime(t0,str='Iteration completed in')
        if abs(Eel-Eold) < scfthresh and nconv >= nocc:
            Print("Converged at iteration {0}".format(k))
            converged = True
            return converged, Eel, gap
    return converged, Eel, gap

def getEnergy(mol,opts=None):
    stage       = pt.getStage(stagename='Hartree-Fock')
    t0          = pt.getWallTime()
    maxdist     = opts.getReal('maxdist', 1.e6)
    maxnnz      = [opts.getInt('maxnnz', 0)]
    guess       = opts.getInt('guess', 0)
    bandwidth   = [opts.getInt('bw', 0)]
    basisname   = opts.getString('basis','sto-3g')
    Enuke = getNuclearEnergy(len(mol), mol, maxdist)
    basis = getBasis(mol,basis=basisname)
    nocc  = mol.get_nel() / 2
    nbf   = len(basis)
    nnz,S     = getS(mol,basis,maxdist=1000)
    F0    = getF0(mol, basis, S)
    dennnz = nnz / (nbf*(nbf+1)/2.0)  * 100.
    Print("Nonzero density percent : {0}".format(dennnz))
    pt.getWallTime(t0,str="Pre-SCF steps finished in")
    converged, Eelec, gap = rhf(opts, nocc, basis, S,F0)
    Etot   = Eelec + Enuke
    Print("Enuc             = {0:20.10f} kcal/mol = {1:20.10f} ev = {2:20.10f} Hartree".format(Enuke*const.ev2kcal,Enuke,Enuke*const.ev2hartree))
    Print("Eelec            = {0:20.10f} kcal/mol = {1:20.10f} ev = {2:20.10f} Hartree".format(Eelec*const.ev2kcal,Eelec,Eelec*const.ev2hartree))
    Print("Eelec+Enuc       = {0:20.10f} kcal/mol = {1:20.10f} ev = {2:20.10f} Hartree".format(Etot*const.ev2kcal,Etot,Etot*const.ev2hartree))
    Print("Gap              = {0:20.10f} kcal/mol = {1:20.10f} ev = {2:20.10f} Hartree".format(gap*const.ev2kcal,gap,gap*const.ev2hartree))
    pt.getWallTime(t0,str="SCF finished in")
    return Etot