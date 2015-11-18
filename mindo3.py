import sys, petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np
import petsctools as pt
import slepctools as st
import PyQuante
from mpi4py import MPI

Print = PETSc.Sys.Print

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

def getGamma(basis,maxdist,maxnnz=[0],bandwidth=[0],matcomm=PETSc.COMM_SELF):
    """
    Computes MINDO3 nuclear repulsion energy and gamma matrix
    Nuclear repulsion energy: Based on PYQuante MINDO3 get_enuke(atoms)
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
    A        = PETSc.Mat().create(comm=matcomm)
    A.setType('aij') #'sbaij'
   # A.setOption(A.Option.SYMMETRIC,True)
    A.setSizes([nbf,nbf]) 
   # if any(maxnnz): A.setPreallocationNNZ(maxnnz) 
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
                distij2 = atomi.dist2(atomj) * const.bohr2ang**2.
                if distij2 < mindist:
                    A[i,j] = gammaii
                    A[j,i] = gammaii
                    nnz += 1
                elif distij2 < maxdist2: 
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
                distij2 = atomi.dist2(atomj) * const.bohr2ang**2.
                if distij2 < mindist:
                    A[i,j] = gammaii
                    A[j,i] = gammaii
                    nnz += 1
                if distij2 < maxdist2: 
                    gammaij=const.e2/np.sqrt(distij2+0.25*(atomi.rho+atomj.rho)**2.)
                    R=np.sqrt(distij2)
                    scale = PyQuante.MINDO3.get_scale(atomi.atno,atomj.atno,R)
                    enuke += atomi.Z*atomj.Z*gammaij +  abs(atomi.Z*atomj.Z*(const.e2/R-gammaij)*scale)
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
                betaij = PyQuante.MINDO3.get_beta0(atomi.atno,atomj.atno)
                Sij = basisi.cgbf.overlap(basisj.cgbf)
                IPij = ipi + basisj.ip
                tmp2 =  betaij * IPij * Sij     # Ref1, Ref2 
                A[i,j] = tmp2
            k = k + 1    
        A[i,i] = tmp        
    A.assemble()
    return A

def getD0(basis,guess=0,T=None,matcomm=PETSc.COMM_SELF):
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
        A= PETSc.Mat().create(comm=matcomm)
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

def getG(basis,comm=PETSc.COMM_SELF,T=None):
    """
    Returns the matrix for one-electron Coulomb term, (mu mu | nu nu) where mu and nu orbitals are centered on the same atom.
    Block diagonal matrix with 1x1 (Hydrogens) or 4x4 blocks.
    If T is given, assumes the nonzero pattern of T.
    """
    if T:
        A = T.duplicate()
    else:        
        nbf             = len(basis)
        maxnnzperrow    = 4
        A               = PETSc.Mat().create(comm=matcomm)
        A.setType('aij') #'sbaij'
        A.setSizes([nbf,nbf]) 
        A.setPreallocationNNZ(maxnnzperrow) 
    
    A.setUp()
    rstart, rend = A.getOwnershipRange()
    for i in xrange(rstart,rend):
        basisi  = basis[i]
        atomi   = basisi.atom
        for j in xrange(maxnnzperrow):
            basisj = basis[i+j]
            atomj   = basisj.atom
            if atomi == atomj:
                A[i,j] = PyQuante.MINDO3.get_g(basisi,basisj)
    A.assemble()
    return A

def getH(basis,T):
    """
    Returns the matrix for one-electron exchange term, (mu nu | mu nu) where mu and nu orbitals are centered on the same atom. 
    Block diagonal matrix with 1x1 (Hydrogens) or 4x4 blocks.
    If T is given, assumes the nonzero pattern of T.
    """
    if T:
        A = T.duplicate()
    else:        
        nbf             = len(basis)
        maxnnzperrow    = 4
        A               = PETSc.Mat().create(comm=matcomm)
        A.setType('aij') #'sbaij'
        A.setSizes([nbf,nbf]) 
        A.setPreallocationNNZ(maxnnzperrow) 
    
    A.setUp()
    rstart, rend = A.getOwnershipRange()
    for i in xrange(rstart,rend):
        basisi  = basis[i]
        atomi   = basisi.atom
        for j in xrange(maxnnzperrow):
            basisj = basis[i+j]
            atomj   = basisj.atom
            if atomi == atomj:
                A[i,j] = PyQuante.MINDO3.get_h(basisi,basisj)
    A.assemble()
    return A

def getFD(atoms, basis, D, diagD, T):
    """
    Density matrix dependent terms of the Fock matrix
    """
    A = T.duplicate()
    A.setUp()
    rstart, rend = A.getOwnershipRange()
    for i in xrange(rstart,rend):
        basisi=basis[i]
        atomi=basisi.atom
        colsT,valsT = T.getRow(i)
        colsD,valsD = D.getRow(i)
        tmpii = 0.5 * diagD[i] * PyQuante.MINDO3.get_g(basisi,basisi) # Since g[i,i]=h[i,i] 
        tmpii1=0
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

def scf(qmol,opts):
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
    if not staticsubint:
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
        nnz,Enuke, T     = getGamma(basis, maxdist,maxnnz=nnzarray, bandwidth=bwarray, matcomm=PETSc.COMM_WORLD)
        dennnz = nnz / (nbf*(nbf+1)/2.0)  * 100.
        Print("Nonzero density percent : {0}".format(dennnz))
        Print("Enuc2          = {0:20.10f} kcal/mol = {1:20.10f} ev = {2:20.10f} Hartree".format(Enuke*const.ev2kcal,Enuke,Enuke*const.ev2hartree))
    else:
        stage = pt.getStage(stagename='getGamma', oldstage=stage)
        nnz, Enuke, T     = getGamma(basis, maxdist,maxnnz=maxnnz, bandwidth=bandwidth, matcomm=PETSc.COMM_WORLD)
        dennnz = nnz / (nbf*(nbf+1)/2.0)  * 100.
        Print("Nonzero density percent : {0}".format(dennnz))
        Print("Enuc3          = {0:20.10f} kcal/mol = {1:20.10f} ev = {2:20.10f} Hartree".format(Enuke*const.ev2kcal,Enuke,Enuke*const.ev2hartree))
    stage = pt.getStage(stagename='F0', oldstage=stage)
    F0    = getF0(atoms, basis, T)
    stage = pt.getStage(stagename='D0', oldstage=stage)
    D     = getD0(basis,guess=1,T=T,matcomm=PETSc.COMM_WORLD)
    stage = pt.getStage(stagename='Ddiag', oldstage=stage)
    Ddiag = pt.convert2SeqVec(D.getDiagonal()) 
        
    Eel   = 0.    
    Print("{0:*^60s}".format("SELF-CONSISTENT-FIELD ITERATIONS"))
    for iter in xrange(1,maxiter):
        Print("{0:*^60s}".format("Iteration "+str(iter)))
        stage = pt.getStage(stagename='FD', oldstage=stage)
        FD    = getFD(atoms,basis, D, Ddiag, T)
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

       