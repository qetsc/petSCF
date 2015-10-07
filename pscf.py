
"""
e.g.:python qetsc.py -options_file options_qetsc
sample options_qetsc:
-N 100
-c 32 

-log_summary
#-malloc_info 

#-mat_view ::ascii_info 
#-mat_view draw -draw_pause -1
-mat_mumps_icntl_13 1 #turn off scaLAPACK for matrix inertia
-mat_mumps_icntl_24 1 #null pivot row detection for matrix inertia
-mat_mumps_icntl_23 0
-mat_mumps_icntl_28 2
-mat_mumps_icntl_29 1

-st_type sinvert 
-st_ksp_type preonly 
-st_pc_type cholesky 
-st_pc_factor_mat_solver_package mumps 

-eps_krylovschur_partitions 1  
-eps_interval 1,10
-eps_tol  1.e-8
-eps_krylovschur_detect_zeros TRUE
"""
import sys, petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc

Print = PETSc.Sys.Print

import numpy as np
try:
    import scipy.sparse
    import scipy.linalg
except:
    Print("no scipy modules")
    pass
import petsctools as pt
import slepctools as st
import PyQuante
import PyQuante.Basis.basis
import PyQuante.LA2
import PyQuante.hartree_fock

from mpi4py import MPI

def getChainMol(N=8,Z=1,d=1.):
    """
    Generates the coordinates of a simple chain of $N$ atoms (with an atomic number $Z$)  separated by the given distance $d$ in Angstroms.
    Creates a PyQuante Molecule object with the generated coordinates.
    """
    mol=[[Z,(0.,0.,i*d)] for i in range(N)]
    return PyQuante.Molecule('chain',mol,units='Angstrom')

def getWallTime(t0=0):
    """
    Returns the walltime - t0 in seconds.
    """
    t = MPI.Wtime() - t0
    if(t0): Print("{0} seconds".format(t))
    return t

def getNBF(atoms):
    "Returns the number of basis functions in an atom list"
    nbf = 0
    for atom in atoms: nbf += atom.nbf
    return nbf

def mkdens(c,nstart,nstop):
    """
    Not done yet
    """
    "Form a density matrix C*Ct given eigenvectors C[nstart:nstop,:]"
    d = c[:,nstart:nstop]
    Dmat = np.dot(d,d.T)
    return Dmat

def getTraceProduct(A,B):
    """
    Returns the trace of the product of A and B where A and B are same shape 2D numpy arrays.
    """
  #  if len(A) != len(B):print "length are not equal", len(A),len(B)
    N=A.shape[0]
    temp=0.0
    for i in xrange(N):
        for j in xrange(N):
            temp += A[i,j]*B[j,i]
    return temp

def getTraceProduct1(A,B):
    """
    Returns the trace of the product of A and B where A and B are same shape 2D numpy arrays.
    """
    #if len(A) != len(B):print "lengths are not equal", len(A),len(B)
    return sum(sum(A*B))

def getTraceProductCSR(A,B):
    """
    Returns the trace of the product of A and B where A and B are scipy CSR matrices.
    """
    nnzA=A.nnz
    nnzB=B.nnz
    tmp=0.
    if nnzA==nnzB:
        for i in xrange(nnzB):
            tmp+=A.data[i]*B.data[i]
    elif nnzB < nnzA:
        B=B.tocoo()
        for i in xrange(nnzB):
            tmp+= B.data[i]*A[B.row[i],B.col[i]]
    else:
        A=A.tocoo()
        for i in xrange(nnzA):
            tmp+= A.data[i]*B[A.row[i],A.col[i]]
    return tmp


def fetch_jints(Ints,i,j,nbf):
    """
    Not done yet
    """
    temp = np.zeros(nbf*nbf,'d')
    kl = 0
    for k in xrange(nbf):
        for l in xrange(nbf):
            index = intindex(i,j,k,l)
            temp[kl] = Ints[index]
            kl += 1
    return temp

def fetch_kints(Ints,i,j,nbf):
    temp = np.zeros(nbf*nbf,'d')
    kl = 0
    for k in xrange(nbf):
        for l in xrange(nbf):
            temp[kl] = Ints[intindex(i,k,j,l)]
            kl += 1
    return temp

def get2JmK(Ints,D):
    """
    Not done yet
    "Form the 2J-K integrals corresponding to a density matrix D"
    """

    nbf = D.shape[0]
    D1d = np.reshape(D,(nbf*nbf,)) #1D version of Dens
    G = np.zeros((nbf,nbf),'d')
    for i in xrange(nbf):
        for j in xrange(i+1):
            temp = 2*fetch_jints(Ints,i,j,nbf)-fetch_kints(Ints,i,j,nbf)
            G[i,j] = np.dot(temp,D1d)
            G[j,i] = G[i,j]
    return G

def rhf(qmol,basisset,spfilter,maxiter,scfthresh):
    """
    Not done yet
    """
    stage = PETSc.Log.Stage('S,h,Enuc')  
    stage.push();
    qbasis=PyQuante.Basis.basis.BasisSet(qmol,basisset)
    enuke = qmol.get_enuke()
    nclosed,nopen = qmol.get_closedopen()
    nocc = nclosed

    Print("number of basis functions: %i" % (len(qbasis)) )
    S,h,myints=PyQuante.Ints.getints(qbasis,qmol)
    stage.pop()
    
    stage = PETSc.Log.Stage('Aij conversion')  
    stage.push();
        #print np.amin(abs(S))
    S_Aij=convert2Aij(convert2csr(S,spfilter))
    h_Aij=convert2Aij(convert2csr(h,spfilter))
    stage.pop()

    stage = PETSc.Log.Stage('Solve(scipy)')  
    stage.push();
    orbe,orbs = scipy.linalg.eigh(h,S)
    print orbe
    stage.pop()

    stage = PETSc.Log.Stage('Solve(slepc)')  
    stage.push();
    orbe,orbs = solve_HEP(h_Aij,S_Aij)
    stage.pop()

    stage = PETSc.Log.Stage('SCF')  
    stage.push();
    eold = 0.
    for i in xrange(maxiter):
        D = mkdens(orbs,0,nocc)
        G = get2JmK(myints,D)
        F = h+G
        F_Aij=convert2Aij(convert2csr(F,spfilter))
       # orbe,orbs = PyQuante.LA2.geigh(F,S)
      #  orbe,orbs = scipy.linalg.eigh(F,S)
        orbe,orbs = solve_HEP(F_Aij,S_Aij)
        # MK: scipy.linalg.eig returns different results: maybe eigensolutions are not sorted?
      #  eone = PyQuante.LA2.trace2(D,h)
      #  etwo = PyQuante.LA2.trace2(D,F)
        eone = getTraceProduct1(D,h)
        etwo = getTraceProduct1(D,F)    
        energy = eone+etwo+enuke
        #print("%d %f" % (i,energy))
        if abs(energy-eold) < scfthresh: break
        print("Iteration: %d    Energy: %f    EnergyVar: %f"%(i,energy,abs(energy-eold)))
        eold = energy
    stage.pop()

    stage = PETSc.Log.Stage('Pyquante')  
    stage.push();
    en, orbe, orbs= PyQuante.hartree_fock.rhf(qmol,basis=basisset,DoAveraging = False)
    print en
    print orbe
    stage.pop()
    return

def getDensityMatrixReplicated(A,nel, B=None):
    """
    The way spectrum slicing works in SLEPc doesn't allow to start with a matrix in subcomm.
    That is the current version will not scale due to collective operations like MatCreateRedundantMatrix
    """
    rank=PETSc.COMM_WORLD.rank
    sizeA = A.getSize()
    Print("Matrix size: %i,%i" % (sizeA[0],A.getLocalSize()[1]))
    D     = A.duplicate()
    E = SLEPc.EPS().create(comm=A.getComm())
    E.setOperators(A,B)
    E.setProblemType(problem_type)
    E.setFromOptions()
    E.solve()
   # F,G = E.getOperators()
   # Print("Operator size: %i,%i" % (F.getSize()[0],F.getLocalSize()[1]))

    nslice = E.getKrylovSchurPartitions()
    nconv  = E.getConverged()
    islice, nconvlocal,Xmpi = E.getKrylovSchurSubcommInfo()
    Print("Operator size: %i,%i" % (Xmpi.getSize(),Xmpi.getLocalSize()))
  #  Dsub =  PETSc.Mat().createSBAIJ(sizeA,1,comm=Xmpi.getComm())
  #  Dsub.setUp()
    Vmpi = PETSc.Vec().createMPI(nslice,comm=PETSc.COMM_WORLD)
    Vmpi.setUp()
    Vmpi.set(0.0)
    Vmpi[islice] = nconvlocal
    Vmpi.assemble()
    V = pt.convert2SeqVec(Vmpi)
    sumV = V.sum()
    partsumV = V[:islice].sum()
    print rank,partsumV
    if sumV != nconv: Print("Total number of eigenvalues mismatch: %d %d" % (nconv,sumV) )
    for m in xrange(nconvlocal):
        if m + partsumV < nel / 2:
            print 'rank, m,nconv',rank,':',m,nconvlocal
            k = E.getKrylovSchurSubcommPairs(m, Xmpi)
            X = mat.convert2SeqVec(Xmpi)
            Istart, Iend = D.getOwnershipRange()
            for i in xrange(Istart, Iend):
                print 'assemble start rank:nconv, m,i',rank,':',nconvlocal,m,i
                cols = D.getRow(i)[0]
                print 'getrpwassemble start rank:nconv, m,i',rank,':',nconvlocal,m,i

                ncols = len(cols)
                values = [ 2.0 * X[i] * X[j] for j in cols]
                D.setValues(i, cols, values, addv=PETSc.InsertMode.ADD_VALUES)
                print 'sete start ra valunk:nconv, m,i',rank,':',nconvlocal,m,i

                D.assemble()
                print 'assemble end rank:nconv, m,i',rank,':',nconvlocal,m,i
        #error = E.computeError(m)
        #if error > 1.e-6: Print(" %12g" % (error)) 
    if k.imag != 0.0:
           Print("Complex eigenvalue dedected: %9f%+9f j  %12g" % (k.real, k.imag, error))
        
    return D


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

def getDistMat(basis,nbf):
    tmp=np.zeros([nbf,nbf])
    for i in xrange(nbf):
        atomi=basis[i].atom
        for j in xrange(i):
            tmp[i,j] = atomi.dist(basis[j].atom)
            tmp[j,i] = tmp[i,j]
    return tmp 

def getDistCSR(basis,nbf,maxdist=1E6):
    """
    Distance matrix that determines the nonzero structure of the Fock matrix
    TODO: Exploit symmetry, scipy has no support.
    """
    nnz=nbf*nbf # nbf*(nbf+1)/2.0
    row=np.zeros(nnz)
    col=np.zeros(nnz)
    val=np.zeros(nnz)
    k=0
    for i in xrange(nbf):
        atomi=basis[i].atom
        for j in xrange(nbf):
            distij=atomi.dist(basis[j].atom)
            if distij < maxdist:
                row[k]=i
                col[k]=j
                val[k]=distij
            k += 1
    row = row[val>=0.]        
    col = col[val>=0.]
    val = val[val>=0.] 
    A = scipy.sparse.csr_matrix((val,(row,col)), shape=(nbf,nbf))
    return  A


       

def getDistAIJ(basis,nbf,maxdist=1E6,matcomm=PETSc.COMM_SELF):
    """
    Distance matrix that determines the nonzero structure of the Fock matrix
    """
    A = PETSc.Mat().createSBAIJ(nbf,1,comm=matcomm)
    A.setUp()
    rstart, rend = A.getOwnershipRange()
    for i in xrange(rstart,rend):
        atomi=basis[i].atom
        A.setValue(i,i,0.,addv=PETSc.InsertMode.INSERT)
        for j in xrange(i+1,nbf):
            distij=atomi.dist(basis[j].atom)
            if distij < maxdist: A.setValue(i,j,distij,addv=PETSc.InsertMode.INSERT)
    A.assemble()        
    return  A

def getTempAIJ(basis,maxdist,maxnnz=[0],bandwidth=[0],matcomm=PETSc.COMM_SELF):
    """
    Template matrix that determines the nonzero structure of the Fock matrix.
    It is used only as a template for the nonzero pattern.
    maxnnz: max number of nonzeros per row. If it is given performance might improve
    TODO:
    Distance calculation loop should be over atoms indeed, which will improve performance by nbf/natom.
    Better to preallocate based on diagonal and offdiagonal nonzeros.
    Maybe setDiagonal more efficiently
    """
    nbf      = len(basis)
    maxdist2 = maxdist * maxdist
    A        = PETSc.Mat().create(comm=matcomm)
    A.setType('sbaij') 
    A.setSizes([nbf,nbf]) 
    if any(maxnnz): A.setPreallocationNNZ(maxnnz) 
    A.setUp()
    #A.setDiagonal(1.0) # TypeError: Argument 'diag' has incorrect type (expected petsc4py.PETSc.Vec, got float)
    rstart, rend = A.getOwnershipRange()
    if any(bandwidth):
        if len(bandwidth)==1: bandwidth=np.array([bandwidth]*nbf)
        for i in xrange(rstart,rend):
            atomi=basis[i].atom
            A.setValue(i,i,0.0,addv=PETSc.InsertMode.INSERT)
            for j in xrange(i+1,min(i+bandwidth[i],nbf)):
                distij2 = atomi.dist2(basis[j].atom)
                if distij2 < maxdist2: A.setValue(i,j,distij2,addv=PETSc.InsertMode.INSERT)
    else:
        for i in xrange(rstart,rend):
            atomi=basis[i].atom
            A.setValue(i,i,0.0,addv=PETSc.InsertMode.INSERT)
            for j in xrange(i+1,nbf):
                distij2=atomi.dist2(basis[j].atom)
                if distij2 < maxdist2: A.setValue(i,j,distij2,addv=PETSc.InsertMode.INSERT)                    
    A.assemble()        
    return  A
       
def getGuessDAIJold(basis):
    """
    Returns the guess (initial) density matrix.
    A very simple guess is used which a diagonal matrix containing atomic charge divided by number of basis functions for the atom.
    """
    nbf=len(basis)
    A = PETSc.Mat().createSBAIJ(nbf,1)
    A.setUp()
    rstart, rend = A.getOwnershipRange() 
    for i in xrange(rstart,rend):
        atomi=basis[i].atom
        if atomi.atno == 1: A[i,i] = atomi.Z/1.
        else:               A[i,i] = atomi.Z/4.
    A.assemble()        
    return  A   

def getGuessDAIJ(basis,guess=0,T=None,matcomm=PETSc.COMM_SELF):
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
        d=1.e-5
        if T:
            A=T.duplicate()
            for i in xrange(rstart,rend):
                k=0
                atomi=basis[i].atom
                cols,vals = T.getRow(i) 
                for j in cols:
                    if i==j:
                        if atomi.atno == 1: vals[0] = atomi.Z/1.
                        else:               vals[0] = atomi.Z/4.
                    else:    
                        vals[k] = d * vals[k]
                    k += 1
                A.setValues(row,cols,vals,addv=PETSc.InsertMode.INSERT)                  
            A.assemble()
        else:
            Print("You need to give a template matrix for guess type {0}".format(guess))            
    return  A 

def getF0AIJ(atoms,basis,B):
    "Form the zero-iteration (density matrix independent) Fock matrix"
    nat = len(atoms)
    nbf = len(basis)
    A = B.duplicate()
    nuclear_attraction = getNuclearAttraction(atoms,nbf)
    A.setUp()
    rstart, rend = A.getOwnershipRange()
    for row in xrange(rstart,rend):
        basisi=basis[row]
        atomi=basisi.atom
        na = nuclear_attraction[row] 
        u_term = basisi.u
        cols,vals = B.getRow(row)
        k=0
        for col in cols:
            basisj=basis[col]
            atomj=basisj.atom
            if row == col:
                vals[k] = u_term + na 
            else:
                betaij = PyQuante.MINDO3.get_beta0(atomi.atno,atomj.atno)
                Sij = basisi.cgbf.overlap(basisj.cgbf)
                IPij = basisi.ip + basisj.ip    
                vals[k] = betaij * IPij * Sij 
            k += 1    
        A.setValues(row,cols,vals[0:k],addv=PETSc.InsertMode.INSERT)    
    A.assemble()
    return A

def getF1AIJ(atoms,basis,D, Ddiag, B=None):
    """
    One-center corrections to the core fock matrix
    Block diagonal matrix.
    Number of blocks = number of atoms
    Block size = number of basis functions per atom
    Each block can be computed independently
    TODO: symmetry
    """
    nat = len(atoms)
    nbf = len(basis)
    blocklist=['']*nat
    ibf=0
    count=0
    
    if B:
        A = B.duplicate()
    else:
        A = D.duplicate()    
    A.zeroEntries()
    rstart, rend = A.getOwnershipRange()
    ibf = 0 # bf number of the first bfn on iat
    for iat in xrange(nat):
        atomi = atoms[iat]
      #  valij = 0.
        for i in xrange(atomi.nbf):
          #  if ibf+i in range(rstart,rend):    
            bfi = atomi.basis[i]
            gii = PyQuante.MINDO3.get_g(bfi,bfi)
            qi =  Ddiag[ibf+i] 
            val_ibfplusi = 0.5*qi*gii
            valij = 0.
            for j in xrange(atomi.nbf):  # ij on same atom
                if j != i:
                    bfj = atomi.basis[j]
                    qj = Ddiag[ibf+j]
                    gij = PyQuante.MINDO3.get_g(bfi,bfj)
                    pij = 0
                    if ibf+i in range(rstart,rend): pij = D[ibf+i,ibf+j]
                    hij = PyQuante.MINDO3.get_h(bfi,bfj)
                    # the following 0.5 is something of a kludge to match
                    #  the mopac results.
                    val_ibfplusi += qj*gij - 0.5*qj*hij
                    valij = 0.5*pij*(3*hij-gij)
                    if j>=i and ibf+i in range(rstart,rend) : A.setValue(ibf+i,ibf+j,valij,addv=PETSc.InsertMode.ADD_VALUES)
            if ibf+i in range(rstart,rend):A.setValue(ibf+i,ibf+i,val_ibfplusi,addv=PETSc.InsertMode.ADD_VALUES)        
        ibf += atomi.nbf
    A.assemble()
    return A

def getF2AIJold(atoms, D, Ddiag, B=None,maxdist=1.E6):
    "Two-center corrections to the core fock matrix"
    nbf = getNBF(atoms)
    nat = len(atoms)
    if B:
        A = B.duplicate()
    else:
        A = D.duplicate()
    A.zeroEntries()    
      
    #x=np.zeros(nbf)    
    rstart, rend = A.getOwnershipRange()
    x = PETSc.Vec()
    x.createSeq(rend-rstart,comm=PETSc.COMM_SELF)
    x.set(0.0)
    A.setUp()
    ibf = 0 # bf number of the first bfn on iat
    for iat in xrange(nat):
        atomi = atoms[iat]
        jbf = 0
        if ibf in xrange(rstart,rend):
            for jat in xrange(nat):
                atomj = atoms[jat]
                if iat != jat and atomi.dist(atomj) < maxdist:
                    gammaij = PyQuante.MINDO3.get_gamma(atomi,atomj)
                    for i in xrange(atomi.nbf):
                       # if ibf+i in xrange(rstart,rend):
                            rowi=ibf+i
                            qi = Ddiag[rowi]
                            qj = 0
                            #valij=0.0
                            for j in xrange(atomj.nbf):
                                
                                colj=jbf+j
                                pij = D[rowi,colj]
                                valij = -0.5*pij*gammaij #MK 0.25 --> 0.5
                                qj += Ddiag[colj]
                                if colj in xrange(rstart,rend): x[colj] += 0.5*qi*gammaij 
                                if ibf+i <= jbf+j: A.setValue(rowi,colj,valij,addv=PETSc.InsertMode.ADD_VALUES)
                            x[ibf+i] += 0.5 * qj*gammaij
                            
                jbf += atomj.nbf
            ibf += atomi.nbf
            
    A.assemblyBegin()
    x.assemblyBegin()
    A.assemblyEnd()
    x.assemblyEnd()
    A.setDiagonal(x,addv=PETSc.InsertMode.ADD_VALUES)       
    return A

def getF2AIJ(atoms, D, Ddiag, B=None,maxdist=1.E6):
    "Two-center corrections to the core fock matrix"
    nbf = getNBF(atoms)
    nat = len(atoms)
    if B:
        A = B.duplicate()
    else:
        A = D.duplicate()
    A.zeroEntries()    
      
    #x=np.zeros(nbf)    
    rstart, rend = A.getOwnershipRange()
    #x = PETSc.Vec()
    #x.createSeq(rend-rstart,comm=PETSc.COMM_SELF)
    x=A.createVecLeft()
    x.set(0.0)
    #x=np.zeros(rend-rstart)
    A.setUp()
    ibf = 0 # bf number of the first bfn on iat
    for iat in xrange(nat):
        atomi = atoms[iat]
        jbf = 0
       # if ibf in xrange(rstart,rend):
        for jat in xrange(nat):
            atomj = atoms[jat]
            if iat != jat and atomi.dist(atomj) < maxdist:
                gammaij = PyQuante.MINDO3.get_gamma(atomi,atomj)
              #  x.set(0.0)
                for i in xrange(atomi.nbf):
                    if ibf+i in xrange(rstart,rend):
                        rowi=ibf+i
                        qi = Ddiag[rowi]
                        qj = 0
                        #valij=0.0
                        tmp=0
                        for j in xrange(atomj.nbf):
                            
                            colj=jbf+j
                            pij = D[rowi,colj]
                            valij = -0.5*pij*gammaij #MK 0.25 --> 0.5
                            qj += Ddiag[colj]
                            
                            tmp += 0.5*qi*gammaij
                           # if colj in xrange(rstart,rend): x[colj] += 0.5*qi*gammaij
                            if rowi <= colj: A.setValue(rowi,colj,valij,addv=PETSc.InsertMode.ADD_VALUES)
                        x[rowi] += 1.0 * qj*gammaij #MK 0.5 --> 1.0
                       # print PETSc.COMM_WORLD.rank,'xrowi',rowi,x[rowi]
                        
            jbf += atomj.nbf
        ibf += atomi.nbf
            
    A.assemblyBegin()
    x.assemblyBegin()
    A.assemblyEnd()
    x.assemblyEnd()
    A.setDiagonal(x,addv=PETSc.InsertMode.ADD_VALUES)       
    return A

def getNuclearAttraction(atoms,nbf):
    """
    Returns the nuclear interaction as implemented in PyQuante.MINDO3
    """
    nat=len(atoms)
    tmp=np.zeros(nbf)
    ibf = 0 # bf number of the first bfn on iat
    for iat in xrange(nat):
        atomi = atoms[iat]
        for jat in xrange(nat):
            atomj = atoms[jat]
            if iat != jat: 
                gammaij = PyQuante.MINDO3.get_gamma(atomi,atomj)
                for i in xrange(atomi.nbf):
                    tmp[ibf+i] -= gammaij*atomj.Z
        ibf += atomi.nbf
    return tmp

def getDij(eigvecs,i,j):
    """
    Returns the i,j element of a density matrix from given eigenvectors.
    """
    tmp=0.
    for k in len(eigvecs[0,:]):
        tmp+=eigvecs[i,k]*eigvecs[j,k]
    return tmp


def getDCSR(eigvecs,CSR,guess=False):
    """
    Returns the density matrix from given eigenvectors.
    """
    nbf = CSR.shape[0]
    nnz = CSR.nnz
    DCSR = scipy.sparse.csr_matrix(CSR)
    Dptr=DCSR.indptr
    Dcol=DCSR.indices
    Ddata=DCSR.data
    Dnnzrow=np.diff(Dptr)
    k=0
    count=0
    for row in xrange(nbf):
        for i in xrange(Dnnzrow[row]):
            Ddata[k]=0.0
            col = Dcol[k]
            tmp=0
            for m in xrange(len(eigvecs[0,:])):
                Ddata[k] += eigvecs[row,m] * eigvecs[col,m]
            k = k + 1 
    return DCSR

def getGuessDCSR(basis):
    """
    Returns the simple guess Density matrix (identity matrix) in CSR format.
    """
    nbf=len(basis)
    DCSR = scipy.sparse.identity(nbf,dtype='d',format='csr')
    for i in xrange(nbf):
        atomi = basis[i].atom
        if atomi.atno == 1:
            DCSR[i,i] = atomi.Z/1.
        else:                      
            DCSR[i,i] = atomi.Z/4.
    return DCSR         

def get_guess_D_diag(basis,nbf):
    """
    Returns the simple guess Density matrix (identity matrix) as a numpy array.
    """
    D=np.zeros(nbf)
    for i in xrange(nbf):
        atomi = basis[i].atom
        if atomi.atno == 1:
            D[i] = atomi.Z/1.
        else:                      
            D[i] = atomi.Z/4.
    return D 
    
def getF0CSR(atoms,basis,nnzCSR):
    "Form the zero-iteration (density matrix independent) Fock matrix"
    nat = len(atoms)
    nbf = len(basis)
    nnz = nnzCSR.nnz
    F0CSR = scipy.sparse.csr_matrix(nnzCSR)

    F0ptr=F0CSR.indptr
    F0col=F0CSR.indices
    F0data=F0CSR.data
    F0nnzrow=np.diff(F0ptr)
    k=0
    count=0
    nuclear_attraction = getNuclearAttraction(atoms,nbf)
    """
    Alternative for loop over nnz, probably less efficient?
    for i in xrange(F0CSR.nnz):
        row = np.amax(np.where(F0ptr<=i))
        col = F0col[i]
    """
    for row in xrange(nbf):
        basisi=basis[row]
        atomi=basisi.atom
        na = nuclear_attraction[row] 
        u_term = basisi.u
        for i in xrange(F0nnzrow[row]):
            F0data[k]=0.0
            col = F0col[k]
            basisj=basis[col]
            atomj=basisj.atom
            if row == col:
                F0data[k] = u_term + na 
            else:
                betaij = PyQuante.MINDO3.get_beta0(atomi.atno,atomj.atno)
                Sij = basisi.cgbf.overlap(basisj.cgbf)
                IPij = basisi.ip + basisj.ip    
                F0data[k] = betaij * IPij * Sij 
            k = k + 1    
    return F0CSR  

def getF1CSR(atoms,basis,D):
    """
    One-center corrections to the core fock matrix
    Block diagonal matrix.
    Number of blocks = number of atoms
    Block size = number of basis functions per atom
    Each block can be computed independently
    TODO: symmetry
    """
    nat = len(atoms)
    nbf = len(basis)
    blocklist=['']*nat
    ibf=0
    count=0
    for iat in xrange(nat):
        atomi = atoms[iat]
        inbf=atomi.nbf
        block=np.zeros((inbf,inbf))
        for i in xrange(inbf):
            bfi = atomi.basis[i]
            gii = PyQuante.MINDO3.get_g(bfi,bfi)
            qi =  D[ibf+i,ibf+i]
            block[i,i] = 0.5*qi*gii
            for j in xrange(inbf):  # ij on same atom
                if j != i:
                    bfj = atomi.basis[j]
                    qj = D[ibf+j,ibf+j]
                    gij = PyQuante.MINDO3.get_g(bfi,bfj)
                    pij = D[ibf+i,ibf+j]
                    hij = PyQuante.MINDO3.get_h(bfi,bfj)
                    # the following 0.5 is something of a kludge to match
                    #  the mopac results.
                    block[i,i] += qj*gij - 0.5*qj*hij
                    block[i,j] = 0.5*pij*(3*hij-gij)
        ibf += atomi.nbf
        blocklist[iat]=block
    return scipy.sparse.block_diag(blocklist,format='csr',dtype='d') 

def getF2CSR(atoms,D,csrMat,maxdist=1.E6):
    "Two-center corrections to the core fock matrix"
    nbf = getNBF(atoms)
    nat = len(atoms)

    F2 = csrMat
    F2.data=np.zeros(F2.nnz)

    ibf = 0 # bf number of the first bfn on iat
    for iat in xrange(nat):
        atomi = atoms[iat]
        jbf = 0
        for jat in xrange(nat):
            atomj = atoms[jat]
            if iat != jat and atomi.dist(atomj) < maxdist:
                gammaij = PyQuante.MINDO3.get_gamma(atomi,atomj)
                for i in xrange(atomi.nbf):
                #    print 'OO ibf+i,ibf+i',ibf+i,ibf+i

                    qi = D[ibf+i,ibf+i]
                    qj = 0
                    for j in xrange(atomj.nbf):
                   #     print '-- ibf+i,jbf+j',ibf+i,jbf+j

                        pij = D[ibf+i,jbf+j]
                        F2[ibf+i,jbf+j] -= 0.25*pij*gammaij
                        F2[jbf+j,ibf+i] = F2[ibf+i,jbf+j]
                        qj += D[jbf+j,jbf+j]
                        F2[jbf+j,jbf+j] += 0.5*qi*gammaij
                 #       print '!!jbf+j,F2[jbf+j,jbf+j]',jbf+j,F2[jbf+j,jbf+j]
                    F2[ibf+i,ibf+i] += 0.5*qj*gammaij
                 #   print '**ibf+i,F2[ibf+i,ibf+i]',ibf+i,F2[ibf+i,ibf+i]
            jbf += atomj.nbf
        ibf += atomi.nbf
    return F2




def mndo3(qmol,spfilter,maxiter,scfthresh):
    import PyQuante.MINDO3_Parameters
    import PyQuante.MINDO3
    qmol=PyQuante.MINDO3.initialize(qmol)
    Enuke = PyQuante.MINDO3.get_enuke(qmol)
    nbf = getNBF(qmol)    
    nel = PyQuante.MINDO3.get_nel(qmol.atoms)
    Eref = PyQuante.MINDO3.get_reference_energy(qmol)
    basis = getBasis(qmol, nbf)
    F0 = PyQuante.MINDO3.get_F0(qmol)
    print 'nbf,nel',nbf,nel
    
    F0_Aij=convert2Aij(convert2csr(F0, spfilter))
    D = PyQuante.MINDO3.get_guess_D(qmol)
    print D

    Eold = 0
    for i in xrange(maxiter):
        F1 = PyQuante.MINDO3.get_F1(qmol,D)
        F2 = PyQuante.MINDO3.get_F2(qmol,D)
        F = F0+F1+F2
        F1_Aij=convert2Aij(convert2csr(F1, spfilter))
        F2_Aij=convert2Aij(convert2csr(F2, spfilter))
        Print("F")
        F_Aij=convert2Aij(convert2csr(F, spfilter))

        Eel = 0.5*getTraceProduct1(D,F0+F)
        print Eel
        if abs(Eel-Eold) < scfthresh:
            print "Exiting because converged",i+1,Eel,Eold
            break
        Eold = Eel
        orbe,orbs = scipy.linalg.eigh(F)
      #  orbe,orbs = solve_HEP(F_Aij,problem_type=SLEPc.EPS.ProblemType.HEP)
        print orbe
        D = 2*mkdens(orbs,0,nel/2)
        Print("D")
        print D
        D_Aij=convert2Aij(convert2csr(D, spfilter))
    print orbe        
    Etot = Eel+Enuke
    print "mndo",Etot,Etot*PyQuante.Constants.ev2kcal+Eref
    print PyQuante.MINDO3.scf(qmol)
    return

def mndo4(qmol,spfilter,maxiter,scfthresh,maxdist=1.E6):
    import PyQuante.MINDO3_Parameters
    import PyQuante.MINDO3
    import scipy.linalg 
    import scipy.sparse
    stage = PETSc.Log.Stage('Initial')  
    stage.push()
    qmol  = PyQuante.MINDO3.initialize(qmol)
    atoms = qmol.atoms
    Enuke = PyQuante.MINDO3.get_enuke(qmol)
    Eref  = PyQuante.MINDO3.get_reference_energy(qmol)
    nbf   = getNBF(qmol)    
    nel   = PyQuante.MINDO3.get_nel(atoms)
    basis = getBasis(qmol, nbf)
    CSR   = getDistCSR(basis, nbf,maxdist=maxdist)
    F0    = getF0CSR(atoms, basis, CSR)
    D     = getGuessDCSR(basis)
    Eold  = 0
    Print("Number of basis functions: %i" % (nbf))
    Print("Number of valance electrons: %i" % (nel))
    Print("Nuclear repulsion: %f" % (Enuke))
    Print("Reference energy: %f" % (Eref))
    Print("Nonzero density: %i" % (100.*CSR.nnz/nbf/nbf))
    stage.pop()
    for i in xrange(maxiter):
        stage = PETSc.Log.Stage('F1'); stage.push();
        F1    = getF1CSR(atoms, basis, D)
        stage.pop(); stage = PETSc.Log.Stage('F2');    stage.push();
        F2    = getF2CSR(atoms, D, CSR, maxdist)
        stage.pop(); stage = PETSc.Log.Stage('Trace'); stage.push();
        F     = F0+F1+F2
        Eel   = 0.5*getTraceProductCSR(D,F0+F)
        if abs(Eel-Eold) < scfthresh:
            Print("Converged at iteration %i" % (i+1))
            break
        Eold = Eel
        F_AIJ=convert2Aij(F)
        stage.pop(); stage = PETSc.Log.Stage('Diag'); stage.push();
        orbe,orbs = scipy.linalg.eigh(F.todense())
      #  orbe,orbs = solve_HEP(F_AIJ,problem_type=SLEPc.EPS.ProblemType.HEP)
        D = 2*getDCSR(orbs[:,0:nel/2], CSR)
        stage.pop(); stage = PETSc.Log.Stage('D'); stage.push();
        stage.pop()
    stage = PETSc.Log.Stage('PyQuante'); stage.push();  
    Etot  = Eel+Enuke
    ihomo=nel/2-1
    ilumo=nel/2
    Print("Max,homo,lumo,min eigenvalues: %f,%f,%f,%f" % (orbe[0],orbe[ihomo],orbe[ilumo],orbe[-1]))
    Print("QETSc: %f, %f" % (Eel,Etot*PyQuante.Constants.ev2kcal+Eref))
    Print('PyQuante')
    Print('%f' % (PyQuante.MINDO3.scf(qmol)))
    stage.pop()
    return

def mindo3AIJ(qmol,spfilter,maxiter,scfthresh,maxnnz=[0],bandwidth=[0],maxdist=1.E6,uniform=True,guess=0, solve=0, debug=False):
    import PyQuante.MINDO3_Parameters
    import PyQuante.MINDO3
    import constants as const
    stage = pt.getStage(stagename='Initialize')
    qmol  = PyQuante.MINDO3.initialize(qmol)
    atoms = qmol.atoms
    stage = pt.getStage(stagename='Enuclear', oldstage=stage)
    Enuke = PyQuante.MINDO3.get_enuke(qmol)
    Print("Enuc           = {0:20.10f} kcal/mol = {1:20.10f} ev = {2:20.10f} Hartree".format(Enuke*const.ev2kcal,Enuke,Enuke*const.ev2hartree))
    stage = pt.getStage(stagename='Ereference', oldstage=stage)
    Eref  = PyQuante.MINDO3.get_reference_energy(qmol)
    Print("Eref           = {0:20.10f} kcal/mol = {1:20.10f} ev = {2:20.10f} Hartree".format(Eref, Eref*const.kcal2ev, Eref*const.kcal2hartree))
    stage = pt.getStage(stagename='getNBF,get_nbl', oldstage=stage)
    nbf   = getNBF(qmol)    
    nel   = PyQuante.MINDO3.get_nel(atoms)
    nocc  = nel/2
    basis = getBasis(qmol, nbf)
    Print("Number of basis functions: %i" % (nbf))
    Print("Number of valance electrons: %i" % (nel))
    if not (all(maxnnz) or all(bandwidth)):
      ##D  stage = pt.getStage(stagename='DistAIJ', oldstage=stage)
      ##D  B     = getDistAIJ(basis, nbf, maxdist=maxdist, matcomm=PETSc.COMM_WORLD)
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
        stage = pt.getStage(stagename='getTempAIJ', oldstage=stage)
        B     = getTempAIJ(basis, maxdist,maxnnz=nnzarray, bandwidth=bwarray, matcomm=PETSc.COMM_WORLD)
    else:
        stage = pt.getStage(stagename='getTempAIJ', oldstage=stage)
        B     = getTempAIJ(basis, maxdist,maxnnz=maxnnz, bandwidth=bandwidth, matcomm=PETSc.COMM_WORLD)
    stage = pt.getStage(stagename='F0', oldstage=stage)
    F0    = getF0AIJ(atoms, basis, B)
    stage = pt.getStage(stagename='D0', oldstage=stage)
    D     = getGuessDAIJ(basis,guess=guess,matcomm=PETSc.COMM_WORLD)
    stage = pt.getStage(stagename='Ddiag', oldstage=stage)
    Ddiag = pt.convert2SeqVec(D.getDiagonal()) 
    if debug:
        BCSR  = getDistCSR(basis, nbf,maxdist=maxdist)
        F0CSR = getF0CSR(atoms, basis, BCSR)
        #mat.compareAIJB(B,BCSR,nbf)
        DCSR   = getGuessDCSR(basis)
        Print("Nonzero density: %i" % (100.*BCSR.nnz/nbf/nbf))
    Eel   = 0.
    Eold  = 0.
    for i in xrange(maxiter):
        Print("****************************Iteration {0}****************************".format(i))
        stage = pt.getStage(stagename='F1', oldstage=stage)
        F1    = getF1AIJ(atoms, basis, D, Ddiag, B=B)
        stage = pt.getStage(stagename='F2', oldstage=stage)
        F2    = getF2AIJ(atoms, D, Ddiag, maxdist=maxdist, B=B)
        stage = pt.getStage(stagename='F0+F1+F2', oldstage=stage)
        F     = F0+F1+F2
        stage = pt.getStage(stagename='Trace', oldstage=stage)
        Eel   = 0.5*pt.getTraceProductAIJslow(D,F0+F)
        Print("Eel            = {0:20.10f} kcal/mol = {1:20.10f} ev = {2:20.10f} Hartree".format(Eel*const.ev2kcal,Eel,Eel*const.ev2hartree))

        if debug:
            F1CSR    = getF1CSR(atoms, basis, DCSR)
            print 'F1 comp', i
            pt.compareAIJB(F1,F1CSR,nbf)
            F2CSR    = getF2CSR(atoms, DCSR, BCSR)
            print 'F2 comp', i  
            pt.compareAIJB(F2,F2CSR,nbf)             
            FCSR  = F0CSR+F1CSR+F2CSR
            #mat.compareAIJB(F,FCSR,nbf) 
            #mat.compareAIJB(F0+F,F0CSR+FCSR,nbf)            
            print 'Eel',Eel,0.5*getTraceProductCSR(DCSR,F0CSR+FCSR)
        if abs(Eel-Eold) < scfthresh:
            Print("Converged at iteration %i" % (i+1))
            break

        Eold = Eel
        if solve > 0:      
            t0 = getWallTime()
            stage = pt.getStage(stagename='Solve', oldstage=stage)
            if uniform or i<2:
                EPS, nconv = st.solveEPS(F)
            else:
                nsubint=st.getNumberOfSubIntervals(EPS)
                subint = st.getSubIntervals(eigarray,nsubint) 
                EPS, nconv = st.solveEPS(F,subintervals=subint)   
            getWallTime(t0)
            if nconv+1 > nocc:
                Print("{0} eigenvalues converged".format(nconv))
            else:
                Print("{0} eigenvalues found, {1} eigenvalues are required".format(nconv,nocc))
                sys.exit()        
            stage = pt.getStage(stagename='Density', oldstage=stage)
            t0 = getWallTime()
            D,eigarray = st.getDensityMatrix(EPS,B, nocc)
            getWallTime(t0)
           # Print("{0} seconds for iter {1} in density".format(t,i))
          #  Print("{0}, {1}, {2} ".format(eigarray[0],eigarray[nocc-1],eigarray[nocc]))
            Print("Range of eigenvalues {0} - {1} ".format(eigarray[0],eigarray[nocc-1]))

        Ddiag=pt.convert2SeqVec(D.getDiagonal()) 
        if debug:
            orbe,orbs = scipy.linalg.eigh(FCSR.todense())
            DCSR = 2*getDCSR(orbs[:,0:nel/2], BCSR)
            print 'D comp', i
            pt.compareAIJB(D,DCSR,nbf)
    Etot   = Eel+Enuke
    Efinal = Etot*const.ev2kcal+Eref
    Print("Enuc           = {0:20.10f} kcal/mol = {1:20.10f} ev = {2:20.10f} Hartree".format(Enuke*const.ev2kcal,Enuke,Enuke*const.ev2hartree))
    Print("Eref           = {0:20.10f} kcal/mol = {1:20.10f} ev = {2:20.10f} Hartree".format(Eref, Eref*const.kcal2ev, Eref*const.kcal2hartree))
    Print("Eel            = {0:20.10f} kcal/mol = {1:20.10f} ev = {2:20.10f} Hartree".format(Eel*const.ev2kcal,Eel,Eel*const.ev2hartree))
    Print("Eel+Enuke      = {0:20.10f} kcal/mol = {1:20.10f} ev = {2:20.10f} Hartree".format(Etot*const.ev2kcal,Etot,Etot*const.ev2hartree))
    Print("Eel+Enuke+Eref = {0:20.10f} kcal/mol = {1:20.10f} ev = {2:20.10f} Hartree".format(Efinal, Efinal*const.kcal2ev,Efinal*const.kcal2hartree))

    return



def main():
    import PyQuante.IO.XYZ
    import os.path
    import xyztools as xt
    stage = pt.getStage('Input')  
    opts = PETSc.Options()
    mol = opts.getString('mol','')
    xyzfile = opts.getString('xyz','')
    maxdist = opts.getReal('maxdist', 1.e6)
    maxiter = opts.getInt('maxiter', 30)
    analysis = opts.getInt('analysis', 0)
    solve = opts.getInt('solve', 0)
    maxnnz = opts.getInt('maxnnz', 0)
    guess = opts.getInt('guess', 0)
    bandwidth = opts.getInt('bw', 0)
    sort = opts.getInt('sort', 0)
    basisset = opts.getString('basis','sto-3g')
    method = opts.getString('method','mindo3')
    scfthresh = opts.getReal('scfthresh',1.e-5)
    spfilter = opts.getReal('spfilter',0.)
    uniform = opts.getBool('uniform',True)
    
    if mol:
        import PyQuante.Molecule 

        Print('xyz from mol input:{0}'.format(mol))  
        qmol=PyQuante.Molecule(mol)
    elif os.path.isfile(xyzfile):
        Print('xyz read from file:{0}'.format(xyzfile))
        if sort > 0:
            stage = pt.getStage('Sort',oldstage=stage)  
            xyz        = xt.readXYZ(xyzfile)
            sortedxyz  = xt.sortXYZ(xyz)
            sortedfile = xt.writeXYZ(sortedxyz)
            Print('sorted xyz file:{0}'.format(sortedfile))
            qmol = xt.xyz2PyQuanteMol(sortedxyz)
        else:   
            qmol = xt.xyzFile2PyQuanteMol(xyzfile)
    else:
        Print("%s file not found" %(xyzfile))
        N = opts.getInt('N', 32)
        c = opts.getInt('c', 3)
        Z = opts.getInt('Z', 8)
        dist = opts.getInt('dist', 0.712) 
        Print("A chain of atoms will be used.")
        qmol=getChainMol(N=N, Ze=Z, d=dist)
    Print("Number of atoms: %i" % (len(qmol.atoms)))
    Print("Number of electrons: %i" % (qmol.get_nel()))
    stage.pop()
    if method == 'file':
        fA = opts.getString('fA')
        fB = opts.getString('fB')
        fA = '/Volumes/s/matrices/petscbinary/nanotube2e-r_P2_A'
        fB = '/Volumes/s/matrices/petscbinary/nanotube2e-r_P2_B'
        A  = pt.getMatFromFile(fA,comm=PETSc.COMM_WORLD)
        B  = pt.getMatFromFile(fB,comm=PETSc.COMM_WORLD)
        st.getEigenSolutions(A,B)
    elif method == "sparsity":
        import PyQuante.MINDO3
        stage = pt.getStage(stagename='Initialize',oldstage=stage)
        qmol  = PyQuante.MINDO3.initialize(qmol)
        nbf   = getNBF(qmol)    
        basis = getBasis(qmol, nbf)
        stage = pt.getStage(stagename='distCSR',oldstage=stage)
        BCSR  = getDistCSR(basis, nbf,maxdist=maxdist)
        stage = pt.getStage(stagename='getSparsityInfo',oldstage=stage)
        pt.getSparsityInfo(BCSR, nbf, maxdensity)
        stage.pop()

    elif method == "HF":
        rhf(qmol,basisset,spfilter,maxiter,scfthresh,maxdist=maxdist)
    elif method == 'mindo3AIJ':
        Print("MINDO/3 calculation starts...")
        mindo3AIJ(qmol,spfilter,maxiter,scfthresh,maxnnz=[maxnnz],bandwidth=[bandwidth],maxdist=maxdist,uniform=uniform,guess=guess, solve=solve, debug=False)
        Print("MINDO/3 calculation finishes.")
     #   stage = PETSc.Log.Stage('PyQuante'); stage.push();  
      #  Print('PyQuante')
      #  Print('%f' % (PyQuante.MINDO3.scf(qmol)))
      #  stage.pop()
    else:
        Print("No valid method specified")

if __name__ == '__main__':
    main()

  
