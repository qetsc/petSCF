from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np

Print = PETSc.Sys.Print

def getNumberOfSubIntervals(eps):
    return eps.getKrylovSchurPartitions()

def getClusters(eigs, chtresh=1.e-6):
    """
    Given a list of eigenvalues and a threshold for clustering,
    returns an array of clusters, and the multiplicities
    of each cluster.
    Input:
    eigs    - numpy array (dtype='float64')
    chtresh - float (optional)
    Returns:
            - numpy array (dtype='float64')
            - numpy array (dtpye='int32')
    """
    eigs = sorted(np.array(eigs))
    neigs          = len(eigs)
    clusters       = np.zeros(neigs)
    multiplicities = np.ones(neigs,dtype='int32')
    clusters[0]    = eigs[0]
    icluster       = 0
    for i in range(1,neigs):
        if abs(eigs[i]-clusters[icluster]) < chtresh:
            multiplicities[icluster] += 1
        else:
            icluster += 1
            clusters[icluster] = eigs[i]
    ncluster = icluster + 1
    return clusters[0:ncluster], multiplicities[0:ncluster]

def getSubIntervals(eigs, nsub, sbuffer=0.1,interval=[0],cthresh=1.e-6):
    """
    Given a list of eigenvalues, (eigs) and number of subintervals (nsub), 
    returns the boundaries for subintervals such that each subinterval has an average number of eigenvalues.
    Doesn't skip gaps, SLEPc doesn't support it, yet.
    range of eigs * sbuffer gives a sbuffer zone for leftmost and rightmost boundaries.
    """
    eigs, mults = getClusters(eigs,cthresh) 
    neigs = len(eigs)
    mean = neigs / nsub
    remainder = neigs % nsub
    erange = eigs[-1] - eigs[0]
    isbuffer = erange * sbuffer
    subint = np.zeros(nsub + 1)
    subint[0] = eigs[0] - isbuffer
    for i in xrange(1, nsub):
        subint[i] = (eigs[mean * i] + eigs[mean * i - 1]) / 2.
        if remainder > 0 and i > 1:
            subint[i] = (eigs[mean * i + 1] + eigs[mean * i]) / 2.
            remainder = remainder - 1
    subint[nsub] = eigs[-1] + isbuffer
    if len(interval)==2:
        subint[0]  = interval[0]
        subint[-1] = interval[1]
    return subint, mults

def getDensityMatrix(eps,T,nocc):
    """
    nocc  = Number of occupied orbitals = N_e / 2 = Number of electrons / 2
    D = One-electron density matrix. The sparsity pattern is the same as T.
    eigarray = An array of length N_o, containing the eigenvalues 
    D_{\mu\nu} = 2 \sum_i^N_o x_{i\mu} x_{j\nu}
    x_i is the i'th eigenvector (ordered by eigenvalues from smallest to largest) Only first N_o is required
    Returns D, and eigarray
    """
    D       = T.duplicate()
    xr,tmp  = T.getVecs()
    xi,tmp  = T.getVecs()
    xr_size = xr.getSize()
   # eigarray = np.zeros(nocc)
    seqx = PETSc.Vec()
    seqx.createSeq(xr_size,comm=PETSc.COMM_SELF)
    fromIS = PETSc.IS().createGeneral(range(xr_size),comm=PETSc.COMM_SELF)
    toIS = PETSc.IS().createGeneral(range(xr_size),comm=PETSc.COMM_SELF)
    for m in xrange(nocc):
        k = eps.getEigenpair(m, xr, xi)
     #   eigarray[m] = k.real
        sctr=PETSc.Scatter().create(xr,fromIS,seqx,toIS)
        sctr.begin(xr,seqx,addv=PETSc.InsertMode.INSERT,mode=PETSc.ScatterMode.FORWARD)
        sctr.end(xr,seqx,addv=PETSc.InsertMode.INSERT,mode=PETSc.ScatterMode.FORWARD)
        Istart, Iend = D.getOwnershipRange()
        for i in xrange(Istart,Iend):
            cols = T.getRow(i)[0] 
            values = [ 2.0 * seqx[i] * seqx[j] for j in cols]
            D.setValues(i,cols,values,addv=PETSc.InsertMode.ADD_VALUES)
        D.assemble()
        error = eps.computeError(m)
        if error > 1.e-6: Print("Error: %12g" % ( error)) 
    if k.imag != 0.0:
          Print("Complex eigenvalue dedected: %9f%+9f j  %12g" % (k.real, k.imag, error))
    return D #,eigarray

def getSIPsDensityMatrix(eps,nocc):
    """
    PetscErrorCode EPSCreateDensityMat(EPS eps,PetscReal *weight,PetscInt idx_start,PetscInt idx_end,Mat *P);

    """
    A= sips.getDensityMat(eps,1,nocc)

    return A

def getDensityMatrixLocal(eps,T,nocc):
    """
    nocc  = Number of occupied orbitals = N_e / 2 = Number of electrons / 2
    D = One-electron density matrix. The sparsity pattern is the same as T.
    eigarray = An array of length N_o, containing the eigenvalues 
    D_{\mu\nu} = 2 \sum_i^N_o x_{i\mu} x_{j\nu}
    x_i is the i'th eigenvector (ordered by eigenvalues from smallest to largest) Only first N_o is required
    Returns D, and eigarray
    Assumes that e
    """

    D       = T.duplicate()
    xr,tmp  = T.getVecs()
    xi,tmp  = T.getVecs()
    xr_size = xr.getSize()
    eigarray = np.zeros(nocc)
    seqx = PETSc.Vec()
    seqx.createSeq(xr_size,comm=PETSc.COMM_SELF)
    fromIS = PETSc.IS().createGeneral(range(xr_size),comm=PETSc.COMM_SELF)
    toIS = PETSc.IS().createGeneral(range(xr_size),comm=PETSc.COMM_SELF)
    for m in xrange(nocc):
        k = eps.getEigenpair(m, xr, xi)
        eigarray[m] = k.real
        sctr=PETSc.Scatter().create(xr,fromIS,seqx,toIS)
        sctr.begin(xr,seqx,addv=PETSc.InsertMode.INSERT,mode=PETSc.ScatterMode.FORWARD)
        sctr.end(xr,seqx,addv=PETSc.InsertMode.INSERT,mode=PETSc.ScatterMode.FORWARD)
        Istart, Iend = D.getOwnershipRange()
        for i in xrange(Istart,Iend):
            cols = D.getRow(i)[0] 
            ncols = len(cols)
            values = [ 2.0 * seqx[i] * seqx[j] for j in cols]
            D.setValues(i,cols,values,addv=PETSc.InsertMode.ADD_VALUES)
            D.assemble()
        error = eps.computeError(m)
        if error > 1.e-6: Print(" %12g" % ( error)) 
    if k.imag != 0.0:
          Print("Complex eigenvalue dedected: %9f%+9f j  %12g" % (k.real, k.imag, error))
    return D,eigarray

def setupEPS(A,B=None,interval=[0]):
    """
    Returns SLEPc eps object for given operators, and options.
    If matrix B is not given, solves the standard eigenvalue problem for a Hermitian matrix A.
    If matrix B (should be positive definite) is given, solve the generalized eigenvalue problem. 
    EPS comm is the same as the comm of A.
    """
    eps = SLEPc.EPS().create(comm=A.getComm())
    eps.setOperators(A,B)
    if B: problem_type=SLEPc.EPS.ProblemType.GHEP
    else: problem_type=SLEPc.EPS.ProblemType.HEP
    eps.setProblemType( problem_type )
    st  = eps.getST()
    st.setType(SLEPc.ST.Type.SINVERT)
    st.setMatStructure(PETSc.Mat.Structure.SAME_NONZERO_PATTERN)
    ksp=st.getKSP()
    ksp.setType(PETSc.KSP.Type.PREONLY)
    pc=ksp.getPC()
    pc.setType(PETSc.PC.Type.CHOLESKY)
    pc.setFactorSolverPackage('mumps')
    PETSc.Options().setValue('mat_mumps_icntl_13',1)
    PETSc.Options().setValue('mat_mumps_icntl_24',1)
    PETSc.Options().setValue('mat_mumps_cntl_3',1.e-12)
    if len(interval)==2:
        eps.setInterval(interval[0],interval[1])
        Print("Solving for eigenvalues in [{0:5.3f}, {1:5.3f}]".format(interval[0], interval[1]))
    eps.setWhichEigenpairs(SLEPc.EPS.Which.ALL)
    eps.setFromOptions()
    eps.setUp()
    return eps

def updateEPS(eps,A,B=None,subintervals=[0],local=True, globalupdate=False,):
    """
    Updates eps object for a new matrix with the same nnz structure
    as the previous one.
    No need to call
        eps.setFromOptions()
        eps.setUp()
    If these functions are called additional (unnecessary)
    1 sym, and 2 num factoizations are performed.
    """
    if local:
        eps.updateKrylovSchurSubcommMats(s=0.0, a=1.0, Au=A, t=1.0, b=1.0, Bu=B, 
                                         structure=A.Structure.SAME_NONZERO_PATTERN, 
                                         globalup=globalupdate)
    else:
        eps.setOperators(A,B)    
    if len(subintervals)>1:
        #Print("subintervals:{0}".format(subintervals))
        eps.setInterval(subintervals[0],subintervals[-1])
        Print("Solution interval: {0:5.3f}, {1:5.3f}".format(subintervals[0], subintervals[-1]))
        if len(subintervals)>2:
            eps.setKrylovSchurPartitions(len(subintervals)-1)
            eps.setKrylovSchurSubintervals(subintervals)
    return eps

def solveEPS(eps):
    """
    Solve the eigenvalue problem defined in eps.
    Return eps
    """
    eps.solve()
    return eps

def getEPSInterval(eps):
    """
    Returns left and right boundaries of the 
    eps interval.
    """
    return eps.getInterval()

def getNumberOfConvergedEigenvalues(eps):
    """
    Returns total number of converged eigenvalues.
    """
    return eps.getConverged()

def getNEigenvalues(eps,N):
    """
    Returns N eigenvalues in a numpy array
    """
    eigarray=np.zeros(N)
    for i in range(N):
        k = eps.getEigenvalue(i)
        eigarray[i]=k.real 
    return eigarray

def getConvergedEigenvalues(eps):
    """
    Returns a numpy array of all converged eigenvalues
    """
    nconv = getNumberOfConvergedEigenvalues(eps)
    return getNEigenvalues(eps, nconv)       

def solveEPSdepreceated(eps,printinfo=False,returnoption=0,checkerror=False,nreq=0):
    """
    returnoption 0:
        Returns SLEPc EPS object, 
        Number of converged eigenvalues 
    returnoption 1:
        Returns SLEPc EPS object, 
        Number of converged eigenvalues
        An array of converged eigenvalues 
    returnoption 2:
        Returns SLEPc EPS object, 
        Number of converged eigenvalues
        An array of converged eigenvalues     
    returnoption 3:
        Returns SLEPc EPS object, 
        Number of converged eigenvalues
        An array of converged eigenvalues     
        A dense matrix of eigenvectors  
        
    TODO: 
        Repeat solve with larger range if nconv<nreq
        Or continue with random (or approximate or save some from previous iter) 
        eigenvectors to replace the missing ones.
        Or and maybe the best option is simply form the Density Matrix with 
        the converged eigenvectors and increase the interval for the next iter.     
    """
    left , right = eps.getInterval()       
    Print("Solving for eigenvalues in [{0:5.3f}, {1:5.3f}]".format(left, right))
    eps.solve()
    nconv = eps.getConverged()
    if nreq :
        Print("Number of converged and required eigenvalues: {0}, {1} ".format(nconv, nreq))
    if printinfo:
        its = eps.getIterationNumber()
        sol_type = eps.getType()
        nev, ncv, mpd = eps.getDimensions()
        tol, maxit = eps.getTolerances()
        Print("EPS dimensions (nev,ncv,mpd): {0},{1},{2}".format(nev,ncv,mpd))
        Print("Number of eps iterations: %i" % its)
        Print("Solution method: %s" % sol_type)
        Print("Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit))
    if nconv < nreq:
        Print("Missing eigenvalues!")
        newleft  = left  - 1.0
        newright = right + 1.0
        eps.setInterval(newleft,newright)
        solveEPS(eps,printinfo=printinfo,returnoption=returnoption,checkerror=checkerror,nreq=nreq)
    if returnoption==0:
        return eps, nconv
    elif returnoption == 1:
        eigarray=np.zeros(nconv)
        for i in range(nconv):
            k = eps.getEigenvalue(i)
            eigarray[i]=k.real
            if checkerror:
                error = eps.computeError(i)
                if error > 1.e-6: Print("Eigenvalue {0} has error {1}".format(k,error)) 
        if nconv >= nreq :
            Print("Range of required eigenvalues: {0:5.3f} , {1:5.3f}".format(eigarray[0],eigarray[nreq-1]))
        else:
            Print("Range of converged eigenvalues: {0:5.3f} , {1:5.3f}".format(eigarray[0],eigarray[-1]))
            Print("Missing eigenvalues!")
        return eps, nconv, eigarray
    elif returnoption == 2:
        eigarray=np.zeros(nconv)
        for i in range(nconv):
            k = eps.getEigenpair(i,None,None)
            eigarray[i]=k.real
            if checkerror:
                error = eps.computeError(i)
                if error > 1.e-6: Print("Eigenvalue {0} has error {1}".format(k,error)) 
        return eps, nconv, eigarray
    elif returnoption == 3:
        A = eps.getOperators(eps)[0]
        eigarray=np.zeros(nconv)
        eigmat=np.zeros((nconv,A.getSize()[0]))
        xr = A.getVecs()[0]
        xi = A.getVecs()[0]
        for i in range(nconv):
            k = eps.getEigenpair(i,xr,xi)
            eigarray[i]=k.real
            eigmat[i,:]=xr
            if checkerror:
                error = eps.computeError(i)
                if error > 1.e-6: Print("Eigenvalue {0} has error {1}".format(k,error)) 
        return eps, nconv, eigarray,eigmat
    