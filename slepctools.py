import sys, slepc4py
slepc4py.init(sys.argv)
from petsc4py import PETSc
from slepc4py import SLEPc

Print = PETSc.Sys.Print

import numpy as np
try:
    import scipy.sparse
except:
    Print("scipy not found")
    pass
   

def getNumberOfSubIntervals(eps):
    return eps.getKrylovSchurPartitions()

def getSubIntervals(eigs, nsub, bufferratio=0.5):
    """
    Given a list of eigenvalues, (eigs) and number of subintervals (nsub), 
    returns the boundaries for subintervals such that each subinterval has an average number of eigenvalues.
    Doesn't skip gaps, SLEPc doesn't support it, yet.
    range of eigs * bufferratio gives a buffer zone for leftmost and rightmost boundaries.
    """
    eigs = sorted(np.array(eigs))
    neigs = len(eigs)
    mean = neigs / nsub
    remainder = neigs % nsub
    irange = eigs[-1] - eigs[0]
    ibuffer = irange * bufferratio
    subint = np.zeros(nsub + 1)
    subint[0] = eigs[0] - ibuffer
    for i in xrange(1, nsub):
        subint[i] = (eigs[mean * i] + eigs[mean * i - 1]) / 2.
        if remainder > 0 and i > 1:
            subint[i] = (eigs[mean * i + 1] + eigs[mean * i]) / 2.
            remainder = remainder - 1
    subint[nsub] = eigs[-1] + ibuffer
    Print("New interval boundaries: {0:5.3f} , {1:5.3f}".format(subint[0],subint[-1]))
    return subint


def getDensityMatrix(eps,T,nocc):
    """
    nocc  = Number of occupied orbitals = N_e / 2 = Number of electrons / 2
    D = One-electron density matrix. The sparsity pattern is the same as T.
    eigarray = An array of length N_o, containing the eigenvalues 
    D_{\mu\nu} = 2 \sum_i^N_o x_{i\mu} x_{j\nu}
    x_i is the i'th eigenvector (ordered by eigenvalues from smallest to largest) Only first N_o is required
    Returns D, and eigarray
    """
    import constants as const

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
        if error > 1.e-6: Print(" %12g" % ( error)) 
    if k.imag != 0.0:
          Print("Complex eigenvalue dedected: %9f%+9f j  %12g" % (k.real, k.imag, error))
  #  HOMO = k.real
  #  LUMO = eps.getEigenpair(nocc, xr, xi).real
  #  eigarray[nocc]=LUMO
  #  gap  = LUMO - HOMO
  #  Print("LUMO-HOMO      = {0:20.10f} kcal/mol = {1:20.10f} ev = {2:20.10f} Hartree".format(gap*const.ev2kcal,gap,gap*const.ev2hartree))
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
    import constants as const

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
  #  HOMO = k.real
  #  LUMO = eps.getEigenpair(nocc, xr, xi).real
  #  eigarray[nocc]=LUMO
  #  gap  = LUMO - HOMO
  #  Print("LUMO-HOMO      = {0:20.10f} kcal/mol = {1:20.10f} ev = {2:20.10f} Hartree".format(gap*const.ev2kcal,gap,gap*const.ev2hartree))
    return D,eigarray


def solveEPS(A,B=None,printinfo=False,returnoption=0,checkerror=False,interval=[0],subintervals=[0],nocc=0):
    """
    If matrix B is not given, solves the standard eigenvalue problem for a Hermitian matrix A.
    If matrix B (should be positive definite) is given, solve the generalized eigenvalue problem. 
    EPS comm is the same as the comm of A.
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
        
    TODO: Resolve with larger range if nconv<nocc     
    """
    eps = SLEPc.EPS().create(comm=A.getComm())
    eps.setOperators(A,B)
    if B: problem_type=SLEPc.EPS.ProblemType.GHEP
    else: problem_type=SLEPc.EPS.ProblemType.HEP
    eps.setProblemType( problem_type )
    eps.setFromOptions()
    if any(interval)==2:
        eps.setInterval(interval[0],interval[1])
    if any(subintervals):
        eps.setInterval(subintervals[0],subintervals[-1])
        eps.setKrylovSchurPartitions(len(subintervals)-1)
        eps.setKrylovSchurSubintervals(subintervals)    
    eps.solve()
    nconv = eps.getConverged()
    Print("Number of converged eigenvalues: ".format(nocc))

    if printinfo:
        its = eps.getIterationNumber()
        sol_type = eps.getType()
        nev, ncv, mpd = eps.getDimensions()
        tol, maxit = eps.getTolerances()
        
        Print("Number of iterations of the method: %i" % its)
        Print("Solution method: %s" % sol_type)
        Print("Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit))
    if nconv < nocc:
        Print("{0} eigenvalues found, {1} eigenvalues are required".format(nconv,nocc))
        sys.exit()
    elif returnoption==0:
        return eps, nconv
    elif returnoption == 1:
        eigarray=np.zeros(nconv)
        for i in range(nconv):
            k = eps.getEigenvalue(i)
            eigarray[i]=k.real
            if checkerror:
                error = eps.computeError(i)
                if error > 1.e-6: Print("Eigenvalue {0} has error {1}".format(k,error)) 
        Print("Range of required eigenvalues: {0:5.3f} , {1:5.3f}".format(eigarray[0],eigarray[nocc-1]))
        #Print("{0}, {1}, {2} ".format(eigarray[0],eigarray[nocc-1],eigarray[nocc]))

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
        eigarray=np.zeros(nconv)
        eigmat=np.zeros((nconv,A.getSize()[0]))
        xr, tmp = A.getVecs()
        xi, tmp = A.getVecs()
        for i in range(nconv):
            k = eps.getEigenpair(i,xr,xi)
            eigarray[i]=k.real
            eigmat[i,:]=xr
            if checkerror:
                error = eps.computeError(i)
                if error > 1.e-6: Print("Eigenvalue {0} has error {1}".format(k,error)) 
        return eps, nconv, eigarray,eigmat
            
def solve_HEP(A, B=None,problem_type=SLEPc.EPS.ProblemType.HEP):
    sizeA= A.getSize()
    D=A.duplicate()
    Print("Matrix size: %i,%i" % (sizeA[0],sizeA[1]))
    Print("Matrix local size: %i,%i" % (A.getLocalSize()[0],A.getLocalSize()[1]))
    # Create the results vectors
    xr, tmp = A.getVecs()
    xi, tmp = A.getVecs()
    xr_size = xr.getSize()
    xr_localsize = xr.getLocalSize()
    Print("Vector size: %i" % (xr_size))
    Print("Vector local size: %i" % (xr.getLocalSize()))
    # Setup the eigensolver
    """
    Here the comm should be the same as the comm that has A
    """
    E = SLEPc.EPS().create(comm=A.getComm())
    E.setOperators(A,B)
    E.setDimensions(3,PETSc.DECIDE)
    E.setProblemType( problem_type )
    E.setFromOptions()

    # Solve the eigensystem
    E.solve()

    #Print("")
    its = E.getIterationNumber()
    #Print("Number of iterations of the method: %i" % its)
    sol_type = E.getType()
    #Print("Solution method: %s" % sol_type)
    nev, ncv, mpd = E.getDimensions()
    Print("Number of requested eigenvalues: %i" % nev)
    tol, maxit = E.getTolerances()
    #Print("Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit))
    nconv = E.getConverged()
   # islice, nconvlocal,xr = E.getKrylovSchurSubcommInfo()
   # subcomm = xr.getComm()
    Print("Number of converged eigenpairs: %d" % nconv)
    seqx = PETSc.Vec()
    seqx.createSeq(xr_size,comm=PETSc.COMM_SELF)
    seqx.setFromOptions()
   # seqx.set(0.0)
    fromIS = PETSc.IS().createGeneral(range(xr_size),comm=PETSc.COMM_SELF)
    toIS = PETSc.IS().createGeneral(range(xr_size),comm=PETSc.COMM_SELF)
    if nconv > 0:
        evals=np.zeros(nconv)
        evecs=np.zeros((xr.getSize(),nconv))

        for i in range(nconv):
            k = E.getEigenpair(i, xr, xi)
          #  k = E.getKrylovSchurSubcommPairs(i, veclocal)
            sctr=PETSc.Scatter().create(xr,fromIS,seqx,toIS)
            sctr.begin(xr,seqx,addv=PETSc.InsertMode.INSERT,mode=PETSc.ScatterMode.FORWARD)
            sctr.end(xr,seqx,addv=PETSc.InsertMode.INSERT,mode=PETSc.ScatterMode.FORWARD)
            evals[i]=k.real
            evecs[:,i]=seqx
            error = E.computeError(i)
            if error > 1.e-6: Print(" %12g" % ( error)) 
        if k.imag != 0.0:
              Print("Complex eigenvalue dedected: %9f%+9f j  %12g" % (k.real, k.imag, error))
        return evals,evecs
    else:
        Print("No eigenvalues in the given interval")
        return None,None  
