import sys, slepc4py
slepc4py.init(sys.argv)
from petsc4py import PETSc
from slepc4py import SLEPc

Print = PETSc.Sys.Print

import numpy as np
try:
    import scipy.sparse
except:
    Print("no scipy modules")
    pass

def getNumberOfSubIntervals(E):
    return E.getKrylovSchurPartitions()

def getSubIntervals(eigs, nint, bufferratio=0.5):
    """
    Given a list of eigenvalues, (eigs) and number of intervals, (nint) returns the boundaries
    for subintervals such that each interval has an average number of eigenvalues.
    Doesn't skip gaps, SLEPc doesn't support it, yet.
    range of eigs * bufferratio gives a buffer zone for leftmost and rightmost boundaries.
    """
    eigs = sorted(np.array(eigs))
    neigs = len(eigs)
    mean = neigs / nint
    remainder = neigs % nint
    irange = eigs[-1] - eigs[0]
    ibuffer = irange * bufferratio
    subint = np.zeros(nint + 1)
    subint[0] = eigs[0] - ibuffer
    for i in xrange(1, nint):
        subint[i] = (eigs[mean * i] + eigs[mean * i - 1]) / 2.
        if remainder > 0 and i > 1:
            subint[i] = (eigs[mean * i + 1] + eigs[mean * i]) / 2.
            remainder = remainder - 1
    subint[nint] = eigs[-1] + ibuffer
    Print("subint")
    Print(subint[0],subint[-1])
    return subint

def getEigenSolutions(A, B=None):
        
    sizeA = A.getSize()
    D     = A.duplicate()
    cols  = D.getRowIJ()[1]
    Print("Matrix size: %i,%i" % (sizeA[0],sizeA[1]))
    xr, tmp = A.getVecs()
    xi, tmp = A.getVecs()

    # Setup the eigensolver
    E = SLEPc.EPS().create()
    E.setOperators(A,B)
    if B: problem_type=SLEPc.EPS.ProblemType.GHEP
    else: problem_type=SLEPc.EPS.ProblemType.HEP
    E.setProblemType( problem_type )
    E.setFromOptions()

    # Solve the eigensystem
    E.solve()

    Print("")
    its = E.getIterationNumber()
    Print("Number of iterations of the method: %i" % its)
    sol_type = E.getType()
    Print("Solution method: %s" % sol_type)
    nev, ncv, mpd = E.getDimensions()
    Print("Number of requested eigenvalues: %i" % nev)
    tol, maxit = E.getTolerances()
    Print("Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit))
    nconv = E.getConverged()
    Print("Number of converged eigenpairs: %d" % nconv)
    if nconv > 0:
        Print("")
        Print("        k          ||Ax-kx||/||kx|| ")
        Print("----------------- ------------------")
        for i in range(nconv):
            k = E.getEigenpair(i, xr, xi)
            error = E.computeError(i)
            if k.imag != 0.0:
              Print(" %9f%+9f j  %12g" % (k.real, k.imag, error))
            else:
              Print(" %12f       %12g" % (k.real, error))
        Print("")
    return 

def getDensityMatrix(E,T,neig):
    import constants as const

    D       = T.duplicate()
    xr,tmp  = T.getVecs()
    xi,tmp  = T.getVecs()
    xr_size = xr.getSize()
    eigarray = np.zeros(neig)
    seqx = PETSc.Vec()
    seqx.createSeq(xr_size,comm=PETSc.COMM_SELF)
    fromIS = PETSc.IS().createGeneral(range(xr_size),comm=PETSc.COMM_SELF)
    toIS = PETSc.IS().createGeneral(range(xr_size),comm=PETSc.COMM_SELF)
    for m in xrange(neig):
        k = E.getEigenpair(m, xr, xi)
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
        error = E.computeError(m)
        if error > 1.e-6: Print(" %12g" % ( error)) 
    if k.imag != 0.0:
          Print("Complex eigenvalue dedected: %9f%+9f j  %12g" % (k.real, k.imag, error))
  #  HOMO = k.real
  #  LUMO = E.getEigenpair(neig, xr, xi).real
  #  eigarray[neig]=LUMO
  #  gap  = LUMO - HOMO
  #  Print("LUMO-HOMO      = {0:20.10f} kcal/mol = {1:20.10f} ev = {2:20.10f} Hartree".format(gap*const.ev2kcal,gap,gap*const.ev2hartree))
    return D,eigarray

def getDensityMatrix1(A,nel, B=None):
    import constants as const

    sizeA = A.getSize()
    D     = A.duplicate()
    cols  = D.getRowIJ()[1]
 #   Print("Matrix size: %i,%i" % (sizeA[0],sizeA[1]))
 #   Print("Matrix local size: %i,%i" % (A.getLocalSize()[0],A.getLocalSize()[1]))
    # Create the results vectors
    xr, tmp = A.getVecs()
    xi, tmp = A.getVecs()
    xr_size = xr.getSize()
    xr_localsize = xr.getLocalSize()
  #  Print("Vector size: %i" % (xr_size))
  #  Print("Vector local size: %i" % (xr.getLocalSize()))
    # Setup the eigensolver
    """
    Here the comm should be the same as the comm that has A
    """
    E = SLEPc.EPS().create(comm=A.getComm())
    E.setOperators(A,B)
    E.setDimensions(3,PETSc.DECIDE)
    if B: problem_type=SLEPc.EPS.ProblemType.GHEP
    else: problem_type=SLEPc.EPS.ProblemType.HEP
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
    tol, maxit = E.getTolerances()
    #Print("Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit))
    nconv = E.getConverged()
   # islice, nconvlocal,xr = E.getKrylovSchurSubcommInfo()
   # subcomm = xr.getComm()
    if nconv<nev:
        Print("Number of requested-converged eigenpairs: {0}-{1}".format(nev,nconv))
    seqx = PETSc.Vec()
    seqx.createSeq(xr_size,comm=PETSc.COMM_SELF)
    seqx.setFromOptions()

   # seqx.set(0.0)
    fromIS = PETSc.IS().createGeneral(range(xr_size),comm=PETSc.COMM_SELF)
    toIS = PETSc.IS().createGeneral(range(xr_size),comm=PETSc.COMM_SELF)
    if nconv > nel/2 :
        for m in xrange(nel/2):
            k = E.getEigenpair(m, xr, xi)
          #  k = E.getKrylovSchurSubcommPairs(m, veclocal)
            sctr=PETSc.Scatter().create(xr,fromIS,seqx,toIS)
            sctr.begin(xr,seqx,addv=PETSc.InsertMode.INSERT,mode=PETSc.ScatterMode.FORWARD)
            sctr.end(xr,seqx,addv=PETSc.InsertMode.INSERT,mode=PETSc.ScatterMode.FORWARD)
            Istart, Iend = D.getOwnershipRange()
   #         for ii in range(xr.getSize()):print xr[ii]
            for i in xrange(Istart,Iend):
                cols = D.getRow(i)[0] #maybe restore later
             #   print m,i,'cols',cols
                ncols = len(cols)
                values = [ 2.0 * seqx[i] * seqx[j] for j in cols]
    #            print 'eig', m,k,'row and cols',i,cols,':',values
                D.setValues(i,cols,values,addv=PETSc.InsertMode.ADD_VALUES)
                D.assemble()
            error = E.computeError(m)
            if error > 1.e-6: Print(" %12g" % ( error)) 
        if k.imag != 0.0:
              Print("Complex eigenvalue dedected: %9f%+9f j  %12g" % (k.real, k.imag, error))
        HOMO = k.real
        LUMO = E.getEigenpair(nel/2, xr, xi).real
        gap  = LUMO - HOMO
        Print("LUMO-HOMO      = {0:20.10f} kcal/mol = {1:20.10f} ev = {2:20.10f} Hartree".format(gap*const.ev2kcal,gap,gap*const.ev2hartree))
        return D
    else:
        Print("Not enough eigenvalues in the given interval")
        import sys
        sys.exit()  

def solveEPS(A,B=None,printinfo=False,returnoption=0,checkerror=False,interval=[0],subintervals=[0]):
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
        A dense matrix of eigenvectors   
    """
    E = SLEPc.EPS().create(comm=A.getComm())
    E.setOperators(A,B)
    if B: problem_type=SLEPc.EPS.ProblemType.GHEP
    else: problem_type=SLEPc.EPS.ProblemType.HEP
    E.setProblemType( problem_type )
    E.setFromOptions()
    if any(interval)==2:
        E.setInterval(interval[0],interval[1])
    if any(subintervals):
        E.setInterval(subintervals[0],subintervals[-1])
        E.setKrylovSchurPartitions(len(subintervals)-1)
        E.setKrylovSchurSubintervals(subintervals)    
    E.solve()
    neig = E.getConverged()

    if printinfo:
        its = E.getIterationNumber()
        sol_type = E.getType()
        nev, ncv, mpd = E.getDimensions()
        tol, maxit = E.getTolerances()
        
        Print("Number of iterations of the method: %i" % its)
        Print("Solution method: %s" % sol_type)
        Print("Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit))
    if neig==0:
        Print("No eigenvalues found")
        return E, neig
    elif returnoption==0:
        return E, neig
    elif returnoption == 1:
        eigarray=np.zeros(neig)
        for i in range(neig):
            k = E.getEigenvalue(i)
            eigarray[i]=k.real
            if checkerror:
                error = E.computeError(i)
                if error > 1.e-6: Print("Eigenvalue {0} has error {1}".format(k,error)) 
        return E, neig, eigarray
    elif returnoption == 2:
        eigarray=np.zeros(neig)
        eigmat=np.zeros((neig,A.getSize()[0]))
        xr, tmp = A.getVecs()
        xi, tmp = A.getVecs()
        for i in range(nconv):
            k = E.getEigenpair(i,xr,xi)
            eigarray[i]=k.real
            eigmat[i,:]=xr
            if checkerror:
                error = E.computeError(i)
                if error > 1.e-6: Print("Eigenvalue {0} has error {1}".format(k,error)) 
        return E, neig, eigarray,eigmat
            
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
