from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np
Print = PETSc.Sys.Print
try: 
    from sklearn.cluster import KMeans
except:
    Print("sklearn.cluster not found.")
    Print("-bintype 3, can not be used")
        
def getDensityOfStates(eigs,width=0.1,npoint=200):
    """
    Computes density of states (dos) by representing
    `eigs` (eigenvalues) as gaussians of given `width`. 
    Parameters
    ----------
    eigs   : Array of floats
             Eigenvalues required to generate dos
    width  : Width of the gaussian function centered
             at eigenvalues
             eigs and width have the same units.
    npoint : Int
             Number of energy points to compute dos.
    Returns
    -------
    energies : Array of floats
               Energies for which DOS is computed.     
    dos      : Array of floats
               len(dos) = len(energies) = npoints 
               Density of states computed at energies.
               dos has the inverse of the units of energies.
    Notes
    -----
    The mathematical definition of density of states is:
    $D(x) = \frac{1}{N}\sum\limits_n \delta(x-x_n)$,
    where $x_n$ is the $n$th eigenvalue and $N$ is 
    the total number of eigenvalues.
    Here, delta function is represented by a gaussian,i.e.
    $\delta(x) = \frac{1}{a\sqrt{\pi}}\exp(-\frac{x^2}{a^2})$
    """
    b = width * 5.
    energies = np.linspace(min(eigs)-b,max(eigs)+b,num=npoint)
    N = len(eigs)
    w2 = width * width
    tmp = np.zeros(len(energies))
    for eig in eigs:
        tmp += np.exp(-(energies-eig)**2 / w2)
    dos = tmp / (np.sqrt(np.pi) * width * N)    
    return energies, dos

def plotDensityOfStates(energies,dos,units='ev',title='DOS',filename='dos.png'):
    """
    Plots density of states.
    Parameters
    ---------
    energies : Array of floats
               Energies for which DOS is computed.     
    dos      : Array of floats
               len(dos) = len(energies) 
               Density of states computed at energies.
               dos has the inverse of the units of energies
    """
    try:
        import matplotlib.pyplot as plt
    except: 
        Print("Requires matplotlib")
        return    
    assert len(energies) == len(dos), "energies and dos should have the same length"
    plt.figure()
    plt.plot(energies,dos)
    plt.xlabel('Energies ({0})'.format(units))
    plt.ylabel('DOS 1/({0})'.format(units))
    plt.title(title)
    plt.savefig(filename)
    return

def getNumberOfSubIntervals(eps):
    return eps.getKrylovSchurPartitions()

def getClusters(x, crange=1.e-6):
    """
    Given an array of numbers (x) and a threshold for clustering,
    returns an array of clusters, and the multiplicities
    of each cluster.
    Input:
    eigs    - numpy array (dtype='float64')
    chtresh - float (optional)
    Returns:
            - numpy array (dtype='float64')
            - numpy array (dtpye='int32')
    """
    nx          = len(x)
    clusters       = np.zeros(nx)
    multiplicities = np.ones(nx,dtype='int32')
    clusters[0]    = x[0]
    icluster       = 0
    for i in range(1,nx):
        if x[i]-clusters[icluster] < crange:
            multiplicities[icluster] += 1
        else:
            icluster += 1
            clusters[icluster] = x[i]
    ncluster = icluster + 1
    return clusters[0:ncluster], multiplicities[0:ncluster]

def getBinEdges1(x, nbin, rangebuffer=0.1,interval=[0],cthresh=1.e-6):
    """
    Given a list of eigenvalues, (x) and number of subintervals (nbin), 
    returns the boundaries for subintervals such that each subinterval has an average number of eigenvalues.
    Doesn't skip gaps, SLEPc doesn't support it, yet.
    range of x * rangebuffer gives a rangebuffer zone for leftmost and rightmost boundaries.
    """
    x, mults = getClusters(x,cthresh) 
    nx = len(x)
    mean = nx / nbin
    remainder = nx % nbin
    erange = x[-1] - x[0]
    isbuffer = erange * rangebuffer
    b = np.zeros(nbin + 1)
    b[0] = x[0] - isbuffer
    for i in range(1, nbin):
        b[i] = (x[mean * i] + x[mean * i - 1]) / 2.
        if remainder > 0 and i > 1:
            b[i] = (x[mean * i + 1] + x[mean * i]) / 2.
            remainder = remainder - 1
    b[nbin] = x[-1] + isbuffer
    if len(interval)==2:
        b[0]  = interval[0]
        b[-1] = interval[1]
    return b, mults

def getBinEdges2(x,nbin,binbuffer=0.001):
    """
    Given an array of numbers (x) and number
    of bins (nbins), returns optimum bin edges,
    to reduce number of shifts required for eps
    solve.
    Input:
    x       - numpy array (dtype='float64')
    nbin    - int
    Returns:
            - numpy array (dtype='float64', len = nbin+1)
    TODO:
    Bisection type algorithms can be used to obtain optimum bin edges.
    Binning score can be used for better optimization of bin edges
    """
    rangex = max(x) - min(x)
    meanbinsize = rangex / float(nbin)
    b = np.zeros(nbin+1)
    maxtrial = 100
    crange = meanbinsize
    i = 0
    while i < maxtrial:
        i += 1
        clusters = getClusters(x,crange=crange)[0]
        ncluster = len(clusters)
        if ncluster > nbin:
            crange = crange * 1.1
        elif ncluster < nbin:    
            crange = crange * 0.9
        else:
            break
    if i == maxtrial :
        Print("Bin optimization failed for bintype 2, switching to bintype 1. Found {0} clusters".format(ncluster))
        Print("Adjust bin edges based on prior knowledge to have a uniform number of eigenvalues in each bin")
        b = getBinEdges1(x,nbin)[0]
    else:
        for i in range(1,nbin):
            b[i] = clusters[i] - binbuffer
    return b

def getBinEdges3(x,nbin,binbuffer=0.001):
    """
    Given an array of numbers (x) and number
    of bins (nbins), returns bin edges,
    based on k-means clustering algorithm.
    Parameters
    ----------
    x       - numpy array (dtype='float64')
    nbin    - int
    Returns:
            - numpy array (dtype='float64', len = nbin+1)
    Notes
    -----
    Requires scikit package.
    Using k-means for 1d arrays is considered to be an overkill.
    However, it seems to me as a practical solution for the binning problem.
    """
    n = len(x)
    b = np.zeros(nbin+1)
    # random_state is used to fix the seed, to avoid random results
    # clusterids is an integer array (with len(x)) returning an index for clusters found.
    clusterids = KMeans(n_clusters=nbin,random_state=17).fit_predict(x.reshape(-1,1))
    assert n == len(clusterids), "Kmeans failed, missing clusters"
    b[0]       = x[0] - binbuffer
    uniqueids  = np.array([clusterids[i+1]-clusterids[i]!=0 for i in range(n-1)],dtype=bool)
    b[1:-1]    = x[1:][uniqueids] - binbuffer
    b[-1]      = x[-1] + binbuffer
    return b                     
            
def getBinEdges(x, nbin,bintype=2,binbuffer=0.001,rangebuffer=0.1,rangetype=0,A=None,interval=[0]):
    """
    Given an array of numbers (x) and number
    of bins (nbins), returns bin edges,
    based on binning algortithm.
    bintype = 0 :
        Fixed uniform width bins
    bintype = 1 :
        Bin edges are adjusted to contain uniform number of values.
    bintype = 2 :
        Bin edges are adjusted to minimize distance from left edge.
    bintype = 3 :
        Bin edges are adjusted based on k-means clustering.    
    Input:
    x       - numpy array (dtype='float64')
    nbin    - int
    Returns:
            - numpy array (dtype='float64', len = nbin+1)
    TODO:
    Bisection type algorithms can be used to obtain optimum bin edges.
    Binning score can be used for better optimization of bin edges
    """
    if len(interval)==2:
        a, b = interval[0], interval[1]
    else: 
        Print("A tuple or list of two values is required to define the interval")
        return interval 
    
    nx = len(x)
    if nx > 1:
        dx   = x[1:] - x[:-1]
        Print("Min seperation of eigenvalues: {0:5.3e}".format(min(dx)))
        Print("Max seperation of eigenvalues: {0:5.3f}".format(max(dx)))
         
    if (nx < nbin or bintype == 0):
        Print("Uniform bins")
        bins = np.array([a,b])
    elif bintype == 1:
        Print("Adjust bin edges to have a uniform number of eigenvalues in each bin")
        bins = getBinEdges1(x,nbin)[0]   
    elif bintype == 2:
        Print("Adjust bin edges to minimize distance of values from the left edge of each bin")
        bins = getBinEdges2(x,nbin,binbuffer=binbuffer) 
    elif bintype == 3:
        Print("Adjust bin edges based on k-means clustering of eigenvalues")
        bins = getBinEdges3(x,nbin,binbuffer=binbuffer)  
                            
    if rangetype == 0:
        Print("Interval is set to: {0:5.3f}, {1:5.3f}".format(a,b))
    elif rangetype == 1 and nx > 1:
        Print("Interval is set to min & max of prior evals: {0:5.3f},{1:5.3f}".format(a,b))
        a = x[0]  - rangebuffer
        b = x[-1] + rangebuffer
    elif rangetype == 2: 
        diag   = A.getDiagonal()
        a = diag.min()[1]  - rangebuffer
        b = diag.max()[1]  + rangebuffer
        Print("Interval based on min & max of F diagonal: {0:5.3f},{1:5.3f}".format(a,b))
    elif rangetype == 3:
        diag   = A.getDiagonal()
        a = getLowerBound(A)                                                                                                        
        b = diag.max()[1]      + rangebuffer 
        Print("Interval based on min eval & max F diagonal: {0:5.3f},{1:5.3f}".format(a,b)) 
    elif rangetype == 4:
        a = getLowerBound(A)  
        b = getUpperBound(A)
        Print("Interval based on min & max evals: {0:5.3f},{1:5.3f}".format(a,b))    
    bins[0]  = a
    bins[-1] = b           
    return np.sort(bins)

def getBinningScore(b,x):
    """
    Returns a score (lower is better) for given
    bin_edges (b), and values (x)
    1) Find the eigenvalues within each slice
    Within a slice:
        2) Compute the sum of distances of eigenvalues from the closest 
           neigbor on the left.
        3) Add this sum to the distance of the leftmost eigenvalue from the 
           left boundary.
    Returns the max sum for each slice.
    """
    nbin   = len(b)-1
    scores = np.zeros(nbin)
    nempty = 0 
    for i in range(nbin):
        xloc=x[(x>b[i]) & (x<b[i+1])] #1
        if len(xloc) > 1:
            tmp = np.sum(xloc[1:]-xloc[:-1]) #2
            scores[i]= xloc[0]-b[i] + tmp #3
        else:
            nempty += 1
    return max(scores), nempty

def getSubcomm(eps):
    """
    Returns the subcommunicator for eps.
    """
    vec = eps.getKrylovSchurSubcommInfo()[2]
    return vec.getComm()       

def getDensityMatrix(eps,T,nocc,checkerror=False):
    """
    nocc  = Number of occupied orbitals = N_e / 2 = Number of electrons / 2
    D = One-electron density matrix. The sparsity pattern is the same as T.
    eigarray = An array of length N_o, containing the eigenvalues 
    D_{\mu\nu} = 2 \sum_i^N_o x_{i\mu} x_{j\nu}
    x_i is the i'th eigenvector (ordered by eigenvalues from smallest to largest) Only first N_o is required
    Returns D, and eigarray
    """
    subcomm = getSubcomm(eps)
    A       = eps.getOperators()[0]
    D       = T.duplicate()
    xr      = A.getVecs()[0]
    xr_size = xr.getSize()
    seqx = PETSc.Vec()
    seqx.createSeq(xr_size,comm=PETSc.COMM_SELF)
    fromIS = PETSc.IS().createGeneral(list(range(xr_size)),comm=PETSc.COMM_SELF)
    toIS = PETSc.IS().createGeneral(list(range(xr_size)),comm=PETSc.COMM_SELF)
    for m in range(nocc):
        eps.getEigenvector(m, xr)
        sctr=PETSc.Scatter().create(xr,fromIS,seqx,toIS)
        sctr.begin(xr,seqx,addv=PETSc.InsertMode.INSERT,mode=PETSc.ScatterMode.FORWARD)
        sctr.end(xr,seqx,addv=PETSc.InsertMode.INSERT,mode=PETSc.ScatterMode.FORWARD)
        Istart, Iend = D.getOwnershipRange()
        for i in range(Istart,Iend):
            cols = T.getRow(i)[0] 
            values = 2.0 * seqx[i] * seqx[cols]
            D.setValues(i,cols,values,addv=PETSc.InsertMode.ADD_VALUES)
        if checkerror:
            error = eps.computeError(m)
            if error > 1.e-6: Print("Error: %12g" % ( error)) 
    D.assemble()
    return D, subcomm

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
    fromIS = PETSc.IS().createGeneral(list(range(xr_size)),comm=PETSc.COMM_SELF)
    toIS = PETSc.IS().createGeneral(list(range(xr_size)),comm=PETSc.COMM_SELF)
    for m in range(nocc):
        k = eps.getEigenpair(m, xr, xi)
        eigarray[m] = k.real
        sctr=PETSc.Scatter().create(xr,fromIS,seqx,toIS)
        sctr.begin(xr,seqx,addv=PETSc.InsertMode.INSERT,mode=PETSc.ScatterMode.FORWARD)
        sctr.end(xr,seqx,addv=PETSc.InsertMode.INSERT,mode=PETSc.ScatterMode.FORWARD)
        Istart, Iend = D.getOwnershipRange()
        for i in range(Istart,Iend):
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

def getLowerBound(A,B=None,comm=None,epstol=1.e-2):
    """
    Computes the smallest eigenvalue with a 
    very loose threshold to approximate the lower bound
    for eps interval.
    Suggested by Jose.
    """
    if comm is None:
        comm = A.getComm()
    eps = SLEPc.EPS().create(comm)
    eps.setOperators(A,B)
    eps.setTolerances(epstol,100)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
    eps.solve()
    if eps.getConverged()==0:
        val = -100000
    else:
        val = eps.getEigenvalue(0).real * (1+epstol)
    eps.destroy()
    return val

def getUpperBound(A,B=None,comm=None,epstol=1.e-2):
    if comm is None:
        comm = A.getComm()
    eps = SLEPc.EPS().create(comm)
    eps.setOperators(A,B)
    eps.setTolerances(epstol,100)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
    eps.solve()
    if eps.getConverged()==0:
        val = 100000
    else:
        val = eps.getEigenvalue(0).real
    eps.destroy()
    return val

def computeRange(F,rangetype,rangebuffer):
    if rangetype == 2: 
        Fdiag   = F.getDiagonal()
        a = Fdiag.min()[1]
        b = Fdiag.max()[1]
        Print("Interval based on min & max of F diagonal: {0:5.3f},{1:5.3f}".format(a,b))
    elif rangetype == 3:
        a = getLowerBound(F)
        b = getUpperBound(F) 
        Print("Interval based on min & max evals: {0:5.3f},{1:5.3f}".format(a,b))
    elif rangetype == 4:
        a = getLowerBound(F)
        Fdiag   = F.getDiagonal()
        b = Fdiag.max()[1]
        Print("Interval based on min eval & max F diagonal: {0:5.3f},{1:5.3f}".format(a,b))     
    return (a-rangebuffer,b+rangebuffer)
    
def setupEPS(A,B=None,binedges=[0]):
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
    eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    PETSc.Options().setValue('mat_mumps_icntl_13',1)
    PETSc.Options().setValue('mat_mumps_icntl_24',1)
    PETSc.Options().setValue('mat_mumps_cntl_3',1.e-12)
    if len(binedges) > 1:
        eps.setInterval(binedges[0],binedges[-1])
        Print("EPS interval set to: {0:5.3f}, {1:5.3f}".format(binedges[0], binedges[-1]))
        if len(binedges)>2:
            eps.setKrylovSchurPartitions(len(binedges)-1)
            eps.setKrylovSchurSubintervals(binedges)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.ALL)
    eps.setFromOptions()
    eps.setUp()
    return eps

def updateEPS(eps,A,B=None,binedges=[0],local=True, globalupdate=False):
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
    if len(binedges)>1:
        #Print("binedges:{0}".format(binedges))
        eps.setInterval(binedges[0],binedges[-1])
        Print("Solution interval: {0:5.3f}, {1:5.3f}".format(binedges[0], binedges[-1]))
        if len(binedges)>2:
            eps.setKrylovSchurPartitions(len(binedges)-1)
            eps.setKrylovSchurSubintervals(binedges)
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
