import sys, petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
from mpi4py import MPI

write  = PETSc.Sys.Print
INSERT = PETSc.InsertMode.INSERT

import numpy as np

def getOptions():
    return PETSc.Options()

def getComm():
    return MPI.COMM_WORLD

def getPetscComm():
    return PETSc.COMM_WORLD

def getOptions():
    return PETSc.Options()

def getCommSum(comm,x,integer=False):
    """
    Sum integers or floats over the communicator.
    Should be faster than comm.reduce()
    """
    if integer:        
        sum=np.zeros(1,dtype='int32')
        send=np.array(x,dtype='int32')
        comm.Reduce([send, MPI.INT ], [sum, MPI.INT],
                op=MPI.SUM, root=0)
    else:    
        sum=np.zeros(1)
        send=np.array(x)
        comm.Reduce([send, MPI.DOUBLE ], [sum, MPI.DOUBLE],
                op=MPI.SUM, root=0)
    return sum[0]

def writeGitHash():
    """
    """
    if not PETSc.COMM_WORLD.rank:
        import subprocess
#        githash = subprocess.check_output(["git", "describe", "--always"]) # short hash or tag
        githash = subprocess.check_output(["git", "rev-parse", "HEAD"])  #long hasg
        write("Git hash: {0}".format(githash.strip()))
    return

def getHostName():
    """
    """
    if not PETSc.COMM_WORLD.rank:
        import socket
        host = socket.gethostname()
        write("PSCF running on host {0}".format(host))
        return host
    return 'host'

def getWallTime(t0=0, str=''):
    """
    Returns the walltime - t0 in seconds.
    """
    t = MPI.Wtime() - t0
    str = str + ' time'
    if(t0): write("{0: <14s}: {1:5.3f} seconds".format(str,t))
    return MPI.Wtime()

def getStage(stagename='stage', oldstage=None, printstage=True):
    """
    Easy PETSc staging tool, helps for profiling.
    """
    if oldstage: oldstage.pop()
    if printstage: write("{0:*^30s}".format("Stage "+stagename))
    stage = PETSc.Log.Stage(stagename); 
    stage.push();
    return stage

def getStageTime(newstage='',oldstage='',t0=0):
    """
    PETSc staging tool, helps for profiling.
    """
    t = MPI.Wtime() - t0
    if oldstage:
        str1 = oldstage.name + ' time'
        write("{0: <14s}: {1:5.3f} seconds".format(str1,t))
        oldstage.pop()
    elif newstage=='':
        str1 = 'Overall'
        write("{0: <14s}: {1:5.3f} seconds".format(str1,t))    
    if newstage:    
        write("{0:*^30s}".format("Stage "+ newstage))
        newstage = PETSc.Log.Stage(newstage); 
        newstage.push();
        return newstage, MPI.Wtime()
    else:
        return
    
def distributeN(comm,N):
    """
    Distribute N consecutive things (rows of a matrix , blocks of a 1D array) 
    as evenly as possible over a given communicator.
    Uneven workload (differs by 1 at most) is on the initial ranks.
    
    Parameters
    ----------
    comm: MPI communicator
    N:  int
        Total number of the things to be ditributed.
    
    Returns
    ----------
    rstart: index of first local row
    rend: 1 + index of last row
    
    Notes
    ----------
    Index is zero based.
    """
    P      = comm.size
    rank   = comm.rank
    rstart = 0
    rend   = 0
    if P >= N:
        if rank < N:
            rstart = rank
            rend   = rank + 1
    else:
        n = N/P
        remainder = N%P
        rstart    = n * rank
        rend      = n * (rank+1)
        if remainder:
            if rank >= remainder:
                rstart += remainder
                rend   += remainder
            else: 
                rstart += rank
                rend   += rank + 1    
    return rstart, rend

def divideWithoutNan(a,b):
    """
    From http://stackoverflow.com/questions/26248654/numpy-return-0-with-divide-by-zero
    Parameters
    ---------
    a, b: numpy arrays
    
    Returns
    -------
    a /b
    
    Notes
    -------
    0/0 handling by adding invalid='ignore' to numpy.errstate()
    introducing numpy.nan_to_num() to convert np.nan to 0.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a,b)
        c[c == np.inf] = 0
        c = np.nan_to_num(c)
    return c    

def getCommutator(A,B):
    """
    Parameters
    ---------
    A, B: mat
    
    Returns
    -------
    [A,B] = A * B - B * A
    """
    return A * B - B * A

def getMaxAbsAIJ(A):
    """
    Parameters
    ---------
    A: mat
    
    Returns
    -------
    Maximum absolute value of A.
    
    Notes
    -----
    Collective
    """
    comm = A.getComm().tompi4py()
    rstart, rend = A.getOwnershipRange()
    maxval=0.0
    for i in xrange(rstart,rend):
        cols,vals = A.getRow(i)
        tmpmax = max(np.absolute(vals))
        if tmpmax > maxval:
            maxval = tmpmax
    return comm.allreduce(maxval,op=MPI.MAX)

def getLocalNNZ(A):
    """
    Parameters
    ---------
    A: mat
    
    Returns
    -------
    Number of nonzeros in the local portion of A.
    """
    rstart, rend = A.getOwnershipRange()
    nnz = 0
    for i in xrange(rstart,rend):
        cols, vals = A.getRow(i)
        nnz += len(cols)
    return nnz    

def getLocalNonZeroArray(A):
    """
    Parameters
    ---------
    A: mat
    
    Returns
    -------
    A numpy array of nonzeros of the local portion of A.
    """
    nnzstart = 0
    nnzend = 0
    nnz = getLocalNNZ(A)
    tmp = np.zeros(nnz)
    rstart, rend = A.getOwnershipRange()
    for i in xrange(rstart,rend):
        cols,vals = A.getRow(i)
        nnzend += len(cols)
        tmp[nnzstart:nnzend] = vals
        nnzstart = nnzend    
    return tmp
      
def printAIJ(A,text=''):
    """
    Parameters
    ---------
    A: mat
    text: string
    
    Returns
    -------
    """
    rank         = A.getComm().rank
    rstart, rend = A.getOwnershipRange()
    for i in xrange(rstart,rend):
        cols,vals = A.getRow(i)
        k=0
        for j in cols:
            print rank,':',text,i,j,vals[k]
            k += 1
    return 0

def writeMat(A,filename='mat.bin'):
    """
    Writes a matrix into a file:
    If the extension is 
        bin: Petsc binary file
        txt: Petsc ASCII file
        mtx: Matrix market formatted file --> TODO
    Parameters
    ----------
    A: PETSc mat object
    filename:
    Returns
    -------
    0 if successful
    """
    ext = filename[-3:]
    if ext == 'bin':
        writer=PETSc.Viewer().createBinary(filename, 'w')
    elif ext == 'txt':
        writer=PETSc.Viewer().createASCII(filename, 'w')
    elif ext == 'mtx':
        pass        
    else:
        write("Use bin, or txt extension")
    writer(A)
    return 0
     
def printVec(V,text=''):
    rank         = MPI.COMM_WORLD.rank
    rstart, rend = V.getOwnershipRange()
    for i in xrange(rstart,rend):
        print rank,':',text,V[i]
    return 0

def createVec(comm):
    return PETSc.Vec().create(comm=comm)

def createMat(comm):
    return PETSc.Mat().create(comm=comm)

def createSquareMat(comm, localsize,globalsize, opt=0):
    A = PETSc.Mat().create(comm=comm)
    A.setType('aij')
    A.setSizes([(localsize,globalsize),(localsize,globalsize)])
    A.setUp()
    A.setOption(A.Option.NEW_NONZERO_ALLOCATION_ERR,True)
    if opt == 1:
        rstart, rend = A.getOwnershipRange()
        for i in range(rstart,rend):
            A[i,i] = i + 1.0
        A.assemble()    
    return A

def perturbMat(A,pert=0.1):
    rstart, rend = A.getOwnershipRange()
    for i in range(rstart,rend):
        A.setValue(i,i,pert,addv=PETSc.InsertMode.ADD_VALUES)    
    A.assemble()
    return A    

        
def getCSRBandwidth(A):
    """
    From QuTIP
    Returns the max(mb), lower(lb), and upper(ub) bandwidths of a 
    qobj or sparse CSR/CSC matrix.
  
    If the matrix is symmetric then the upper and lower bandwidths are 
    identical. Diagonal matrices have a bandwidth equal to one.
  
    Parameters
    ----------
    A : qobj, csr_matrix, csc_matrix
        Input matrix
  
    Returns
    -------
    mb : int
        Maximum bandwidth of matrix.
    lb : int
        Lower bandwidth of matrix.
    ub : int
        Upper bandwidth of matrix.
  
    """
    from qutip.cy.sparse_utils import _sparse_bandwidth
    return _sparse_bandwidth(A.indices, A.indptr, A.shape[0])


def getLocalNnzPerRow(basis,rstart,rend,maxdist2):
    """
    Returns an array containing local number of nonzeros per row based on distance between atoms.
    Size depends on the number of rows per process.
    Locality is based on a temporarily created AIJ matrix. Is there a better way?
    I am not  sure if this is needed, I could do this for all processes since I only need to create a vector of size nbf
    """
    nbf=len(basis)
    dnnz=np.ones(rend-rstart)
    onnz=np.zeros(rend-rstart)
    
    for i in xrange(rstart,rend):
        atomi=basis[i].atom
        for j in xrange(i,rend):
            distij2=atomi.dist2(basis[j].atom)
            if distij2 < maxdist2: 
                dnnz[i] += 2
        for j in xrange(rend,nbf):
            distij2=atomi.dist2(basis[j].atom)
            if distij2 < maxdist2: 
                onnz[i] += 2
    return dnnz,onnz

def getLocalNnzInfoSym(basis,rstart,rend,maxdist2):
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

def getLocalNnzInfo(basis,rstart,rend,maxdist2):
    """
    Returns three arrays that contains: 
    dnnz: local numbers of nonzseros per row in diagonal blocks (square) 
    onnz: local numbers of nonzeros per row in off-diagonal blocks (rectangular)
    jmax: max column index that contains a nonzero.
    Nonzeros are based on distance between atoms.
    TODO: Exploit symmetry, not sure how to do that.
    """
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
    return dnnz, onnz, jmax

def getNnzVec(basis,maxdist):
    """
    Returns number of nonzeros per row based on distance between atoms.
    This is based on basis set rather than atoms, so there is redundancy in the calculations.
    TODO:
    Parallel version with a PETSc vector.
    Distance calculation loop should be over atoms indeed, which will improve performance by nbf/natom.
    If atoms are ordered based on a distance from a pivot point, loop can be reduced to exclude far atoms.
    """
    nbf=len(basis)
    maxdist2 = maxdist * maxdist
    Vnnz        = PETSc.Vec().createMPI((None,nbf),comm=PETSc.COMM_WORLD)
    rstart, rend = Vnnz.getOwnershipRange()
    for i in xrange(rstart,rend):
        atomi=basis[i].atom
        nnz=1
        for j in xrange(i):
            distij2=atomi.dist2(basis[j].atom)
            if distij2 < maxdist2: 
                nnz += 1
        Vnnz[i] = nnz
    Vnnz.assemble()            
    return Vnnz 

def getMaxNnzPerRow(mol,nat,maxdist):
    """
    Returns number of nonzeros per row based on distance between atoms.
    This is based on basis set rather than atoms, so there is redundancy in the calculations.
    TODO:
    Parallel version with a PETSc vector.
    Distance calculation loop should be over atoms indeed, which will improve performance by nbf/natom.
    If atoms are ordered based on a distance from a pivot point, loop can be reduced to exclude far atoms.
    """
    maxnnz=0
    maxband=0
    maxdist2 = maxdist * maxdist
    Vnnz        = PETSc.Vec().createMPI((None,nat),comm=MPI.COMM_WORLD)
    rstart, rend = Vnnz.getOwnershipRange()
    for i in xrange(rstart,rend):
        atomi=mol[i]
        nnz=0
        for j in xrange(i):
            distij2=atomi.dist2(mol[j])
            if distij2 < maxdist2: 
                band  = i - j 
                nnz += 1
                if band > maxband:
                    maxband = band
        if nnz > maxnnz:
            maxnnz = nnz        
    maxnnz  = MPI.COMM_WORLD.allreduce(maxnnz,op=MPI.MAX)        
    maxband = MPI.COMM_WORLD.allreduce(maxband,op=MPI.MAX)        
    return maxnnz*2+1, 2 * maxband   

def getNnzPerRowold(basis,maxdist):
    """
    Returns number of nonzeros per row based on distance between atoms.
    This is based on basis set rather than atoms, so there is redundancy in the calculations.
    TODO:
    Parallel version with a PETSc vector.
    Distance calculation loop should be over atoms indeed, which will improve performance by nbf/natom.
    If atoms are ordered based on a distance from a pivot point, loop can be reduced to exclude far atoms.
    """
    nbf=len(basis)
    maxdist2 = maxdist * maxdist
    nnzarray=np.ones(nbf,dtype='int16')
    for i in xrange(nbf):
        atomi=basis[i].atom
        for j in xrange(i+1,nbf):
            distij2=atomi.dist2(basis[j].atom)
            if distij2 < maxdist2: nnzarray[i] += 1
    return nnzarray   

def getNnzInfo(basis,maxdist):
    """
    Returns number of nonzeros per row and  bandwidth per row, based on distance between atoms.
    This is based on basis set rather than atoms, so there is redundancy in the calculations.
    TODO:
    Parallel version with a PETSc vector.
    Distance calculation loop should be over atoms indeed, which will improve performance by nbf/natom.
    If atoms are ordered based on a distance from a pivot point, loop can be reduced to exclude far atoms.
    """
    import constants as const
    nbf=len(basis)
    maxdist2 = maxdist * maxdist * const.ang2bohr**2.
    nnzarray=np.ones(nbf,dtype='int16')
    bwarray=np.zeros(nbf,dtype='int16')
    for i in xrange(nbf):
        atomi=basis[i].atom
        for j in xrange(i+1,nbf):
            distij2=atomi.dist2(basis[j].atom)
            if distij2 < maxdist2: 
                nnzarray[i] += 1
                bwarray[i]   = j - i + 1
    maxnnz = max(nnzarray)
    maxbw = max(bwarray)
    sumnnz = sum(nnzarray)
    avgnnz = sumnnz / float(nbf)
    dennnz = sumnnz / (nbf*(nbf+1)/2.0)  * 100.
    write("Maximum nonzeros per row: {0}".format(maxnnz))
    write("Maximum bandwidth       : {0}".format(maxbw))
    write("Average nonzeros per row: {0}".format(avgnnz))
    write("Total number of nonzeros: {0}".format(sumnnz))
    write("Nonzero density percent : {0}".format(dennnz))            
    return nnzarray, bwarray   

def getSparsityInfo(BCSR, nbf, maxdensity, savefig=True):
    import matplotlib.pyplot as plt
    plt.figure()
    nnz = BCSR.nnz
    density = 100.*nnz / nbf / nbf
    bw = pt.getCSRBandwidth(BCSR)
    write("Number of basis functions: %i" % (nbf))
    write("Nonzero density: %i" % (100.*nnz / nbf / nbf))
    write("Bandwidth: %i" % (bw[1]))
    plt.title(("N={0}, nnz={1}, density={2:.1f}, maxd={3:.2f}".format(nbf, nnz, density, maxdist)))
    plt.spy(BCSR, precision='present', marker='.', markersize=4, mec='black', mfc='black')  # , marker='.', markersize=1)
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_ticks([])
    frame1.axes.get_yaxis().set_ticks([])
    if savefig: plt.savefig("sparsity.eps")
    return

def allReduceAIJ(A,op=MPI.SUM):
    """
    
    """
    B=A.duplicate()
    rstart, rend = A.getOwnershipRange()
    for i in xrange(rstart,rend):
        cols,vals = A.getRow(i)
        sendbuf=np.zeros(len(cols),dtype='d')
        recvbuf=np.zeros(len(cols),dtype='d')
        sendbuf=vals
        MPI.COMM_WORLD.Allreduce(sendbuf,recvbuf,op=MPI.SUM)
        B.setValues(i,cols,recvbuf,addv=PETSc.InsertMode.ADD_VALUES)
    B.assemble()    
    return B

def getCartComm(n,comm=PETSc.COMM_WORLD):
    """
        PetscErrorCode EigCommCreate(MPI_Comm comm,PetscMPIInt nprocEps,EIG_Comm **commEig) 
    {
      PetscErrorCode ierr;
      EIG_Comm       *commEig_tmp; 
      PetscMPIInt    id,nproc,dims[2],periods[2],belongs[2],idRow,nprocRow,idCol,nprocCol;
      MPI_Comm       commCart,rowComm,colComm;
      PetscBool      flag;
    
      PetscFunctionBegin;
      ierr  = MPI_Comm_size(comm,&nproc);CHKERRQ(ierr);
      ierr  = MPI_Comm_rank(comm,&id);CHKERRQ(ierr); 
    
      dims[1] = nproc/nprocEps;       /* nprocRow = nprocMat */
      dims[0] = nprocEps;             /* nprocCol */
      if (dims[0]*dims[1] != nproc) {
        SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"nprocMat %d * nprocEps %d != nproc %d",dims[0],dims[1],nproc);
      } 
    
      periods[0] = periods[1] = 0;
      ierr = MPI_Cart_create(comm,2,dims,periods,0,&commCart);CHKERRQ(ierr);
    
      /* create rowComm = matComm */
      belongs[0] = 0;
      belongs[1] = 1; /* this dim belongs to rowComm */
      ierr = MPI_Cart_sub(commCart,belongs,&rowComm);CHKERRQ(ierr);
      ierr = MPI_Comm_size(rowComm,&nprocRow);CHKERRQ(ierr);
      ierr = MPI_Comm_rank(rowComm,&idRow);CHKERRQ(ierr);
    
      /* create colComm = epsComm */
      belongs[0] = 1;  /* this dim belongs to colComm */
      belongs[1] = 0; 
      ierr = MPI_Cart_sub(commCart,belongs,&colComm);CHKERRQ(ierr);
      ierr = MPI_Comm_size(colComm,&nprocCol);CHKERRQ(ierr);
      ierr = MPI_Comm_rank(colComm,&idCol);CHKERRQ(ierr);
    
      ierr = PetscNew(&commEig_tmp);CHKERRQ(ierr);
      *commEig = commEig_tmp;
      commEig_tmp->world    = comm;
      commEig_tmp->id       = id;
      commEig_tmp->idMat    = idRow; 
      commEig_tmp->idEps    = idCol; 
      commEig_tmp->nproc    = nproc;
      commEig_tmp->nprocMat = dims[1]; 
      commEig_tmp->nprocEps = dims[0]; 
      commEig_tmp->mat      = rowComm;
      commEig_tmp->eps      = colComm;
    
      flag = PETSC_FALSE;
      ierr = PetscOptionsHasName(NULL, "-view_commEig", &flag);CHKERRQ(ierr);
      if (flag) {
        ierr = PetscSynchronizedPrintf(comm,"[%D], idEps %D, nprocEps %D, idMat %D, nprocMat %D\n",id,idCol,nprocCol,idRow,nprocRow);
        ierr = PetscSynchronizedFlush(comm,PETSC_STDOUT);CHKERRQ(ierr);
      }
      PetscFunctionReturn(0);   
    }
    """
    pass
    return 

def getMatComm(n,comm=MPI.COMM_WORLD,debug=False):
    orig_group = MPI.COMM_WORLD.group
    rank       = MPI.COMM_WORLD.rank
    size       = MPI.COMM_WORLD.size
    if size%n != 0: write("Comm problem, %d not commensurate with %d:" % (size,n))
    if    n > size: write("Comm problem, %d < %d:" % (size,n))
    pn         = size / n
    firstrank  = rank - rank%pn
    matgroupranks = xrange(firstrank, firstrank+pn)
    if debug: print rank, matgroupranks
    matgroup   = orig_group.Incl(matgroupranks)
    matcomm    = MPI.COMM_WORLD.Create(matgroup)
    return matcomm

def getMatFromFile(filename,comm):
    matreader=PETSc.Viewer().createBinary(filename,mode='r',comm=comm)
    A=PETSc.Mat().create(comm=comm)
    A.setType('sbaij')
    A.setFromOptions()
    A.load(matreader)
    return A

def writeMatToTxtFile(A,filename):
    N=A.getSize()[0]
    B=getSeqAIJ(A)
    for i in xrange(rstart,rend):   
        pass
    return 0
    
def writeMatToBinFile(A,filename):
    N=A.getSize()[0]
    B=getSeqAIJ(A)
    for i in xrange(rstart,rend):   
        pass
    return 0

def getWorldSize():
    return PETSc.COMM_WORLD.size

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

def getTraceProductDiag(A,B):
    """
    Returns the trace of the product of A,B when one of them is a diagonal matrix
    getDiagonal returns a parallel Vec so no need to reduce.
    """
    a = A.getDiagonal()
    b = B.getDiagonal()
    return a.dot(b)

def getTraceProductAIJ(A,B):
    """
    Returns the trace of the product which is:
    sum_i sum_j A(i,j) B(j,i), 
    so it is simpler than taking the product first and then computing the trace.
    """
    temp=0.0
    rstart, rend = B.getOwnershipRange()
    comm = B.getComm().tompi4py()
    for i in xrange(rstart,rend):
        a = A.getRow(i)[1]
        b = B.getRow(i)[1]
        temp += a.dot(b)
    return comm.allreduce(temp,op=MPI.SUM)

def getTraceProductAIJslow(A,B):
    """
    Returns the trace of the product which is:
    sum_i sum_j A(i,j) B(j,i), 
    so it is simpler than taking the product first and then computing the trace.
    """
    N=A.getSize()[0]
    temp=0.0
    rstart, rend = A.getOwnershipRange()
    for i in xrange(rstart,rend):
        temp += A[i,i] * B[i,i]
        for j in xrange(i+1,N):
           #A and B are symmetric
            temp += 2 * A[i,j]*B[i,j]
    return MPI.COMM_WORLD.allreduce(temp,op=MPI.SUM)

def getTrace(A):
    """
    Returns the trace of A i.e. sum_i A_{ii}
    Note:
    Missing in petsc4py, should be easy to add. 
    """
    temp=0.0
    rstart, rend = A.getOwnershipRange() 
    for i in xrange(rstart,rend):
        temp += A[i,i]
    return MPI.COMM_WORLD.allreduce(temp,op=MPI.SUM)

def getSeqMat(A):
    """
    If A is a MPIAIJ matrix returns a seqAIJ matrix identical to A.
    Else returns A
    """
    myis=PETSc.IS().createStride(A.getSize()[0],first=0,step=1,comm=PETSc.COMM_SELF)    
    (A,)=A.getSubMatrices(myis,myis)
    return A

def getRedundantMat(A,nsubcomm, subcomm=None, out=None):
    """
    Create copies of the matrix in subcommunicators
    """
    return A.createRedundantMatrix(nsubcomm,subcomm=subcomm, out=out)
   
def getSeqAIJ(A):
    """
    If A is a MPIAIJ matrix returns a seqAIJ matrix identical to A.
    Else returns A
    """
    matcomm=A.getComm()
    if matcomm.getSize()==1: return A
    N=A.getSize()[0]
    B=PETSc.Mat().create(comm=PETSc.COMM_SELF)
    B.setType(PETSc.Mat.Type.SEQAIJ)
    B.setSizes(N)
    B.setUp()
    rstart, rend = A.getOwnershipRange()
    for i in xrange(rstart,rend):
        cols,vals = A.getRow(i) #maybe restore later
        ncols = len(cols)
        sendbuf=[i,cols,vals]
        matcomm.Allgather(sendbuf,recvbuf)
        B.setValues(Bi,Bcols,Bvals,addv=PETSc.InsertMode.INSERT)
    B.assemble()  
    return B

def getSeqAIJ2(A):
    N=A.getSize()[0]
    B=PETSc.Mat().create(comm=PETSc.COMM_SELF)
    B.setType(PETSc.Mat.Type.SEQAIJ)
    B.setSizes(N)
    B.setUp()
    rank = PETSc.COMM_WORLD.rank
    rstart, rend = A.getOwnershipRange()
    for i in xrange(rstart,rend):
        cols,vals = A.getRow(i) #maybe restore later
    #    print rank, ":", i,cols,vals
        ncols = len(cols)
        k=0
        for j in cols:
            B[i,j]=vals[k]
            k+=1
    B.assemble()  
    return B

def getAij():
    B=PETSc.Mat().create(comm=PETSc.COMM_SELF)
    B.setType(PETSc.Mat.Type.SEQAIJ)
    B.setSizes(2)
    B.setUp()
    B[0,:]=[1,2]
    B[1,:]=[3,4]
    B.assemblyBegin()
    B.assemblyEnd()  
    return B

def getDistAIJslow(N):
    A=PETSc.Mat().create(comm=PETSc.COMM_WORLD)
    A.setType(PETSc.Mat.Type.MPISBAIJ)
    A.setSizes(N,1)
    A.setUp()
    rstart, rend = A.getOwnershipRange()
    for i in xrange(rstart,rend):
        for j in xrange(i,N):
            A[i,j]=j-i
    A.assemble()
    return A

def getDistAIJfast(N):
    A=PETSc.Mat().create(comm=PETSc.COMM_WORLD)
    A.setType(PETSc.Mat.Type.MPISBAIJ)
    A.setSizes(N,1)
    A.setUp()
    A.setPreallocationNNZ(N*(N+1)/2)
    rstart, rend = A.getOwnershipRange()
    for i in xrange(rstart,rend):
        cols=range(i,N)
        vals=[j-i for j in cols]
        A.setValues(i,cols,vals,addv=PETSc.InsertMode.ADD_VALUES)
    A.assemble()
    return A

def convertAIJ2CSR(A):
    import scipy.sparse
    N=A.getSize()[0]
    B=np.zeros((N,N))

    for i in xrange(N):
        for j in xrange(N):
            B[i,j]=A[i,j]
    return scipy.sparse.csr_matrix(B)

def compareAIJB(Aij,B,N,thresh=1.e-5,comment='Comparison'):
    """
    Compares all matrix elements of two matrices based on a given thresh
    Matrix A should be petsc AIJ.
    Matrix B can be AIJ or a numpy CSR matrix stored redundantly at all ranks.
    """

    rank = PETSc.COMM_WORLD.rank
    rstart, rend = Aij.getOwnershipRange()
    k=0
    for i in xrange(rstart,rend):
        for j in xrange(i,N):
            if abs(Aij[i,j]-B[i,j]) > thresh: 
                k += 1
                print('{0}, Rank: {1},!!!!!!!!Differs in {2} {3}: {4} vs {5}'.format(comment,rank,i,j, Aij[i,j],B[i,j]))
    if k==0: write("{0} ok  within {1}".format(comment,thresh))
    return

def compareAB(A,B,N,thresh=1.e-5):
    """
    Compares all matrix elements of two matrices based on a given thresh
    Matrices can be of any sequential type (numpy, csr, SEQAIJ).
    """
    k=0
    for i in xrange(N):
        for j in xrange(N):
            if abs(A[i,j]-B[i,j]) > thresh: 
                k += 1
                write('!!!!!!!!Differs in %i,%i,%i,%i' % (i,j, A[i,j],B[i,j]))
    if k==0: write("No difference within %f" % thresh)
    return      

def convertA2CSR(A,spfilter=0):
    """
    Given a  numpy matrix A, returns scipy csr matrix
    Optional parameter spfilter can filter matrix element with abs value smaller than spfilter
    if spfilter > 0:
        A < spfilter = 0
    """
    import scipy.sparse
    if spfilter>0: 
        Acsr=scipy.sparse.csr_matrix(A)
        Acsr=Acsr.multiply(abs(Acsr)>spfilter)
        write("Nonzero density: %3g:" % (float(Acsr.nnz*100)/float(Acsr.shape[0]*Acsr.shape[0])))
    else: 
        Acsr=scipy.sparse.csr_matrix(A)
    return Acsr


def convertCSR2AIJ(Acsr,comm=PETSc.COMM_WORLD):
    """
    Given a scipy csr matrix converts to PETSC AIJ matrix
    """
    A = PETSc.Mat().createAIJ(Acsr.shape[0])
    print 'size',A.getSize()
    A.setUp()
    rstart, rend = A.getOwnershipRange()
    return A.createAIJ(size=Acsr.shape[0],csr=(Acsr.indptr[rstart:rend+1] - Acsr.indptr[rstart],
                                                      Acsr.indices[Acsr.indptr[rstart]:Acsr.indptr[rend]],
                                                      Acsr.data[Acsr.indptr[rstart]:Acsr.indptr[rend]]),comm=comm) 
def convert2SeqVec(xr):
    xr_size = xr.getSize()
    seqx = PETSc.Vec()
    seqx.createSeq(xr_size,comm=PETSc.COMM_SELF)
    seqx.setFromOptions()
   # seqx.set(0.0)
    fromIS = PETSc.IS().createGeneral(range(xr_size),comm=PETSc.COMM_SELF)
    toIS = PETSc.IS().createGeneral(range(xr_size),comm=PETSc.COMM_SELF)
    sctr=PETSc.Scatter().create(xr,fromIS,seqx,toIS)
    sctr.begin(xr,seqx,addv=PETSc.InsertMode.INSERT,mode=PETSc.ScatterMode.FORWARD)
    sctr.end(xr,seqx,addv=PETSc.InsertMode.INSERT,mode=PETSc.ScatterMode.FORWARD)
    return seqx

def main():
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
    opts = PETSc.Options()
    M = opts.getInt('M', 32)
    n = opts.getInt('n', 1)
    matcomm=getMatComm(n, comm=MPI.COMM_WORLD)
    A = np.zeros((M,M))
    A[0,0] = rank + 1 
    A[M-1,M-1] = 10*rank
    Acsr=convert2csr(A)
    Aij= PETSc.Mat().createAIJ(M,comm=matcomm)
    Aij.setUp()
    Aij[0,0] = rank + 1 
    Aij[M-1,M-1] = 11*(rank+1)
    Aij.assemble()
    B=allReduceAIJ(Aij,matcomm) 
    printAIJ(B, 'B')   
#    if rank==0:Aij=convert2Aij(Acsr, comm=matcomm)
    #Aij=convert2Aij(Acsr, comm=matcomm)
   # Aij=convert2Aij(Acsr, comm=PETSc.COMM_WORLD)
     
    """
    if N==32: N=size
    stage = PETSc.Log.Stage('slow');    stage.push();
    A=getDistAIJslow(N)
    stage.pop(); stage = PETSc.Log.Stage('fast');    stage.push();
    B=getDistAIJfast(N)
    stage.pop();stage = PETSc.Log.Stage('diff');    stage.push();
    compareAijB(A, B, N)
    stage.pop()
    """
    
    """
    Checks to see convert2AIJ is working
    """
    """
    A=np.zeros((N,N))
    for i in xrange(N):
        A[i,:]=np.arange(N)+i
    for i in xrange(N):
        for j in xrange(N):
            write(' %i,%i:%f' % (i,j, A[i,j]) )
    x = PETSc.Vec()
    x.create(comm=PETSc.COMM_WORLD)
    x.setSizes((rank+1,None))
    x.setFromOptions()
    x.set(1.0)
    for i in xrange(N-rank):
        x.setValues(i,1,addv=PETSc.InsertMode.ADD_VALUES)
    x.assemble()
    y=convert2seqVec(x) 
    for i in xrange(N*(N+1)/2):
        write(y[i])      
    Acsr = convert2csr(A)
    Aij = convert2Aij(Acsr)
    Aijseq = getSeqAIJ2(Aij)
    B = getAij()
    write('Compare Aij to Acsr')
    compareAijB(Aij,Acsr,N)
    write('Compare Aijseq to A')
    compareAijB(Aijseq,A,N)
    """
if __name__ == '__main__':
    main()
