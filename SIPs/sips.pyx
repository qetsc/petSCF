from petsc4py.PETSc cimport Mat,  PetscMat, Vec, PetscVec
from slepc4py.SLEPc cimport EPS, SlepcEPS
from petsc4py.PETSc import Error

cimport petsc4py
from petsc4py import PETSc
import numpy as np
cimport numpy as np
cimport cython

cdef extern from "sips_impl.h":
    int EPSCreateDensityMat(SlepcEPS eps,int idx_start,int idx_end,PetscMat*);
    int getF(int atomids,PetscMat,PetscMat,PetscMat, PetscMat, PetscMat*);


def getDensityMat(EPS eps, int i, int j):
    cdef int ierr
    cdef Mat A = Mat()
    ierr = EPSCreateDensityMat(eps.eps,i,j,&A.mat)
    if ierr != 0: raise Error(ierr)
    return A

def getFc(atomids,Mat T,Mat D, Mat GH1, Mat GH2):
    cdef int ierr
    cdef Mat F = Mat()
    ierr = getF(atomids, T.mat, D.mat, GH1.mat, GH2.mat, &F.mat)
    if ierr != 0: raise Error(ierr)
    return F    

DTYPE_INT  = np.int
DTYPE_FLOAT  = np.float32
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix. MK: From cython.org
ctypedef np.int_t DTYPE_INT_t
ctypedef np.float32_t DTYPE_FLOAT_t
@cython.boundscheck(False)
def getFCython(np.ndarray[DTYPE_INT_t, ndim=1] atomids, Mat T,Mat D, Mat GH1, Mat GH2):
    """
    Returns density matrix dependent terms of the Fock matrix.
    Parameters
    ----------
    atomids: 1D int array
             Length of array should be equal to size 
             of the matrix.
    T,D,
    GH1,GH2: Petsc aij mat of same size and 
             same nnz pattern.
    GH1 =        G - 0.5 * H
    GH2 = -0.5 * G + 1.5 * H
    TODO
    ----
    Vectorize inner loop
    """
    cdef int i,j,k,kdiag,rstart,rend,atomidi
    cdef double tmpii,tmpij,Dij,Djj,Tij
    cdef np.ndarray[int, ndim=1] cols
    cdef np.ndarray[double, ndim=1] diag,valsT,valsD,valsGH1,valsGH2
#    cdef np.ndarray[DTYPE_FLOAT_t,ndim=1] localdiagD,localdiagGH1,diagD,valsT,valsD,valsGH1,valsGH2
    cdef Vec localdiagD,localdiagGH1,diagD
    cdef Mat A
    localdiagD   = D.getDiagonal()
    localdiagGH1 = GH1.getDiagonal()
    diag    = localdiagGH1.array_r * localdiagD.array_r
    diagD   = convert2SeqVec(localdiagD) 
    A       = T.duplicate( )
    rstart, rend = A.getOwnershipRange()
    for i in xrange(rstart,rend):
        atomidi      = atomids[i]
        cols         = T.getRow(i)[0]
        valsT        = T.getRow(i)[1]
        valsD        = D.getRow(i)[1]
        valsGH1      = GH1.getRow(i)[1]
        valsGH2      = GH2.getRow(i)[1]
        valsF        = np.zeros_like(valsT)
        tmpii        = diag[i-rstart] # 0.5 * diagD[i] * G[i,i] # Note G[i,i]=H[i,i]
        for k,j in enumerate(cols):
            if i != j:
                Djj   = diagD.array_r[j] # D[j,j]
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
        A.setValues(i,cols,valsF)        
    A.assemble()
    return A

def convert2SeqVec(xr):
    xr_size = xr.getSize()
    seqx = PETSc.Vec()
    seqx.createSeq(xr_size,comm=PETSc.COMM_SELF)
    seqx.setFromOptions()
    fromIS = PETSc.IS().createGeneral(range(xr_size),comm=PETSc.COMM_SELF)
    toIS = PETSc.IS().createGeneral(range(xr_size),comm=PETSc.COMM_SELF)
    sctr=PETSc.Scatter().create(xr,fromIS,seqx,toIS)
    sctr.begin(xr,seqx,addv=PETSc.InsertMode.INSERT,mode=PETSc.ScatterMode.FORWARD)
    sctr.end(xr,seqx,addv=PETSc.InsertMode.INSERT,mode=PETSc.ScatterMode.FORWARD)
    return seqx