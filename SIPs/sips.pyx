from petsc4py.PETSc cimport Mat,  PetscMat
from slepc4py.SLEPc cimport EPS, SlepcEPS
cimport petsc4py
from petsc4py.PETSc import Error

cdef extern from "sips_impl.h":
    int EPSCreateDensityMat(SlepcEPS eps,int idx_start,int idx_end,PetscMat*);


def getDensityMat(EPS eps, int i, int j):
    cdef int ierr
    cdef Mat A = Mat()
    ierr = EPSCreateDensityMat(eps.eps,i,j,&A.mat)
    if ierr != 0: raise Error(ierr)
    return A
