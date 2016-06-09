#ifndef SIPS_H
#define SIPS_H

#include <slepceps.h>
#include <petsc/private/petscimpl.h>
PetscErrorCode EPSKrylovSchurGetLocalInterval(EPS eps,PetscReal **interval,PetscInt **inertia);
PetscErrorCode EPSKrylovSchurGetSubComm(EPS eps,MPI_Comm *matComm,MPI_Comm *epsComm);
PetscErrorCode EPSCreateDensityMat(EPS eps,PetscInt idx_start,PetscInt idx_end,Mat *P);
PetscErrorCode MatMatMultGetTrace(EPS eps,Mat P,Mat B,PetscInt sgn,Vec vdiag,PetscReal *tr);
PetscErrorCode getF(PetscInt *atomids,Mat T,Mat D,Mat GH1,Mat GH2, Mat *F);

#endif/*SIPS_H*/
