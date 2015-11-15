#ifndef SIPS_H
#define SIPS_H

#include <slepceps.h>
#include <petsc/private/petscimpl.h>
PetscErrorCode EPSKrylovSchurGetLocalInterval(EPS eps,PetscReal **interval,PetscInt **inertia);
PetscErrorCode EPSKrylovSchurGetSubComm(EPS eps,MPI_Comm *matComm,MPI_Comm *epsComm);
PetscErrorCode EPSCreateDensityMat(EPS eps,PetscInt idx_start,PetscInt idx_end,Mat *P);
PetscErrorCode EPSSolveCreateDensityMat(EPS eps,PetscInt idx_start,PetscInt idx_end,Mat *P);
PetscErrorCode MatMatMultGetTrace(EPS eps,Mat P,Mat B,PetscInt sgn,Vec vdiag,PetscReal *tr);
#endif/*SIPS_H*/
