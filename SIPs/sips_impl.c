#include "sips_impl.h"

#include <../../petsc/src/mat/impls/sbaij/mpi/mpisbaij.h>
#include <slepc/private/epsimpl.h>
#include <../../slepc/src/eps/impls/krylov/krylovschur/krylovschur.h>

#undef __FUNCT__
#define __FUNCT__ "EPSKrylovSchurGetLocalInterval"
PetscErrorCode EPSKrylovSchurGetLocalInterval(EPS eps,PetscReal **interval,PetscInt **inertia)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSKrylovSchurGetSubComm"
PetscErrorCode EPSKrylovSchurGetSubComm(EPS eps,MPI_Comm *matComm,MPI_Comm *epsComm)
{
  PetscErrorCode    ierr;
  PetscMPIInt       npart;
  EPS_KRYLOVSCHUR   *ctx=(EPS_KRYLOVSCHUR*)eps->data;
  PetscMPIInt       size;
  MPI_Comm          comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)eps,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  npart = ctx->npart;

  /* create subcommunicators */
  if (npart == 1) {
    *matComm = comm;
    *epsComm = PETSC_COMM_SELF;
  } else {
    PetscMPIInt     dims[2],periods[2],belongs[2];
    MPI_Comm        commCart;
    PetscSubcomm    subc = ctx->subc;

    *matComm = PetscSubcommChild(subc);CHKERRQ(ierr);

    dims[0] = npart;      /* nprocCol */
    dims[1] = size/npart; /* nprocMat */
    if (dims[0]*dims[1] != size) {
      SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"nprocMat %d * npart %d != nproc %d",dims[0],dims[1],size);
    }
    periods[0] = periods[1] = 0;
    ierr = MPI_Cart_create(comm,2,dims,periods,0,&commCart);CHKERRQ(ierr);

    /* create colComm = epsComm */
    belongs[0] = 1;  /* this dim belongs to epsComm */
    belongs[1] = 0;
    ierr = MPI_Cart_sub(commCart,belongs,epsComm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------
   Calculate a weighted density matrix P from eigenvectors.
   P is in SBAIJ format (upper triangular part),
   P(row,col) = sum_(i) { weight[i] * evec[i,row] * evec[i,col] }
 ----------------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "EPSCreateDensityMat"
PetscErrorCode EPSCreateDensityMat(EPS eps, PetscInt idx_start,PetscInt idx_end,Mat *P)
{
  PetscErrorCode    ierr;
  MPI_Comm          matComm,epsComm;
  PetscInt          v,nconv_loc,nz,i,mbs;
  PetscMPIInt       idMat,idEps,nprocMat,nprocEps;
  Vec               evec;
  Mat               A,Dmat;
  PetscScalar       *pv,*buf,lambda;
  PetscInt          *pi,*pj,ncols,row,j,col,ns,*inertias,myinertia[2],k,nskip;
  PetscReal         *shifts,myinterval[2]; 
  const PetscScalar *evec_arr;
  EPS_KRYLOVSCHUR   *ctx=(EPS_KRYLOVSCHUR*)eps->data;
  PetscMPIInt       size,rank;
  MPI_Comm          comm;

  PetscFunctionBegin;
  /* create subcommunicators */
  ierr = PetscObjectGetComm((PetscObject)eps,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
 
  if (!rank) printf("\nEPSCreateDensityMat: idx_start/end: %d, %d\n",idx_start,idx_end); 
  ierr = EPSKrylovSchurGetSubComm(eps,&matComm,&epsComm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(matComm,&idMat);CHKERRQ(ierr);
  ierr = MPI_Comm_size(matComm,&nprocMat);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(epsComm,&idEps);CHKERRQ(ierr);
  ierr = MPI_Comm_size(epsComm,&nprocEps);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] idMat %d, nprocMat %d; idEps %d, nprocEps %d\n",rank,idMat,nprocMat,idEps,nprocEps);

  /* get num of local converged eigensolutions */
  ierr = EPSKrylovSchurGetSubcommInfo(eps,&idEps,&nconv_loc,&evec);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] idEps %d, nconv_loc %d\n",rank,idEps,nconv_loc);

  /* get local operator A */
  if (nprocEps == 1) {
    ierr = EPSComputeVectors(eps);CHKERRQ(ierr);
    ierr = EPSGetOperators(eps,&A,NULL);CHKERRQ(ierr);
  } else {
    ierr = EPSComputeVectors(ctx->eps);CHKERRQ(ierr);
    ierr = EPSGetOperators(ctx->eps,&A,NULL);CHKERRQ(ierr);
  }

  /* find myinterval and myinertia */
  ierr = EPSKrylovSchurGetInertias(eps,&ns,&shifts,&inertias);CHKERRQ(ierr);

  /* myinterval */
  if (nprocEps == 1) {
    ierr = EPSGetInterval(eps,&myinterval[0],&myinterval[1]);CHKERRQ(ierr);
  } else {
    myinterval[0] = ctx->subintervals[idEps];
    myinterval[1] = ctx->subintervals[idEps + 1];
  }
  
  /* myinertia */
  i = idEps;
  while (i<ns) {
    if (shifts[i] == myinterval[0]) {
      myinertia[0] = inertias[i++];
      break;
    }
    i++;
  }
  while (i<ns) {
    if (shifts[i] == myinterval[1]) {
      myinertia[1] = inertias[i++];
      break;
    }
    i++;
  }
  ierr = PetscFree(shifts);CHKERRQ(ierr);
  ierr = PetscFree(inertias);CHKERRQ(ierr);
  ierr = MPI_Barrier(comm);CHKERRQ(ierr);
  if (myinertia[1] < idx_start || myinertia[0] > idx_end) {
    nconv_loc = 0;
  }

  ierr = MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,&Dmat);CHKERRQ(ierr);

  if (nprocMat == 1) { 
    /* P and evec are sequential */
    /*---------------------------*/
    Mat_SeqSBAIJ *pp=(Mat_SeqSBAIJ*)Dmat->data;
   
    pv = pp->a; pi = pp->i; pj = pp->j;
    mbs = pp->mbs; /* dim of P */
    nz = pp->nz;

    /* initialize P as a zero matrix */
   // ierr = PetscMemzero(pv,nz*sizeof(PetscScalar));CHKERRQ(ierr);

    k     = myinertia[0];
    nskip = 0;
    for (v=0; v<nconv_loc; v++) {
      if (k < idx_start) {
        k++; /* skip this evec */
        nskip++;
        continue;
      } else if (k == idx_end) {
        nskip += nconv_loc - v;
        break;
      }
      k++;
      
      ierr = EPSKrylovSchurGetSubcommPairs(eps,v,&lambda,evec);CHKERRQ(ierr);
#if 0
      if (idMat == 0 && idEps == 0 && nprocMat == 1 && v==0) {
        PetscScalar norm;
        Vec         x;
        Mat         B = NULL;
        ierr = MatCreateVecs(A,NULL,&x);CHKERRQ(ierr);
        if (B) {
          ierr = MatMult(B,evec,x);CHKERRQ(ierr); /* x = B*evec  */
          ierr = VecTDot(x,evec,&norm);CHKERRQ(ierr);
          printf("%d |evec|_B = %g\n",v,norm);
        } else {
          ierr = VecNorm(evec,NORM_2,&norm);CHKERRQ(ierr);
          printf("%d |evec| = %g\n",v,norm);
        }
        ierr = VecDestroy(&x);CHKERRQ(ierr);
      }
#endif

      ierr = VecGetArrayRead(evec,&evec_arr);   
      for (row=0; row<mbs; row++) {
        ncols = pi[row+1] - pi[row];
        for (j=0; j<ncols; j++){
          col = pj[pi[row]+j];     /* col index */
          pv[pi[row]+j] += evec_arr[row]*evec_arr[col]; /* P(row,col) */
          //printf(" v=%d: evec[%d]=%g, evec[%d]=%g; pv[%d]=%g\n",v,row,evec_arr[row],col,evec_arr[col],pi[row]+j,pv[pi[row]+j]); 
        }
      }      
      ierr = VecRestoreArrayRead(evec,&evec_arr);CHKERRQ(ierr);   
    } /* endof for (v=0; v<nconv_loc; v++) */
    
    if (nprocEps > 1) { /* sum Dmat over epsComm */         
      ierr = PetscMalloc((nz+1)*sizeof(PetscScalar),&buf);CHKERRQ(ierr);
      for (i=0; i<nz; i++) buf[i] = pv[i];
      ierr = MPI_Allreduce(buf,pv,nz, MPI_DOUBLE,MPI_SUM,epsComm);CHKERRQ(ierr);
      ierr = PetscFree(buf);CHKERRQ(ierr);
    }
    if (idEps==100) {
      ierr = MatView(Dmat,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
    }

  } else { 
    /* nprocMat > 1 */
    /*--------------*/
    PetscInt     nzA,nzB,lvec_size;
    Mat_MPISBAIJ *pp =(Mat_MPISBAIJ*)Dmat->data;
    Mat_SeqSBAIJ *pA =(Mat_SeqSBAIJ*)(pp->A->data);
    Mat_SeqBAIJ  *pB =(Mat_SeqBAIJ*)(pp->B->data);
    mbs = pA->mbs;
    nzA = pA->nz; nzB = pB->nz;
    nz  = PetscMax(nzA, nzB);
    
    /* pA part */
    pv = pA->a; pi = pA->i; pj = pA->j;
    mbs = pA->mbs; /* dim of P */
    ierr = PetscMemzero(pv,nzA*sizeof(PetscScalar));CHKERRQ(ierr);

    k     = myinertia[0];
    nskip = 0;
    for (v=0; v<nconv_loc; v++) {
      if (k < idx_start) {
        k++; /* skip this evec */
        nskip++;
        continue;
      } else if (k == idx_end) {
        nskip += nconv_loc - v;
        break;
      }
      k++;

      ierr = EPSKrylovSchurGetSubcommPairs(eps,v,&lambda,evec);CHKERRQ(ierr);
      ierr = VecGetArrayRead(evec,&evec_arr);   
      for (row=0; row<mbs; row++) {
        ncols = pi[row+1] - pi[row];
        for (j=0; j<ncols; j++){
          col = pj[pi[row]+j];     /* col index */
          pv[pi[row]+j] += evec_arr[row]*evec_arr[col]; /* P(row,col) */
          //printf(" v=%d: evec[%d]=%g, evec[%d]=%g; pv[%d]=%g\n",v,row,evec_arr[row],col,evec_arr[col],pi[row]+j,pv[pi[row]+j]); 
        }
      }   
      ierr = VecRestoreArrayRead(evec,&evec_arr);CHKERRQ(ierr);
    } /* endof for (v=0; v<nconv_loc; v++) */
    //ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] nconv_loc = %d\n",rank,nconv_loc-nskip);

    if (nprocEps > 1) { /* sum Dmat over commEps */
      ierr = PetscMalloc((nz+1)*sizeof(PetscScalar),&buf);CHKERRQ(ierr);
      for (i=0; i<nzA; i++) buf[i] = pv[i];
      ierr = MPI_Allreduce(buf,pv,nzA, MPI_DOUBLE,MPI_SUM,epsComm);CHKERRQ(ierr);
    }

    /* pB part */
    pv = pB->a; pi = pB->i; pj = pB->j;
    ierr = PetscMemzero(pv,nzB*sizeof(PetscScalar));CHKERRQ(ierr);

    for (v=0; v<nconv_loc; v++) {
      ierr = EPSKrylovSchurGetSubcommPairs(eps,v,&lambda,evec);CHKERRQ(ierr);

      /* get evec components from other processes */
      ierr = VecGetSize(pp->lvec,&lvec_size);CHKERRQ(ierr);
      ierr = VecScatterBegin(pp->Mvctx,evec,pp->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(pp->Mvctx,evec,pp->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      
      ierr = VecGetArrayRead(evec,&evec_arr);CHKERRQ(ierr);
      
      if (lvec_size){
        const PetscScalar *lvec_arr;
        ierr = VecGetArrayRead(pp->lvec,&lvec_arr);CHKERRQ(ierr);
        for (row=0; row<mbs; row++) {
          ncols = pi[row+1] - pi[row];
          for (j=0; j<ncols; j++){
            col = pj[pi[row]+j];     /* index of pp->lvec */
            pv[pi[row]+j] += evec_arr[row]*lvec_arr[col]; /* P(row,col) */
            //printf("[%d,%d],row %d, col %d, pv[%d]=%g\n",idMat,idEps,row,col,pi[row]+j,pv[pi[row]+j]);
          }
        }
        ierr = VecRestoreArrayRead(pp->lvec,&lvec_arr);CHKERRQ(ierr);      
      } 
      ierr = VecRestoreArrayRead(evec,&evec_arr);CHKERRQ(ierr);
    } /* endof for (v=0; v<nconv_loc; v++) */
    if (nprocEps >1){ /* sum pB over commEps */
      if (nzB){
        for (i=0; i<nzB; i++) buf[i] = pv[i];
        ierr = MPI_Allreduce(buf,pv,nzB, MPI_DOUBLE,MPI_SUM, epsComm);CHKERRQ(ierr);
      }
      ierr = PetscFree(buf);CHKERRQ(ierr);
    }
  }
  if (idMat == 0) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] interval [%g, %g], inertia %d %d; nconv_loc %d\n",rank,myinterval[0],myinterval[1],myinertia[0],myinertia[1],nconv_loc-nskip);CHKERRQ(ierr);
  }
  ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] density done",rank);CHKERRQ(ierr);

  *P = Dmat;
  ierr = VecDestroy(&evec);CHKERRQ(ierr);
  if (idEps == 100) {
    ierr = MatView(Dmat,PETSC_VIEWER_STDOUT_(matComm));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* -----------------------------------------------------------------------------
  input:
    P, B - matrix in SBAIJ format, same data structure, same copy for each commMat[]
    sgn - sign for summation of opposite triangular sides
         sgn = 0        for symmetric
         sgn = +1 or -1 for antisymmetric
            (contribution from either lower or upper triangle is inverted)
    vdiag       - seq vector in all processes
  output:
    vdiag       - diag(P*B) in all processes
    tr           - tr(P*B)
--------------------------------------------------------------------------- */

#undef __FUNCT__
#define __FUNCT__ "MatMatMultGetTrace"
PetscErrorCode MatMatMultGetTrace(EPS eps,Mat P,Mat B,PetscInt sgn,Vec vdiag,PetscReal *tr)
{
  PetscErrorCode    ierr;
  PetscInt          j,col,ncols,rstart,rend,row,M;
  const PetscInt    *cols;
  const PetscScalar *Pvals,*Bvals;
  PetscReal         *rsum,pb,*buf;
  PetscInt          rsign,csign,dsign;
  MPI_Comm          matComm;
  EPS_KRYLOVSCHUR   *ctx=(EPS_KRYLOVSCHUR*)eps->data;
  PetscMPIInt       nprocB;
  PetscInt          npart=ctx->npart;
  Mat               myB;

  PetscFunctionBegin;
  ierr = VecGetSize(vdiag,&M);CHKERRQ(ierr);
  ierr = VecGetArray(vdiag,&rsum);CHKERRQ(ierr);
  ierr = PetscMalloc(M*sizeof(PetscReal),&buf);CHKERRQ(ierr);
  ierr = PetscMemzero(buf,M*sizeof(PetscReal));CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(P,&rstart,&rend);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)eps,&matComm);CHKERRQ(ierr);
  if (npart == 1) {
    myB = B;
    ierr = MPI_Comm_size(matComm,&nprocB);CHKERRQ(ierr);
  } else {
    matComm = PetscSubcommChild(ctx->subc);CHKERRQ(ierr);
    ierr = MPI_Comm_size(matComm,&nprocB);CHKERRQ(ierr);
    ierr = EPSGetOperators(ctx->eps,NULL,&myB);CHKERRQ(ierr);
  }

  ierr = MatSetOption(P,MAT_GETROW_UPPERTRIANGULAR,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatSetOption(myB,MAT_GETROW_UPPERTRIANGULAR,PETSC_TRUE);CHKERRQ(ierr);

  /* Signs for contributions from upper/lower triangles, and diagonal.
   * Diagonal is skipped when the summation is to be anti-symmetric.
   */
  rsign = csign = dsign = 1;
  if (sgn > 0) { rsign = -1; dsign = 0; }
  if (sgn < 0) { csign = -1; dsign = 0; }

  for (row = rstart; row < rend; row++){
    ierr = MatGetRow(P,row,&ncols,&cols,&Pvals);CHKERRQ(ierr);
    ierr = MatGetRow(myB,row,&ncols,PETSC_NULL,&Bvals);CHKERRQ(ierr);
    for (j=0; j<ncols; j++){
      col = cols[j];
      pb  = Pvals[j]*Bvals[j]; /* P(row,col)*B(row,col) */

      if (col == row) {
	  buf[row] += dsign * pb;
      } else {
	  buf[row] += rsign * pb;
	  buf[col] += csign * pb;
      }
    }
    ierr = MatRestoreRow(P,row,&ncols,&cols,&Pvals);
    ierr = MatRestoreRow(myB,row,&ncols,PETSC_NULL,&Bvals);
  }

  if (nprocB == 1){ /* sequential P and B*/
    for (row = rstart; row < rend; row++) rsum[row] = buf[row];
  } else {          /* mpi P and B */
    ierr = MPI_Allreduce(buf,rsum,M,MPI_DOUBLE,MPI_SUM,matComm);
  }
  *tr = 0.0;
  for (row = 0; row < M; row++) *tr += rsum[row];

  ierr = PetscFree(buf);CHKERRQ(ierr);
  ierr = VecRestoreArray(vdiag,&rsum);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
