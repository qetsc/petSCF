#include "sips_impl.h"

#include <../../petsc/src/mat/impls/sbaij/mpi/mpisbaij.h>
#include <slepc/private/epsimpl.h>
#include <../../slepc/src/eps/impls/krylov/krylovschur/krylovschur.h>

#include <petsctime.h>
#include <slepceps.h>
typedef struct {
  PetscLogDouble time;
} DFTBCtx;

typedef struct {
  PetscInt ngaps;
  PetscReal *gap;
} DFTBGap;

/*====================================================================*/
/*
  EPSGetEigenClusters - Get eigenclusters from array of eigenvalues

   Input Parameter:
-  eps - eigensolver context obtained from EPSCreate()
.  eval_deg - tolerance for eigenvalue separation: if |eig[i] - eig[j]| < eval_deg, eig[i] and eig[j] belong to the same eigencluter
+  ecluster, mecluster - arrays to store eigenclusters and multiplicities

   Output Parameter:
-  neclusters_out - number of clusters
+  ecluster, mecluster - eigenclusters and multiplicities
 */
#undef __FUNCT__
#define __FUNCT__ "EPSGetEigenClusters"
PetscErrorCode EPSGetEigenClusters(EPS eps,PetscReal eval_deg,PetscInt* neclusters_out,PetscReal *ecluster,PetscInt *mecluster,DFTBGap *dftbgap)
{
  PetscErrorCode ierr;
  PetscInt       i,neclusters,nconv,ngaps;
  PetscReal      eigr,avegap,r=50.0,eigr_min,eigr_max;
  
  PetscFunctionBegin;
  ierr = EPSGetConverged(eps,&nconv);CHKERRQ(ierr);

  i = 0; neclusters = -1;
  ierr = EPSGetEigenpair(eps,i,&eigr,NULL,NULL,NULL);CHKERRQ(ierr);
  eigr_min = eigr;
  
  neclusters++;
  ecluster[neclusters]  = eigr;
  mecluster[neclusters] = 1;
  for (i=1; i<nconv; i++){
    ierr = EPSGetEigenpair(eps,i,&eigr,NULL,NULL,NULL);CHKERRQ(ierr);
    if (eigr - ecluster[neclusters] < eval_deg){
      mecluster[neclusters]++;
    } else {
      neclusters++; ecluster[neclusters] = eigr; mecluster[neclusters] = 1; 
    }
  }
  eigr_max = eigr;
  neclusters++;
  *neclusters_out = neclusters;

  /* compute gaps */
  avegap = (eigr_max - eigr_min)/nconv;
  //printf(" avegap = %g\n",avegap);
  ngaps = 0;
  for (i=1; i<neclusters; i++) {
    if (ecluster[i] - ecluster[i-1] > r*avegap) ngaps++;
  }
  //printf("ngaps %d\n",ngaps);

  ierr = PetscMalloc1(2*ngaps,&dftbgap->gap);CHKERRQ(ierr);
  ngaps = 0;
  for (i=1; i<neclusters; i++) {
    if (ecluster[i] - ecluster[i-1] > r*avegap) {
      dftbgap->gap[2*ngaps]   = ecluster[i-1];
      dftbgap->gap[2*ngaps+1] = ecluster[i];
      ngaps++;
      //printf("%d - gap (%g, %g), %g\n",i,ecluster[i-1],ecluster[i],ecluster[i] - ecluster[i-1]);
    }
  }
  
  dftbgap->ngaps = ngaps;
  PetscFunctionReturn(0);
}

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

#undef __FUNCT__
#define __FUNCT__ "EPSCreateDensityMat_seqApart"
PetscErrorCode EPSCreateDensityMat_seqApart(EPS eps,Mat Dmat,PetscInt myidx_start,PetscInt myidx_end,Vec evec,PetscMPIInt nprocEps,MPI_Comm epsComm,PetscBool isSBAIJ)
{
  Mat_SeqSBAIJ      *pA;
  Mat_SeqAIJ        *pA1;
  PetscErrorCode    ierr;
  PetscScalar       *pv; 
  PetscInt          *pi,*pj,mbs,nz,v,i,j,row,ncols,col;
  PetscScalar       lambda,*buf;
  const PetscScalar *evec_arr;
  
  PetscFunctionBegin;
  if (isSBAIJ) {
    pA=(Mat_SeqSBAIJ*)Dmat->data;
    pv = pA->a; pi = pA->i; pj = pA->j; mbs = pA->mbs; nz = pA->nz;
  } else { /* isSeqAIJ */
    pA1=(Mat_SeqAIJ*)Dmat->data;
    pv = pA1->a; pi = pA1->i; pj = pA1->j; mbs = Dmat->rmap->n; nz = pA1->nz;
  }

  /* initialize P as a zero matrix */
  ierr = PetscMemzero(pv,nz*sizeof(PetscScalar));CHKERRQ(ierr);

  for (v=myidx_start; v<myidx_end; v++) {   
    ierr = EPSKrylovSchurGetSubcommPairs(eps,v,&lambda,evec);CHKERRQ(ierr);
    ierr = VecGetArrayRead(evec,&evec_arr);   
    for (row=0; row<mbs; row++) {
      ncols = pi[row+1] - pi[row];
      for (j=0; j<ncols; j++){
        col = pj[pi[row]+j];     /* col index */
        pv[pi[row]+j] += 2.0 * evec_arr[row]*evec_arr[col]; /* P(row,col) */
      }
    }      
    ierr = VecRestoreArrayRead(evec,&evec_arr);CHKERRQ(ierr);   
  } 
 
  if (nprocEps > 1) { /* sum Dmat over epsComm */         
    ierr = PetscMalloc((nz+1)*sizeof(PetscScalar),&buf);CHKERRQ(ierr);
    for (i=0; i<nz; i++) buf[i] = pv[i];
    ierr = MPI_Allreduce(buf,pv,nz, MPI_DOUBLE,MPI_SUM,epsComm);CHKERRQ(ierr);
    ierr = PetscFree(buf);CHKERRQ(ierr);
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
PetscErrorCode EPSCreateDensityMat(EPS eps,PetscInt idx_start,PetscInt idx_end,Mat *P)
{
  PetscErrorCode    ierr;
  MPI_Comm          matComm,epsComm;
  PetscInt          v,nconv_loc,i,mbs;
  PetscMPIInt       idMat,idEps,nprocMat,nprocEps;
  Vec               evec;
  Mat               A,Dmat;
  PetscScalar       *pv,*buf,lambda;
  PetscInt          *pi,*pj,ncols,row,j,col,ns,*inertias,myinertia[2];
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
  /* ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] idMat %d, nprocMat %d; idEps %d, nprocEps %d\n",rank,idMat,nprocMat,idEps,nprocEps); */

  /* get num of local converged eigensolutions */
  ierr = EPSKrylovSchurGetSubcommInfo(eps,&idEps,&nconv_loc,&evec);CHKERRQ(ierr);

  /* get local operator A */
  if (nprocEps == 1) {
    ierr = EPSComputeVectors(eps);CHKERRQ(ierr);
    ierr = EPSGetOperators(eps,&A,NULL);CHKERRQ(ierr);
  } else {
    ierr = EPSComputeVectors(ctx->eps);CHKERRQ(ierr);
    ierr = EPSGetOperators(ctx->eps,&A,NULL);CHKERRQ(ierr);
  }

  /* get interval for this process */
  if (nprocEps == 1) {
    ierr = EPSGetInterval(eps,&myinterval[0],&myinterval[1]);CHKERRQ(ierr);
  } else {
    myinterval[0] = ctx->subintervals[idEps];
    myinterval[1] = ctx->subintervals[idEps + 1];
  }
  
  /* get inertia for this process */
  ierr = EPSKrylovSchurGetInertias(eps,&ns,&shifts,&inertias);CHKERRQ(ierr);
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

  PetscInt myidx_start,myidx_end;
  myidx_start = idx_start - myinertia[0];
  myidx_end   = idx_end - myinertia[0];
  if (myidx_start < 0) myidx_start = 0;
  if (myidx_end > nconv_loc) myidx_end = nconv_loc;

  /* create Dmat which has same data structure as A */
  ierr = MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,&Dmat);CHKERRQ(ierr);

  PetscBool      isSBAIJ;
  if (nprocMat == 1) {
    ierr = PetscObjectTypeCompare((PetscObject)Dmat,MATSEQSBAIJ,&isSBAIJ);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectTypeCompare((PetscObject)Dmat,MATMPISBAIJ,&isSBAIJ);CHKERRQ(ierr);
  }
  if (isSBAIJ) {
    PetscInt bs;
    ierr = MatGetBlockSize(Dmat,&bs);CHKERRQ(ierr);
    if (bs != 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only block size 1 is supported");
  } else{
  }
  if (nprocMat == 1) { /* P and evec are sequential */
    ierr = EPSCreateDensityMat_seqApart(eps,Dmat,myidx_start,myidx_end,evec,nprocEps,epsComm,isSBAIJ);CHKERRQ(ierr);
  } else { /* nprocMat > 1, P and evec are distributed */
    PetscInt     nzB,lvec_size;
    Mat          Aloc;
    Vec          lvec;
    VecScatter   Mvctx;
    mbs  = Dmat->rmap->n;
    if (isSBAIJ) {
      PetscPrintf(PETSC_COMM_SELF,"SBAIJ");
      Mat_MPISBAIJ *pp =(Mat_MPISBAIJ*)Dmat->data;
      Mat_SeqBAIJ  *pB =(Mat_SeqBAIJ*)(pp->B->data);
      Aloc = pp->A; lvec  = pp->lvec; Mvctx = pp->Mvctx;
      pv = pB->a; pi = pB->i; pj = pB->j; nzB  = pB->nz;
    } else {
      Mat_MPIAIJ   *pp =(Mat_MPIAIJ*)Dmat->data;
      Mat_SeqAIJ   *pB =(Mat_SeqAIJ*)(pp->B->data);
      Aloc = pp->A; lvec  = pp->lvec; Mvctx = pp->Mvctx;
      pv = pB->a; pi = pB->i; pj = pB->j; nzB  = pB->nz;
    }
   
    /* pA part */
    ierr = EPSCreateDensityMat_seqApart(eps,Aloc,myidx_start,myidx_end,evec,nprocEps,epsComm,isSBAIJ);CHKERRQ(ierr);

    /* pB part */
    ierr = PetscMemzero(pv,nzB*sizeof(PetscScalar));CHKERRQ(ierr);
    for (v=myidx_start; v<myidx_end; v++) {
      ierr = EPSKrylovSchurGetSubcommPairs(eps,v,&lambda,evec);CHKERRQ(ierr);

      /* get evec components from other processes */
      ierr = VecGetSize(lvec,&lvec_size);CHKERRQ(ierr);
      ierr = VecScatterBegin(Mvctx,evec,lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(Mvctx,evec,lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      
      ierr = VecGetArrayRead(evec,&evec_arr);CHKERRQ(ierr);    
      if (lvec_size){
        const PetscScalar *lvec_arr;
        ierr = VecGetArrayRead(lvec,&lvec_arr);CHKERRQ(ierr);
        for (row=0; row<mbs; row++) {
          ncols = pi[row+1] - pi[row];
          for (j=0; j<ncols; j++){
            col = pj[pi[row]+j];     /* index of lvec */
            pv[pi[row]+j] += 2.0 * evec_arr[row]*lvec_arr[col]; /* P(row,col) */
          }
        }
        ierr = VecRestoreArrayRead(lvec,&lvec_arr);CHKERRQ(ierr);      
      } 
      ierr = VecRestoreArrayRead(evec,&evec_arr);CHKERRQ(ierr);
    } 
    if (nprocEps >1 && nzB){ /* sum pB over commEps */
      ierr = PetscMalloc((nzB+1)*sizeof(PetscScalar),&buf);CHKERRQ(ierr);
      for (i=0; i<nzB; i++) buf[i] = pv[i];
      ierr = MPI_Allreduce(buf,pv,nzB, MPI_DOUBLE,MPI_SUM, epsComm);CHKERRQ(ierr);
      ierr = PetscFree(buf);CHKERRQ(ierr);
    }
  }
  if (idMat == 0) {
    nconv_loc = myidx_end-myidx_start;
    if (nconv_loc < 0) nconv_loc = 0;
    ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] interval [%g, %g], inertia %d %d; nconv_loc %d\n",rank,myinterval[0],myinterval[1],myinertia[0],myinertia[1],nconv_loc);CHKERRQ(ierr);
  }

  *P = Dmat;
  ierr = VecDestroy(&evec);CHKERRQ(ierr);
  if (idEps == 100) {
    ierr = MatView(Dmat,PETSC_VIEWER_STDOUT_(matComm));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
/* --------------------------------------------------------------------------
   Calculate a weighted density matrix P from eigenvectors.
   P is in AIJ format ,
   P(row,col) = sum_(i) { weight[i] * evec[i,row] * evec[i,col] }
 ----------------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "EPSSolveCreateDensityMat"
PetscErrorCode EPSSolveCreateDensityMat(EPS eps,PetscInt idx_start,PetscInt idx_end,Mat *P)
{
  PetscErrorCode    ierr;
  MPI_Comm          matComm,epsComm;
  PetscInt          v,nconv_loc,i,mbs;
  PetscMPIInt       idMat,idEps,nprocMat,nprocEps;
  Vec               evec;
  Mat               A,Dmat;
  PetscScalar       *pv,*buf,lambda;
  PetscInt          *pi,*pj,ncols,row,j,col,ns,*inertias,myinertia[2];
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
  /* ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] idMat %d, nprocMat %d; idEps %d, nprocEps %d\n",rank,idMat,nprocMat,idEps,nprocEps); */

  ierr = EPSSolve(eps);CHKERRQ(ierr);

  /* get num of local converged eigensolutions */
  ierr = EPSKrylovSchurGetSubcommInfo(eps,&idEps,&nconv_loc,&evec);CHKERRQ(ierr);

  /* get local operator A */
  if (nprocEps == 1) {
    ierr = EPSComputeVectors(eps);CHKERRQ(ierr);
    ierr = EPSGetOperators(eps,&A,NULL);CHKERRQ(ierr);
  } else {
    ierr = EPSComputeVectors(ctx->eps);CHKERRQ(ierr);
    ierr = EPSGetOperators(ctx->eps,&A,NULL);CHKERRQ(ierr);
  }

  /* get interval for this process */
  if (nprocEps == 1) {
    ierr = EPSGetInterval(eps,&myinterval[0],&myinterval[1]);CHKERRQ(ierr);
  } else {
    myinterval[0] = ctx->subintervals[idEps];
    myinterval[1] = ctx->subintervals[idEps + 1];
  }

  /* get inertia for this process */
  ierr = EPSKrylovSchurGetInertias(eps,&ns,&shifts,&inertias);CHKERRQ(ierr);
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

  PetscInt myidx_start,myidx_end;
  myidx_start = idx_start - myinertia[0];
  myidx_end   = idx_end - myinertia[0];
  if (myidx_start < 0) myidx_start = 0;
  if (myidx_end > nconv_loc) myidx_end = nconv_loc;

  /* create Dmat which has same data structure as A */
  ierr = MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,&Dmat);CHKERRQ(ierr);

  PetscBool      isSBAIJ;
  if (nprocMat == 1) {
    ierr = PetscObjectTypeCompare((PetscObject)Dmat,MATSEQSBAIJ,&isSBAIJ);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectTypeCompare((PetscObject)Dmat,MATMPISBAIJ,&isSBAIJ);CHKERRQ(ierr);
  }
  if (isSBAIJ) {
    PetscInt bs;
    ierr = MatGetBlockSize(Dmat,&bs);CHKERRQ(ierr);
    if (bs != 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only block size 1 is supported");
  } else{
  }
  if (nprocMat == 1) { /* P and evec are sequential */
    ierr = EPSCreateDensityMat_seqApart(eps,Dmat,myidx_start,myidx_end,evec,nprocEps,epsComm,isSBAIJ);CHKERRQ(ierr);
  } else { /* nprocMat > 1, P and evec are distributed */
    PetscInt     nzB,lvec_size;
    Mat          Aloc;
    Vec          lvec;
    VecScatter   Mvctx;
    mbs  = Dmat->rmap->n;
    if (isSBAIJ) {
      PetscPrintf(PETSC_COMM_SELF,"SBAIJ");
      Mat_MPISBAIJ *pp =(Mat_MPISBAIJ*)Dmat->data;
      Mat_SeqBAIJ  *pB =(Mat_SeqBAIJ*)(pp->B->data);
      Aloc = pp->A; lvec  = pp->lvec; Mvctx = pp->Mvctx;
      pv = pB->a; pi = pB->i; pj = pB->j; nzB  = pB->nz;
    } else {
      Mat_MPIAIJ   *pp =(Mat_MPIAIJ*)Dmat->data;
      Mat_SeqAIJ   *pB =(Mat_SeqAIJ*)(pp->B->data);
      Aloc = pp->A; lvec  = pp->lvec; Mvctx = pp->Mvctx;
      pv = pB->a; pi = pB->i; pj = pB->j; nzB  = pB->nz;
    }

    /* pA part */
    ierr = EPSCreateDensityMat_seqApart(eps,Aloc,myidx_start,myidx_end,evec,nprocEps,epsComm,isSBAIJ);CHKERRQ(ierr);

    /* pB part */
    ierr = PetscMemzero(pv,nzB*sizeof(PetscScalar));CHKERRQ(ierr);
    for (v=myidx_start; v<myidx_end; v++) {
      ierr = EPSKrylovSchurGetSubcommPairs(eps,v,&lambda,evec);CHKERRQ(ierr);

      /* get evec components from other processes */
      ierr = VecGetSize(lvec,&lvec_size);CHKERRQ(ierr);
      ierr = VecScatterBegin(Mvctx,evec,lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(Mvctx,evec,lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

      ierr = VecGetArrayRead(evec,&evec_arr);CHKERRQ(ierr);
      if (lvec_size){
        const PetscScalar *lvec_arr;
        ierr = VecGetArrayRead(lvec,&lvec_arr);CHKERRQ(ierr);
        for (row=0; row<mbs; row++) {
          ncols = pi[row+1] - pi[row];
          for (j=0; j<ncols; j++){
            col = pj[pi[row]+j];     /* index of lvec */
            pv[pi[row]+j] += 2.0 * evec_arr[row]*lvec_arr[col]; /* P(row,col) */
          }
        }
        ierr = VecRestoreArrayRead(lvec,&lvec_arr);CHKERRQ(ierr);
      }
      ierr = VecRestoreArrayRead(evec,&evec_arr);CHKERRQ(ierr);
    }
    if (nprocEps >1 && nzB){ /* sum pB over commEps */
      ierr = PetscMalloc((nzB+1)*sizeof(PetscScalar),&buf);CHKERRQ(ierr);
      for (i=0; i<nzB; i++) buf[i] = pv[i];
      ierr = MPI_Allreduce(buf,pv,nzB, MPI_DOUBLE,MPI_SUM, epsComm);CHKERRQ(ierr);
      ierr = PetscFree(buf);CHKERRQ(ierr);
    }
  }
  if (idMat == 0) {
    nconv_loc = myidx_end-myidx_start;
    if (nconv_loc < 0) nconv_loc = 0;
    ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] interval [%g, %g], inertia %d %d; nconv_loc %d\n",rank,myinterval[0],myinterval[1],myinertia[0],myinertia[1],nconv_loc);CHKERRQ(ierr);
  }

  *P = Dmat;
  ierr = VecDestroy(&evec);CHKERRQ(ierr);
  if (idEps == 100) {
    ierr = MatView(Dmat,PETSC_VIEWER_STDOUT_(matComm));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* -----------------------------------------------------------------------------
  input:
    P, A - matrix in SBAIJ format, same data structure, same copy for each commMat[]
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
PetscErrorCode MatMatMultGetTrace(EPS eps,Mat P,Mat A,PetscInt sgn,Vec vdiag,PetscReal *tr)
{
  PetscErrorCode    ierr;
  PetscInt          j,col,ncols,rstart,rend,row,M;
  const PetscInt    *cols;
  const PetscScalar *Pvals,*Bvals;
  PetscReal         *rsum,pb,*buf;
  PetscInt          rsign,csign,dsign;
  MPI_Comm          matComm;
  EPS_KRYLOVSCHUR   *ctx=(EPS_KRYLOVSCHUR*)eps->data;
  PetscMPIInt       nprocA;
  PetscInt          npart=ctx->npart;
  Mat               myA;
  PetscBool         isSBAIJ;

  PetscFunctionBegin;
  ierr = VecGetSize(vdiag,&M);CHKERRQ(ierr);
  ierr = VecSet(vdiag,0.0);CHKERRQ(ierr);
  ierr = VecGetArray(vdiag,&rsum);CHKERRQ(ierr);
  ierr = PetscMalloc(M*sizeof(PetscReal),&buf);CHKERRQ(ierr);
  ierr = PetscMemzero(buf,M*sizeof(PetscReal));CHKERRQ(ierr);
 
  ierr = MatGetOwnershipRange(P,&rstart,&rend);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)eps,&matComm);CHKERRQ(ierr);
  if (npart == 1) {
    myA = A;
    ierr = MPI_Comm_size(matComm,&nprocA);CHKERRQ(ierr);
  } else {
    matComm = PetscSubcommChild(ctx->subc);CHKERRQ(ierr);
    ierr = MPI_Comm_size(matComm,&nprocA);CHKERRQ(ierr);
    ierr = EPSGetOperators(ctx->eps,&myA,NULL);CHKERRQ(ierr);
  }
  
  if (nprocA == 1) {
    ierr = PetscObjectTypeCompare((PetscObject)P,MATSEQSBAIJ,&isSBAIJ);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectTypeCompare((PetscObject)P,MATMPISBAIJ,&isSBAIJ);CHKERRQ(ierr);
  }

  if (isSBAIJ) {
    ierr = MatSetOption(P,MAT_GETROW_UPPERTRIANGULAR,PETSC_TRUE);CHKERRQ(ierr);
    ierr = MatSetOption(myA,MAT_GETROW_UPPERTRIANGULAR,PETSC_TRUE);CHKERRQ(ierr);
  }

  /* Signs for contributions from upper/lower triangles, and diagonal.
   * Diagonal is skipped when the summation is to be anti-symmetric.
   */
  rsign = csign = dsign = 1;
  if (sgn > 0) { rsign = -1; dsign = 0; }
  if (sgn < 0) { csign = -1; dsign = 0; }

  for (row = rstart; row < rend; row++){
    ierr = MatGetRow(P,row,&ncols,&cols,&Pvals);CHKERRQ(ierr);
    ierr = MatGetRow(myA,row,&ncols,PETSC_NULL,&Bvals);CHKERRQ(ierr);
    for (j=0; j<ncols; j++){
      col = cols[j];
      pb  = Pvals[j]*Bvals[j]; /* P(row,col)*B(row,col) */

      if (isSBAIJ) {
        if (col == row) {
	  buf[row] += dsign * pb;
        } else {
	  buf[row] += rsign * pb;
	  buf[col] += csign * pb;
        }
      } else { /* isAIJ */
        buf[row] += dsign * pb;
      }
    }
    ierr = MatRestoreRow(P,row,&ncols,&cols,&Pvals);
    ierr = MatRestoreRow(myA,row,&ncols,PETSC_NULL,&Bvals);
  }

  if (nprocA == 1){ /* sequential P and A or isAIJ */
    for (row = rstart; row < rend; row++) rsum[row] = buf[row];
  } else {          /* mpi P and A */
    ierr = MPI_Allreduce(buf,rsum,M,MPI_DOUBLE,MPI_SUM,matComm);
  }
  *tr = 0.0;
  for (row = 0; row < M; row++) *tr += rsum[row];
 
  ierr = PetscFree(buf);CHKERRQ(ierr);
  ierr = VecRestoreArray(vdiag,&rsum);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* ----------------------------------------------- */
/*
  GetSubIntsFromEvals - Get subintervals from eigenvalue clusters and gaps

   Input Parameter:

   Output Parameter:
.  subint - array determines a set of contiguous subintervals
 */

#undef __FUNCT__
#define __FUNCT__ "GetSubIntsFromEvals"
PetscErrorCode GetSubIntsFromEvals(PetscInt neclusters,PetscReal *ecluster,PetscReal *interval,DFTBGap *dftbgap,PetscInt nintervals,PetscReal *subint)
{
  PetscInt  nsubclusters[nintervals],remainder,isub,i,j,ngap=dftbgap->ngaps,k;
  PetscReal *gap=dftbgap->gap;
  
  PetscFunctionBegin;
  /* nsubclusters[i]: num of eigenvalue clusters in i-th subinterval */
  isub = neclusters/nintervals;      
  remainder = neclusters - isub*nintervals;
  for (i=0; i<remainder; i++) nsubclusters[i] = isub + 1;      
  for (i=remainder; i<nintervals; i++) nsubclusters[i] = isub;
  
  /* Set subint[i] as the mid-point of two eclusters */
  subint[0] = interval[0]; subint[nintervals] = interval[1];
  j = nsubclusters[0];
  k = 0;
  for (i=1; i<nintervals; i++) {
    subint[i] = (ecluster[j-1] + ecluster[j])/2.0;
    j += nsubclusters[i];

    if (ngap && gap[2*k+1] < subint[i] && subint[i-1] < (gap[2*k] + gap[2*k+1])/2.0) { /* a gap falls into (subint[i-1], subint[i]) */
      //printf("mv %g to %g\n",subint[i],(gap[2*k] + gap[2*k+1])/2.0);
      subint[i] = (gap[2*k] + gap[2*k+1])/2.0;
      ngap--; k++;
    }
  }
  /* if last subinterval contains a gap */
  if (ngap && gap[2*k] > subint[nintervals-1]) {
    //printf("mv %g to %g\n",subint[nintervals-1],(gap[2*k] + gap[2*k+1])/2.0);
    subint[nintervals-1] = (gap[2*k] + gap[2*k+1])/2.0;
    ngap--; k++;
  }
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------- */
/*
   dftbMonitor - User-provided routine to monitor the solution computed at each iteration. 
*/
#undef __FUNCT__
#define __FUNCT__ "DFTBMonitor"
PetscErrorCode DFTBMonitor(EPS eps,PetscInt it,void *ctx)
{
  DFTBCtx        *appctx = (DFTBCtx*)ctx;   /* user-defined application context */
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  MPI_Comm       comm; 

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)eps,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  printf("[%d] it %d ..... time %g\n",rank,it,(PetscReal)appctx->time);
#if 0
  MPI_Comm    matComm;
  PetscMPIInt idMat;
  ierr = SIPSGetSubComm(sips,&matComm,&epsComm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(matComm,&idMat);CHKERRQ(ierr);
  
  if (!idMat) {
    PetscReal time[2],time_extrem[2],time_max,time_min;
    time[0] = (PetscReal)appctx->time;
    time[1] = -time[0];
    ierr = MPI_Allreduce(&time,&time_extrem,2,MPI_DOUBLE,MPI_MAX,epsComm);CHKERRQ(ierr);
    time_max = time_extrem[0];
    time_min = - time_extrem[1];
    if (!rank) {
      ierr = PetscPrintf(PETSC_COMM_SELF,"Iter %D,  intervType %d, SIPSSolve time(max/min) = %g/%g = %g\n",it,sips->inttype,time_max,time_min,time_max/time_min);CHKERRQ(ierr);
    }
  }
#endif
  PetscFunctionReturn(0);
}
