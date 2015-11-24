"""
This module implements Hartree-Fock method.
F0[mu,nu] = T[mu,nu] + N[mu,nu]
T[mu,nu]  = < mu | -0.5 del^2 | nu > = -0.5 * int dr mu(r) del^2 nu(r)

"""
from PyQuante.Ints import coulomb

def getNuclearAttraction(mol,basisi,basisj):
    tmp = 0.
    for atom in mol:
        tmp += atom.Z * basisi.nuclear(basisj,atom.pos())
    return tmp

def get2eInts(a,b,c,d):
    return coulomb(a, b, c, d) - 0.5 * coulomb(a, c, b, d)

def getDterm(basisi,basisj,basis,D):
    """
    Density matrix dependent terms of the Fock matrix
    """
    rstart, rend = D.getOwnershipRange()
    for k in xrange(rstart,rend):
        basisk=basis[k]
        colsD,valsD = D.getRow(i)
        for l in colsT: 
            basisl=basis[l]
            tmp += D[i,j] * get2eInts(basisi, basisj, basisk, basisl)
    return MPI.COMM_WORLD.allreduce(tmp)

def getS(basis,maxdist,maxnnz=[0],bandwidth=[0],comm=PETSc.COMM_SELF):
    """
    Computes overlap matrix and nuclear repulsion energy.
    Sparsity induced by distance cutoff
    TODO:
    Better to preallocate based on diagonal and offdiagonal nonzeros.
    Cythonize
    """
    import constants as const

    nbf      = len(basis)
    maxdist2 = maxdist * maxdist
    enuke=0.0
    Vdiag = PETSc.Vec().create(comm=comm)
    A        = PETSc.Mat().create(comm=comm)
    A.setType('aij') #'sbaij'
    A.setSizes([nbf,nbf]) 
    if any(maxnnz): 
        A.setPreallocationNNZ(maxnnz) 
    else:
        A.setPreallocationNNZ(nbf)
    A.setUp()
    A.setOption(A.Option.NEW_NONZERO_ALLOCATION_ERR,False)
    rstart, rend = A.getOwnershipRange()
    localsize = rend-rstart
    Vdiag.setSizes((localsize,nbf))
    Vdiag.setUp()
    nnz = 0
    bohr2ang2 = const.bohr2ang**2.
    e2        = const.e2
    if any(bandwidth):
        if len(bandwidth)==1: bandwidth=np.array([bandwidth]*nbf)
    else:
        bandwidth=np.array([nbf]*nbf)    
    for i in xrange(rstart,rend):
        atomi   = basis[i].atom
        basisi  = basis[i]
        Zi      = atomi.get_nuke_chg()
        nbfi    = atomi.nbf 
        Vdiag[i] = 1.0
        for j in xrange(i+1,min(i+bandwidth[i],nbf)):
            basisj = basis[j]
            atomj = basisj.atom
            if atomi == atomj:
                A[i,j] = basis[i].overlap(basisj)
                nnz += 1
            else:                        
                distij2 = atomi.dist2(atomj) * bohr2ang2
                if distij2 < maxdist2:
                    Zj      = atomj.get_nuke_chg()
                    nbfj    = atomj.nbf 
                    atnoj   = atomj.atno
                    rhoj    = atomj.rho 
                    gammaij=const.e2/np.sqrt(distij2 + 0.25*(rhoi + rhoj)**2.)
                    R=np.sqrt(distij2)
                    enuke += Zi * Zj / R / ( nbfi * nbfj )
                    A[i,j] = basisi.overlap(basisj)
                    nnz += 1
    A.setDiagonal(Vdiag) 
    A.assemblyBegin()
    enuke =  MPI.COMM_WORLD.allreduce(enuke)        
    nnz =  MPI.COMM_WORLD.allreduce(nnz)  + nbf      
    A.assemblyEnd()
    B = A.duplicate(copy=True)
    B = B + A.transpose() 
    B.setDiagonal(Vdiag) 
    return  nnz,enuke, B

def getF0(mol,basis,T):
    """
    Form the zero-iteration (density matrix independent) Fock matrix, 
    also known as core Hamiltonian

    TODO: 
    Cythonize
    """
    A = T.duplicate()
    A.setUp()
    rstart, rend = A.getOwnershipRange()
    for i in xrange(rstart,rend):
        basisi=basis[i]
        for j in cols:
            basisj  = basis[j]
            tmp  = basisi.kinetic(basisj) 
            tmp += getNuclearAttraction(mol, basisi, basisj)
            A[i,j] = tmp
    A.assemble()
    return A

def getF(atomIDs, D, F0):
    """
    Density matrix dependent terms of the Fock matrix
    """
    diagD = pt.convert2SeqVec(D.getDiagonal()) 
    A     = D.duplicate()
    A.setUp()
    rstart, rend = A.getOwnershipRange()
    for i in xrange(rstart,rend):
        basisi=basis[i]
        colsD,valsD = D.getRow(i)
        colsT,valsT = T.getRow(i)
        k=0
        idxG=0
        for j in colsT: 
            basisj=basis[j]
            A[i,j] = getDterm(D,basisi,basisj)
    A.assemble()
    return A