import numpy as np
import petsctools as pt
from os.path import isfile, basename

def saveall(opts,k, Floc,D,eigs):
    """
    Saves Fock and Density matrices and eigenvalues to disk
    TODO:
    Put files in a dir, allow a prefix etc.
    Perform IO only on root, bcast filestr
    """
    t0 = pt.getWallTime()
    maxdstr = str(int(opts.getReal('maxdist', 1.e6)))
    xyzstr  = basename(opts.getString('xyz',''))[:-4]
    filestr = xyzstr + '_d' + maxdstr + '_i' + str(k)
    i = 2
    while (isfile('E_' + filestr + '.txt')):
        filestr = filestr + '_' + str(i)
        i += 1
    efile   = 'E_' + filestr + '.txt'
    dfile   = 'D_' + filestr + '.bin'
    ffile   = 'F_' + filestr + '.bin'      
    if not pt.rank:
        np.savetxt(efile, eigs)
    pt.writeMat(Floc, ffile)
    pt.writeMat(D, dfile)
    pt.getWallTime(t0,str='Save mat')
    return 

def extrapolate3c(A0,A1,A2):
    """
    Aitken 3-point extrapolation to accelerate SCF convergence
    A = A_{k+1} + \Delta_{k+1} / (\Delta_{k} - \Delta_{k+1})
    \Delta_{k} = A_{k} - A_{k-1} 
    """
    A = A0.duplicate()
    rstart, rend = A.getOwnershipRange()
    for i in range(rstart,rend):
        cols,vals0 = A0.getRow(i) 
        cols,vals1 = A1.getRow(i) 
        cols,vals2 = A2.getRow(i) 
        delta1     = vals1 -vals0
        delta2     = vals2 -vals1
        delta      = delta1 - delta2
#        vals2      = vals2 + divideWithoutNan(delta2, delta)
        vals2      = vals2 + delta2 / delta
        A.setValues(i,cols,vals2)
    A.assemble()    
    return A

def extrapolate3b(A0,A1,A2):
    """
    Aitken 3-point extrapolation to accelerate SCF convergence
    A = A_{k+1} + \Delta_{k+1} / (\Delta_{k} - \Delta_{k+1})
    \Delta_{k} = A_{k} - A_{k-1} 
    """
    A = A0.duplicate()
    rstart, rend = A.getOwnershipRange()
    for i in range(rstart,rend):
        cols,vals0 = A0.getRow(i) 
        cols,vals1 = A1.getRow(i) 
        cols,vals2 = A2.getRow(i) 
        delta1     = vals1 -vals0
        delta2     = vals2 -vals1
        delta      = delta2 - delta1
        d2         = delta2 / delta
        d1         = delta1 / delta
        vals2      = vals1 * d2 - vals2 * d1
        A.setValues(i,cols,vals2)
    A.assemble()    
    return A

def extrapolate3(A0,A1,A2):
    """
    Aitken 3-point extrapolation to accelerate SCF convergence
    A = A_{k+1} + \Delta_{k+1} / (\Delta_{k} - \Delta_{k+1})
    \Delta_{k} = A_{k} - A_{k-1} 
    Parameters
    ----------
    A0, A1, A2 : mat
              
    Returns
    -------
    Extrapolated matrix
    
    Note
    ----
    Ref: Pulay, P. Chem. Phys. Lett. 1980, 73 (2), 393-398.
    """
    A = A0.duplicate()
    rstart, rend = A.getOwnershipRange()
    for i in range(rstart,rend):
        cols,vals0 = A0.getRow(i) 
        cols,vals1 = A1.getRow(i) 
        cols,vals2 = A2.getRow(i) 
        delta1     = vals1 -vals0
        delta2     = vals2 -vals1
        delta      = delta2 - delta1
        vals2      = vals2 - delta2 * delta2 / delta
        A.setValues(i,cols,vals2)
    A.assemble()    
    return A

def getDIISSolution(n, errs):
    """
    Pulay's DIIS extrapolation to accelerate SCF convergence:   
    $ A_k = \sum_i^k c_i A_i$
    $ {c_k} = \arg \min {sum_i^k c_i e_i} $ with the constraint:
    $ \sum_i^k c_i = 1
    error function e can be chosen as [F,D].
    Parameters
    ----------
    n : int
        Number of vectors in the subspace
    errs : a list of mats
        matrices correspond to error functions 
      
    Returns
    -------
    A numpy array of $n$ coefficients for extrapolation.
    
    Note
    ----
    Ref: Pulay, P. Chem. Phys. Lett. 1980, 73 (2), 393-398.
    """
    a = np.zeros((n+1,n+1),'d')
    b = np.zeros(n+1,'d')
    for i in range(n):
        for j in range(i+1):
            a[i,j] = a[j,i] = pt.getTraceProductAIJ(errs[i],errs[j])
    for i in range(n):
        a[n,i] = a[i,n] = -1.0
        b[i] = 0.
    a[n,n] = 0.
    b[n] = -1.0
    x = np.linalg.solve(a, b)
    if np.allclose(np.dot(a, x), b):
        return x
    else:
        x = np.linalg.lstsq(a, b)[0]
        return x