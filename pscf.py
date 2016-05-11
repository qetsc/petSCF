
"""
e.g.:python pscf.py -options_file options.txt
sample options_qetsc:
-N 100
-c 32 

-log_summary
#-malloc_info 

#-mat_view ::ascii_info 
#-mat_view draw -draw_pause -1
-mat_mumps_icntl_13 1 #turn off scaLAPACK for matrix inertia
-mat_mumps_icntl_24 1 #null pivot row detection for matrix inertia
-mat_mumps_icntl_23 0
-mat_mumps_icntl_28 2
-mat_mumps_icntl_29 1

-st_type sinvert 
-st_ksp_type preonly 
-st_pc_type cholesky 
-st_pc_factor_mat_solver_package mumps 

-eps_krylovschur_partitions 1  
-eps_interval 1,10
-eps_tol  1.e-8
-eps_krylovschur_detect_zeros TRUE
-eps_view_values # prints eigenvalues to STD OUT
-eps_view_values ascii:eigs.txt #writes eigenvalues to a file (overwrites if there are many iterations)
"""
import sys
import petsctools as pt
import slepctools as st
import xyztools as xt
import pyquantetools as qt
import os.path
from mindo3 import runMindo3
from hf import runHF

def printGitHash():
    """
    Not portable
    Will not work if not on the same directory
    """
    import subprocess
#    githash = subprocess.check_output(["git", "describe", "--always"]) # short hash or tag
    githash = subprocess.check_output(["git", "rev-parse", "HEAD"])  #long hasg
    print("Git hash: {0}".format(githash.strip()))
    return

def printHostName():
    from socket import gethostname
    print("Running on host {0}".format(gethostname()))
    return 

def printDateTime():
    from time import strftime
    print(strftime("%Y-%m-%d %H:%M"))
    return

def main():
    comm = pt.getComm()
    nrank = comm.size
    rank  = comm.rank
    if not rank:
        print("{0:*^72s}".format("  PSCF  "))
        printDateTime()
        printHostName()
        print("Number of MPI ranks: {0}".format(nrank))
    stage, t0   = pt.getStageTime(newstage='Read input')
    pt.sync()
    t = pt.getWallTime(t0=t0,str='Sync')  
    opts        = pt.getOptions()
    mol         = opts.getString('mol','')
    xyzfile     = opts.getString('xyz','')
    solve       = opts.getInt('solve', 0)
    sort        = opts.getInt('sort', 0)
    method      = opts.getString('method','mindo3').lower()
    writeXYZ    = opts.getBool('writeXYZ',False)
    sync        = opts.getBool('sync',False)

    if sync: 
        pt.sync()
        t = pt.getWallTime(t0=t,str='Barrier - options')
    qmol=None
    if mol:
        pt.write('xyz from mol input:{0}'.format(mol))  
        qmol=qt.getMol(mol)
    elif method == 'file':
        fA = opts.getString('fA','')
        fB = opts.getString('fB','')
        if os.path.isfile(fA):
            pt.write('Matrix A from file:{0}'.format(fA))
            stage = pt.getStage(stagename='Load Mat',oldstage=stage)
            A  = pt.getMatFromFile(fA,comm=comm)
            if os.path.isfile(fB):
                pt.write('Matrix B from file:{0}'.format(fB))
                B  = pt.getMatFromFile(fB,comm=comm)
                stage = pt.getStage(stagename='Solve',oldstage=stage)
                st.solveEPS(A,B,returnoption=solve)    
            else:
                stage = pt.getStage(stagename='Solve',oldstage=stage)
                st.solveEPS(A,returnoption=solve)    
        else:       
            pt.write('This method requires binary files for matrices')
            sys.exit()
    elif os.path.isfile(xyzfile):
        pt.write('xyz file:{0}'.format(xyzfile))
        if sort > 0:
            if rank == 0: 
                xyz         = xt.readXYZ(xyzfile)
                t = pt.getWallTime(t,str='Read xyz in')
                sortedxyz   = xt.sortXYZ(xyz)
                t = pt.getWallTime(t,str='Sorted xyz in')
            else:
                sortedxyz = None  
            sortedxyz = comm.bcast(sortedxyz,root=0)     
            pt.getWallTime(t,str='Bcast xyz in')
            if writeXYZ: 
                sortedfile  = xt.writeXYZ(sortedxyz)
                pt.write('sorted xyz file:{0}'.format(sortedfile))
            qmol        = qt.xyz2PyQuanteMol(sortedxyz)
        else:   
            qmol        = qt.xyzFile2PyQuanteMol(xyzfile)
    else:
        pt.write("{0} not found".format(xyzfile))
        pt.write("A chain of atoms will be used.")
        N       = opts.getInt('N', 4)
        Z       = opts.getInt('Z', 8)
        dist    = opts.getReal('d', 0.712) 
        qmol    = xt.getChainMol(N=N, Z=Z, d=dist)
        
    if qmol:
        pt.write("Number of atoms: %i" % (len(qmol.atoms)))
        pt.write("Number of electrons: %i" % (qmol.get_nel()))
        if sync: 
            pt.sync()
            t = pt.getWallTime(t0=t,str='Barrier - qmol')
        if method.startswith('hf'):
            t1 = pt.getWallTime()
            stage.pop()        
            runHF(qmol,opts)
            pt.getWallTime(t1,'MINDO3')
        elif method.startswith('mindo'):
            t = pt.getWallTime()
            qmol = qt.initializeMindo3(qmol)
            if sync: 
                pt.sync()
                t = pt.getWallTime(t0=t,str='Barrier - init')
            t = pt.getWallTime(t,'Initialization')            
            stage.pop()        
            runMindo3(qmol,opts)
            pt.getWallTime(t,'MINDO3')
        else:
            pt.write("No valid method specified")
    t = pt.getWallTime(t0, str="PSCF")
    if sync: 
        pt.sync()
        pt.getWallTime(t,str='Barrier - final')    
if __name__ == '__main__':
    main()

  
