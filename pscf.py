
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
"""
import sys
import petsctools as pt
import slepctools as st
import xyztools as xt
import os.path

def main():
    pt.write("{0:*^72s}".format("  PSCF  "))
    host        = pt.getHostName()
    # vesta gives the following error :
    #Traceback (most recent call last):
    #  File "pscf.py", line 145, in <module>
    #    main()
    #  File "pscf.py", line 37, in main
    #    pt.writeGitHash()
    #  File "/gpfs/vesta-fs0/projects/SEMO/qetsc/pscf/petsctools.py", line 22, in writeGitHash
    #    githash = subprocess.check_output(["git", "rev-parse", "HEAD"])  #long hasg
    #  File "/bgsys/tools/python2.7.5-gnu-20130730/lib64/python2.7/subprocess.py", line 568, in check_output
    #    process = Popen(stdout=PIPE, *popenargs, **kwargs)
    #  File "/bgsys/tools/python2.7.5-gnu-20130730/lib64/python2.7/subprocess.py", line 703, in __init__
    #    errread, errwrite) = self._get_handles(stdin, stdout, stderr)
    #  File "/bgsys/tools/python2.7.5-gnu-20130730/lib64/python2.7/subprocess.py", line 1110, in _get_handles
    #    c2pread, c2pwrite = self.pipe_cloexec()
    #  File "/bgsys/tools/python2.7.5-gnu-20130730/lib64/python2.7/subprocess.py", line 1153, in pipe_cloexec
    #    r, w = os.pipe()
    #OSError: [Errno 38] Function not implemented
    if 'vesta' not in host:
        pt.writeGitHash()
        
    stage, t0   = pt.getStageTime(newstage='Read input')  
    opts        = pt.getOptions()
    mol         = opts.getString('mol','')
    xyzfile     = opts.getString('xyz','')
    maxdist     = opts.getReal('maxdist', 1.e6)
    maxiter     = opts.getInt('maxiter', 30)
    analysis    = opts.getInt('analysis', 0)
    solve       = opts.getInt('solve', 0)
    maxnnz      = opts.getInt('maxnnz', 0)
    guess       = opts.getInt('guess', 0)
    bandwidth   = opts.getInt('bw', 0)
    sort        = opts.getInt('sort', 0)
    nsubint     = opts.getInt('eps_krylovschur_partitions', 1)
    method      = opts.getString('method','mindo3').lower()
    pyquante    = opts.getBool('pyquante',False)
    writeXYZ    = opts.getBool('writeXYZ',False)
    scfthresh   = opts.getReal('scfthresh',1.e-5)
    checkenergy = opts.getReal('checkenergy',0) #
    comm = pt.getComm()
    pt.write("Number of MPI ranks: {0}".format(comm.size))
    pt.write("Number of subintervals: {0}".format(nsubint))
    qmol=None
    if mol:
        import PyQuante.Molecule 
        pt.write('xyz from mol input:{0}'.format(mol))  
        qmol=PyQuante.Molecule(mol)
    elif method == 'file':
        from os.path import isfile
        fA = opts.getString('fA','')
        fB = opts.getString('fB','')
        if isfile(fA):
            pt.write('Matrix A from file:{0}'.format(fA))
            stage = pt.getStage(stagename='Load Mat',oldstage=stage)
            A  = pt.getMatFromFile(fA,comm=comm)
            if isfile(fB):
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
            t0          = pt.getWallTime() 
            xyz         = xt.readXYZ(xyzfile)
            sortedxyz   = xt.sortXYZ(xyz)
            pt.getWallTime(t0,str='Sorted xyz in')
            if writeXYZ: 
                sortedfile  = xt.writeXYZ(sortedxyz)
                pt.write('sorted xyz file:{0}'.format(sortedfile))
            qmol        = xt.xyz2PyQuanteMol(sortedxyz)
        else:   
            qmol        = xt.xyzFile2PyQuanteMol(xyzfile)
    else:
        pt.write("{0} not found".format(xyzfile))
        pt.write("A chain of atoms will be used.")
        N       = opts.getInt('N', 4)
        c       = opts.getInt('c', 3)
        Z       = opts.getInt('Z', 8)
        dist    = opts.getReal('d', 0.712) 
        qmol    = xt.getChainMol(N=N, Z=Z, d=dist)
        
    if qmol:
        pt.write("Number of atoms: %i" % (len(qmol.atoms)))
        pt.write("Number of electrons: %i" % (qmol.get_nel()))
        if method == "sparsity":
            import PyQuante.MINDO3
            stage = pt.getStage(stagename='Initialize',oldstage=stage)
            qmol  = PyQuante.MINDO3.initialize(qmol)
            nbf   = getNBF(qmol)    
            basis = getBasis(qmol, nbf)
            stage = pt.getStage(stagename='distCSR',oldstage=stage)
            BCSR  = getDistCSR(basis, nbf,maxdist=maxdist)
            stage = pt.getStage(stagename='getSparsityInfo',oldstage=stage)
            pt.getSparsityInfo(BCSR, nbf, maxdensity)
            stage.pop()
    
        elif method.startswith('hf'):
            from hf import getEnergy
            finalenergy = getEnergy(qmol,opts)
            if checkenergy:
                energydiff = abs(finalenergy-checkenergy)
                if energydiff < scfthresh:
                    pt.write("Passed final energy test with difference (kcal/mol): {0:20.10f}".format(energydiff))
                else:
                    pt.write("Failed final energy test with difference (kcal/mol): {0:20.10f}".format(energydiff))
        elif method.startswith('mindo3'):
            t1 = pt.getWallTime()
            from mindo3 import getEnergy
            stage.pop()
            finalenergy = getEnergy(qmol,opts)
            if checkenergy:
                energydiff = abs(finalenergy-checkenergy)
                if energydiff < scfthresh:
                    pt.write("Passed final energy test with difference (kcal/mol): {0:20.10f}".format(energydiff))
                else:
                    pt.write("Failed final energy test with difference (kcal/mol): {0:20.10f}".format(energydiff))
            pt.getWallTime(t1,'MINDO3')
        else:
            pt.write("No valid method specified")
    pt.getWallTime(t0, str="PSCF")
if __name__ == '__main__':
    main()

  
