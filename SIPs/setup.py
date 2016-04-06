#!/usr/bin/env python

#$ python setup.py build_ext --inplace

from numpy.distutils.command import build_src

# a bit of monkeypatching ...
import Cython.Compiler.Main
build_src.Pyrex = Cython
build_src.have_pyrex = True
def have_pyrex():
    import sys
    try:
        import Cython.Compiler.Main
        sys.modules['Pyrex'] = Cython
        sys.modules['Pyrex.Compiler'] = Cython.Compiler
        sys.modules['Pyrex.Compiler.Main'] = Cython.Compiler.Main
        return True
    except ImportError:
        return False
build_src.have_pyrex = have_pyrex

def configuration(parent_package='',top_path=None):
    INCLUDE_DIRS = []
    LIBRARY_DIRS = []
    LIBRARIES    = []

    # PETSc
    import os
    import socket
    PETSC_DIR  = os.environ['PETSC_DIR']
    SLEPC_DIR  = os.environ['SLEPC_DIR']
    PETSC_ARCH = os.environ.get('PETSC_ARCH', '')
    from os.path import join, isdir
    if PETSC_ARCH and isdir(join(PETSC_DIR, PETSC_ARCH)) and isdir(join(SLEPC_DIR, PETSC_ARCH)):
        INCLUDE_DIRS += [join(PETSC_DIR, PETSC_ARCH, 'include'),join(PETSC_DIR, 'include')]
        INCLUDE_DIRS +=	[join(SLEPC_DIR, PETSC_ARCH, 'include'),join(SLEPC_DIR, 'include')]
        LIBRARY_DIRS += [join(PETSC_DIR, PETSC_ARCH, 'lib')]
        LIBRARY_DIRS += [join(SLEPC_DIR, PETSC_ARCH, 'lib')]
        host = socket.gethostname()
        if 'vesta' in host or 'mira' in host:
            INCLUDE_DIRS += ['/bgsys/drivers/V1R2M2/ppc64/comm/include', 
                             '/bgsys/drivers/V1R2M2/ppc64/comm/lib/gnu',
                             '/bgsys/drivers/V1R2M2/ppc64', 
                             '/bgsys/drivers/V1R2M2/ppc64/comm/sys/include', 
                             '/bgsys/drivers/V1R2M2/ppc64/spi/include']
    else:
        if PETSC_ARCH: pass # XXX should warn ...
        INCLUDE_DIRS += [join(PETSC_DIR, 'include'),join(SLEPC_DIR, 'include')]
        LIBRARY_DIRS += [join(PETSC_DIR, 'lib'),join(SLEPC_DIR, 'lib')]
    LIBRARIES += [#'petscts', 'petscsnes', 'petscksp',
                  #'petscdm', 'petscmat',  'petscvec',
                  'petsc',]

#    LIBRARIES += ['slepcsys', 'slepcsvd', 'slepcst','slepcmpi','slepcmfn','slepcfn','allocate','slepcrg',
#		  'slepcpep','slepcnep','slepcds','slepcbv','slepceps']	

    LIBRARIES += ['slepc']
    # PETSc for Python
    import petsc4py
    import slepc4py
    INCLUDE_DIRS += [petsc4py.get_include()]
    INCLUDE_DIRS += [slepc4py.get_include()]

    # Configuration
    from numpy.distutils.misc_util import Configuration
    config = Configuration('', parent_package, top_path)
    config.add_extension('sips',
                         sources = ['sips.pyx',
                                    'sips_impl.c'],
                         depends = ['sips_impl.h'],
                         include_dirs=INCLUDE_DIRS + [os.curdir],
                         libraries=LIBRARIES,
                         library_dirs=LIBRARY_DIRS,
                         runtime_library_dirs=LIBRARY_DIRS)
    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
