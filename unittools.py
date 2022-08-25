#Constants
c = 2.99792458e8    # speed of light in m/s
k = 3.166830e-6     # Boltzmann constant in Hartree / Kelvin
h = 6.62607004e-34  # Planck constant in m2 kg /s
e2 = 14.399         # TODO: From PyQuante:  Coulomb's law coeff if R in Angstrom, E in eV
na = 6.02214e23     # Avagadro's number

#Conversion
bohr2ang = 0.529177249  
ang2bohr = 1./bohr2ang
hartree2kcal = 627.5095 
kcal2hartree = 1/hartree2kcal
ev2kcal = 23.061
kcal2ev = 1./ev2kcal
hartree2joule = 4.3597482e-18   
joule2hartree = 1./hartree2joule
hartree2ev = 27.211396132 
ev2hartree = 1/hartree2ev 
hartree2rcm = 2.194746e+05 
rcm2hartree = 1./hartree2rcm
amu2me = 1822.882       
me2amu = 1./amu2me        
R = k*hartree2kcal*1000.0 # gas constant R = 1.98722 cal/mole/K

def convertEnergy(en,unit,unit2=None,option=1):
    unit = unit.lower()
    if unit.startswith('kcal'):
        Ekcal = en
        Eev   = en * kcal2ev
        Ehart = en * kcal2hartree
    elif unit.startswith('ev'):
        Ekcal = en * ev2kcal
        Eev   = en
        Ehart = en * ev2hartree
    elif unit.startswith('hart'):
        Ekcal = en * hartree2kcal
        Eev   = en * hartree2ev
        Ehart = en
    else:
        print(('not imlemented unit', unit))    
    return Ekcal, Eev, Ehart    