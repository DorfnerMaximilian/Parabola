#Import all Packages used by this module
import os
#########################################################################
## Dictionaries for Physical constants & other information
#########################################################################
def StandardAtomicWeights():
    ##   List of the Standard Atomic Weight from 
    ##   https://www.nist.gov/pml/atomic-weights-and-isotopic-compositions-relative-atomic-masses
    ##   Access: 10.11.2022
    ##   
    ##   if a range [a,b] of Standard Atomic Weights is given we take the mean b+a/2 as the Standard Atomic Weight
    ##   if an uncertainty e.g. 28.976494700(22)is given in the Standard Atomic Weights, we take the mean.
    ##   input:   -             (void)
    ##   output: StandardAtomicWeights  (dictionary)

    StandardAtomicWeights={}
    path=os.path.dirname(os.path.realpath(__file__))
    #get the Periodic Table data
    with open(path+"/periodictable.dat","r") as f:
        lines=f.readlines()
        for line in lines:
            splitedline=line.split()
            if len(splitedline)>0:
                if splitedline[0]=='Atomic' and splitedline[1]=='Symbol':
                    atomicSymbol=splitedline[3]
                    NewIsotopeflag=True
                if splitedline[0]=='Standard' and NewIsotopeflag:
                    if len(splitedline)==4:
                        NewIsotopeflag=False
                    else:
                        if splitedline[4][0]=="[":
                            ran=splitedline[4]
                            srun=ran.split(',')
                            if len(srun)==1:
                                SaW=float(srun[0][1:-1])
                            else:
                                SaW=0.5*float(srun[0][1:])+0.5*float(srun[1][0:-1])
                        else:
                            ran=splitedline[4]
                            srun=ran.split('(')
                            SaW=float(srun[0])
                        NewIsotopeflag=False
                        StandardAtomicWeights[atomicSymbol]=SaW
    f.close()
    return StandardAtomicWeights
#-------------------------------------------------------------------------
def PhysicalConstants():
    ##   List of Physical constants in SI-units
    ##   https://physics.nist.gov/cuu/Constants
    ##   Access: 10.11.2022
    ##   
    ##   We take the numerical value for information on the Standard uncertainty and Relative Standard uncertainty 
    ##   consult https://physics.nist.gov/cuu/Constants
    PhysicalConstants={}

    #Mass Constants

    #Atomic Mass constant in [kg]
    PhysicalConstants['m_u']=1.66053906660*10**(-27)
    #Electron mass in [kg]
    PhysicalConstants['m_e']=9.109383701510**(-31)
    #Proton mass in [kg]
    PhysicalConstants['m_p']=1.67262192369**(-27)


    #Constants related to Thermodynamics

    #Avogadro constant in [1/mol]
    PhysicalConstants['N_A']=6.02214076*10**(23)
    #Boltzmann constant in [J/K]
    PhysicalConstants['k_B']=1.380649*10**(-23)


    #Quantum mechanical constants

    #Plancks constant in [Js]
    PhysicalConstants['h']=6.62607015*10**(-34)
    #reduced Plancks constant in [Js]
    PhysicalConstants['hbar']=1.054571817*10**(-34)


    #Quantum mechanical constants

    #Elementary Charge in [C]
    PhysicalConstants['e']=1.602176634*10**(-19)
    #Vacuum electric permittivity in [F/m]
    PhysicalConstants['epsilon_0']=8.854878128*10**(-12)
    #Vacuum magnetic permittivity in [N/A**2]
    PhysicalConstants['mu_0']=1.25663706212*10**(-6)

    return PhysicalConstants
#-------------------------------------------------------------------------
def ConversionFactors():
    ConversionFactor={}

    #Mass conversions

    #Conversion from atomic Mass constant to atomic mass units
    ConversionFactor['u->a.u.']=1.82288848426455E+03
    ConversionFactor['a.u.->u']=(1.82288848426455E+03)**(-1)

    #Length conversions
    #Conversion from Angstroem  to Bohr [a.u.]
    ConversionFactor['A->a.u.']=1.88972613288564
    ConversionFactor['a.u.->A']=(1.88972613288564)**(-1)
    
    #Energy conversions
    #Conversion from atomic energy units  to electron volts
    ConversionFactor['a.u.->eV']=2.72113838565563E+01
    ConversionFactor['eV->a.u.']=(2.72113838565563E+01)**(-1)
    ConversionFactor['a.u.->1/cm']=2.19474631370540E+05
    ConversionFactor['1/cm->a.u.']=(2.19474631370540E+05)**(-1)
    ConversionFactor['1/cm->J']=1.98644585714893E-23
    ConversionFactor['J->1/cm']=(1.98644585714893E-23)**(-1)

    #Force conversions
    ConversionFactor['a.u.->N']=8.2387234983E-8
    ConversionFactor['N->a.u.']=(8.2387234983E-8)**(-1)

    #Conversion factor for coupling constants
    ConversionFactor['E_H/a_0*hbar/sqrt(2*m_H)->cm^(3/2)']=1.7028697996*10**(6)

    return ConversionFactor
#-------------------------------------------------------------------------
def AtomSymbolToAtomnumber(Symbol):
    ## Function to convert to Atomic Symbols to Atomnumbers
    ## input:    atomic symbol  ("H", "He" ect.) to be converted            (string)
    ## output:   Atomnumber                                                 (int)
    atom_mapping = {
        "H": 1,
        "He": 2,
        "Li": 3,
        "Be": 4,
        "B": 5,
        "C": 6,
        "N": 7,
        "O": 8,
        "F": 9,
        "Ne": 10,
        "Na": 11,
        "Mg": 12,
        "Al": 13,
        "Si": 14,
        "P": 15,
        "S": 16,
        "Cl": 17,
        "Ar": 18,
        "K": 19,
        "Ca": 20,
        "Sc": 21,
        "Ti": 22,
        "V": 23,
        "Cr": 24,
        "Mn": 25,
        "Fe": 26,
        "Ni": 27,
        "Co": 28,
        "Cu": 29,
        "Zn": 30,
        "Ga": 31,
        "Ge": 32,
        "As": 33,
        "Se": 34,
        "Br": 35,
        "Kr": 36,
        "Rb": 37,
        "Sr": 38,
        "Y": 39,
        "Zr": 40,
        "Nb": 41,
        "Mo": 42,
        "Tc": 43,
        "Ru": 44,
        "Rh": 45,
        "Pd": 46,
        "Ag": 47,
        "Cd": 48,
    }

    if Symbol in atom_mapping:
        return atom_mapping[Symbol]
    else:
        print("Atom not implemented yet")
        return -1
#########################################################################
## END Dictionaries for Physical constants & other Data
#########################################################################
