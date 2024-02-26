# Parabola/Parabola.py
import sys
from pathlib import Path

#########################################################################
## Python Modules for the computation of exciton-phonon coupling elements
#########################################################################
#10.10.2022 Dorfner Maximilian

#########################################################################
## Setting paths
#########################################################################
pathtocp2k="/home/max/cp2k-2024.1"
modulespath="/home/max/Sync/PhD_TUM/Code/CP2K/CP2K_Python_Modules"
pathtocpp_lib = "/media/max/SSD1/PHD/Data/CP2K/CP2K_Python_Modules/parabola/CPP_Extension/bin/AtomicBasis.so" 
#########################################################################
## END Setting paths
#########################################################################

#########################################################################
## External Packages to Import
#########################################################################
import numpy as np
import scipy as sci
import copy as cp
import os
import sys


#########################################################################
## END External Packages to Import
#########################################################################

os.environ["cp2kpath"] = "/home/max/cp2k-2024.1/"
os.environ["parabolapath"] = "/media/max/SSD1/PHD/Data/CP2K/CP2K_Python_Modules/parabola/"

#########################################################################
## Dictionaries for Physical constants & other information
#########################################################################
import Modules.PhysConst as PhysConst
#########################################################################
## END Dictionaries for Physical constants & other Data
#########################################################################

#########################################################################
## Gaussian Basis Construction Routines
#########################################################################
import Modules.AtomicBasis as AtomicBasis
#########################################################################
## END Gaussian Basis Construction Routines
#########################################################################


#########################################################################
## Module to read in KS Hamiltonian, OLM, MOS, .cube file, Forces, Vibrations
#########################################################################
import Modules.Read as Read
#########################################################################
## END Module to read in KS Hamiltonian, OLM, MOS ect.
#########################################################################

#########################################################################
## Module to write .cube, .xsf, .mol
#########################################################################
import Modules.Write as Write
#########################################################################
## END Module to read in KS Hamiltonian, OLM, MOS ect.
#########################################################################

#########################################################################
## utility module unclude 
#########################################################################
import Modules.Util as util
#########################################################################
## END Module to read in KS Hamiltonian, OLM, MOS ect.
#########################################################################

#########################################################################
## module to modify the molecular/crystalline geometry
#########################################################################
import Modules.Geometry as Geometry
#########################################################################
## END module to modify the molecular/crystalline geometry
#########################################################################

#########################################################################
## module to perform & analize Vibrational Analysis
#########################################################################
import Modules.VibAna as VibAna
#########################################################################
## END module to perform & analize Vibrational Analysis
#########################################################################
#########################################################################
## module to perform & analize Convergence Checks of num. Parameters
#########################################################################
import Modules.TDDFT as TDDFT
#########################################################################
## END module to perform & analize Convergence Checks of num. Parameters
#########################################################################
#########################################################################
## module to perform & analize Convergence Checks of num. Parameters
#########################################################################
import Modules.ConvTests as ConvTests
#########################################################################
## END module to perform & analize Convergence Checks of num. Parameters
#########################################################################


#########################################################################
## Cp2k input changer functions
#########################################################################


#-------------------------------------------------------------------------




#-------------------------------------------------------------------------

#-------------------------------------------------------------------------






def compressKSfile(parentfolder="./"):
    _,_=Read.readinMatrices(parentfolder)
#-------------------------------------------------------------------------



        
#########################################################################
## END Parser and Basis Construction functions 
#########################################################################

#########################################################################
## Functions for Computing Overlap and Basis-Transformation Matrices
#########################################################################


#########################################################################
## END Functions for Computing Overlap and Basis-Transformation Matrices
#########################################################################
#########################################################################
## Functions for Computing Kinetic Energy Matrix Elements 
#########################################################################
def TInt(X,lm1,lm2,A1,A2,cs):
    # computes the T integral using the monomial decomposition of the 
    # solid harmonics.
    #Input: X (numpy.array) of the difference vector R1-R2
    #A1: positive numerical
    #A2: positive numerical
    A12red=-A1*A2/(A1+A2)
    Exponent=A12red*np.dot(X,X)
    gaussianPrefactor=np.exp(Exponent)
    Y1=A2*X/(A1+A2)
    Y2=-A1*X/(A1+A2)
    Z1=cs[lm1]
    Z2=cs[lm2]
    integral=0.0
    for P1 in Z1:
        for P2 in Z2:
            c1=P1[1]
            c2=P2[1]
            is1=P1[0]
            is2=P2[0]
            j1=is2[0]
            j2=is2[1]
            j3=is2[2]
            integral+=c1*c2*j1*(j1-1)*KFunction(Y1,Y2,is1,(j1-2,j2,j3),A1+A2)
            integral+=c1*c2*j2*(j2-1)*KFunction(Y1,Y2,is1,(j1,j2-2,j3),A1+A2)
            integral+=c1*c2*j3*(j3-1)*KFunction(Y1,Y2,is1,(j1,j2,j3-2),A1+A2)
            integral-=c1*c2*A2*(4*(j1+j2+j3)+6)*KFunction(Y1,Y2,is1,(j1,j2,j3),A1+A2)
            integral+=4*c1*c2*A2**2*(KFunction(Y1,Y2,is1,(j1+2,j2,j3),A1+A2)+KFunction(Y1,Y2,is1,(j1,j2+2,j3),A1+A2)+KFunction(Y1,Y2,is1,(j1,j2,j3+2),A1+A2))
    return gaussianPrefactor*integral
def getTMatrixElement(R1,lm1,dalpha1,R2,lm2,dalpha2,cs):
    #Compute kinetic energy matrix elemebt of two basis functions <phi_s1,n1,l1,m1(R_s1)|T|phi_s2,n2,l2,m2(R_s2)>
    #input: 
    #R1:    (numpy.array)                                   position of nucleii 1
    #R2     (numpy.array)                                   position of nucleii 2
    #lm1:    (string='s','py','pz','px','d-2'...)           angular momentum label for phi_s1,n1,l1,m1(R_s1) 
    #dalpha1: (list of list)    specifies the first Gaussian type of wave function 
    #lm2:    (string='s','py','pz','px','d-2'...)           angular momentum label for phi_s2,n2,l2,m2(R_s2) 
    #dalpha2: (list of list)    specifies the second Gaussian type of wave function 
    kinEnergyMatrixElement=0.0
    for obj1 in dalpha1:
        for obj2 in dalpha2:
            d1=obj1[1]
            alpha1=obj1[0]
            d2=obj2[1]
            alpha2=obj2[0]
            kinEnergyMatrixElement+=d1*d2*TInt(R1,alpha1,lm1,R2,alpha2,lm2,cs)
    return kinEnergyMatrixElement*(-0.5)
def getKineticEnergymatrix(Atoms,Basis,cs):
    ##Compute the overlap & transformation matrix of the Basis functions with respect to the conventional basis ordering
    ##input: Atoms               atoms of the first index
    ##                           list of sublists. 
    ##                           Each of the sublists has five elements. 
    ##                           Sublist[0] contains the atomorder as a int.
    ##                           Sublist[1] contains the symbol of the atom.
    ##                           Sublist[2:] containst the x y z coordinates.
    ##                                       unit: Angstroem
    ##
    ##
    ##       Basis               dic. of lists of sublists. The keys of the dic. are
    ##                           the atomic symbols.
    ##                           list contains sublist, where each Basisfunction of the 
    ##                           considered atom corresponds the one sublist.
    ##                           sublist[0] contains the set index as a string. 
    ##                           sublist[1] contains the shell index as a string
    ##                           sublist[2] contains the angular momentum label 
    ##                           as a string (e.g. shellindex py ect.)
    ##                           sublist[3:] are lists with two elements.
    ##                           The first corresponds the the exponent of the Gaussian
    ##                           The second one corresponds to the contraction coefficient
    ##
    ##          cs               see getcs function
    ##
    ##output:   Overlapmatrix    The Transformation matrix as a numpy array
    ConFactors=PhysConst.ConversionFactors()
    msize=0
    for atom in Atoms:
        Atom_type=atom[1]
        msize+=len(Basis[Atom_type])
    Tmatrix=np.zeros((msize,msize))
    it1=0
    it2=0
    for itAtom1 in range(len(Atoms)):
        Atom_type1=Atoms[itAtom1][1]
        B1=Basis[Atom_type1]
        for itBasis1 in range(len(Basis[Atom_type1])):
            R1=np.array(Atoms[itAtom1][2:])*ConFactors['A->a.u.'] #conversion from angstroem to atomic units
            state1=B1[itBasis1]
            dalpha1=state1[3:]
            lm1=state1[2][1:]
            for itAtom2 in range(len(Atoms)):
                Atom_type2=Atoms[itAtom2][1]
                B2=Basis[Atom_type2]
                for itBasis2 in range(len(Basis[Atom_type2])):
                    #get the position of the Atoms
                    R2=np.array(Atoms[itAtom2][2:])*ConFactors['A->a.u.'] #conversion from angstroem to atomic units
                    state2=B2[itBasis2]
                    dalpha2=state2[3:]
                    lm2=state2[2][1:]
                    TmatrixElement=getTMatrixElement(R1,lm1,dalpha1,R2,lm2,dalpha2,cs)
                    Tmatrix[it1][it2]=TmatrixElement
                    it2+=1
            it1+=1
            it2=0
    #Symmetrize the Overlapmatrix
    Tmatrix=0.5*(Tmatrix+np.transpose(Tmatrix))
    return Tmatrix
#########################################################################
##END Functions for Computing Kinetic Energy Matrix Elements 
#########################################################################










               

             









def CompressFolder(writeexcludelist=False):
    dirs=[x[0] for x in os.walk("./")]
    dirs=dirs[1:]
    excludelist=[]
    for it in progressbar(range(len(dirs)),"Compression Progress:",40):
        try:
            compressKSfile(dirs[it])
            excludelist.append(dirs[it])
        except:
            pass
    if writeexcludelist:
        with open("rsync_exclude.txt","w") as f:
            for element in excludelist:
                f.write(element[2:]+"\n")



