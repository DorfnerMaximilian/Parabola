#########################################################################
## Python Modules for the computation of exciton-phonon coupling elements
#########################################################################
#10.10.2022 Dorfner Maximilian
#########################################################################
## External Packages to Import
#########################################################################
import os
#########################################################################
## Setting paths & environmental variables
#########################################################################
os.environ["cp2kpath"] = "/home/max/cp2k-2024.1/"
os.environ["parabolapath"] =__file__.split("/Parabola.py")[0]+"/"
#########################################################################
## Dictionaries for Physical constants & other information
#########################################################################
import Modules.PhysConst as PhysConst
#########################################################################
## Gaussian Basis Construction Routines
#########################################################################
import Modules.Structure as Structure
#########################################################################
## Gaussian Basis Construction Routines
#########################################################################
import Modules.AtomicBasis as AtomicBasis
#########################################################################
## Module to read in KS Hamiltonian,OLM, .cube file, Forces, Vibrations
#########################################################################
import Modules.Read as Read
#########################################################################
## Module to write .cube, .xsf, .mol, ect.
#########################################################################
import Modules.Write as Write
#########################################################################
## utility module include 
#########################################################################
import Modules.Util as Util
#########################################################################
## module to modify the molecular/crystalline geometry
#########################################################################
import Modules.Geometry as Geometry
#########################################################################
## module to perform & analize Vibrational Analysis
#########################################################################
import Modules.VibAna as VibAna
#########################################################################
## module to analize TDDFT calculations, Transition diplole moments ect.
#########################################################################
import Modules.TDDFT as TDDFT
#########################################################################
## module to perform & analize Convergence Checks of num. Parameters
#########################################################################
#import Modules.ConvTests as ConvTests
#########################################################################
##  module to compute the linear coupling constants
#########################################################################
import Modules.LCC as LCC
#########################################################################
##  module to compute the linear coupling constants
#########################################################################
import Modules.Symmetry as Symmetry











               

             








