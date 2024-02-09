#########################################################################
## Python Modules for the computation of exciton-phonon coupling elements
#########################################################################
#10.10.2022 Dorfner Maximilian

#########################################################################
## Setting paths
#########################################################################
pathtocp2k="/home/max/cp2k-2022.2"
pathtobinaries=pathtocp2k+"/exe/local/"
modulespath="/home/max/Sync/PhD_TUM/Code/CP2K/CP2K_Python_Modules"
pathtocpp_lib = "/media/max/SSD1/PHD/Data/CP2K/CP2K_Python_Modules/parabola/CPP_Extension/bin/get_T_Matrix.so" 
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
import matplotlib.pyplot as plt
from ctypes import c_char_p, cdll, POINTER, c_double, c_int
from copy import deepcopy
#For standard Latex fonts
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
#########################################################################
## END External Packages to Import
#########################################################################

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


#########################################################################
## Cp2k input changer functions
#########################################################################
def getCellSize(path="./"):
    ## Function read out the cell size from the .inp file
    ## input: (opt.)   path   path to the folder of the calculation         (string)
    ## output:  -                                                           (void)

    #Get the .inp file
    inp_files = [f for f in os.listdir(path) if f.endswith('.inp')]
    if len(inp_files) != 1:
        raise ValueError('InputError: There should be only one inp file in the current directory')
    inp_file= path+"/"+inp_files[0]
    cellvectors=[np.zeros((3,1)),np.zeros((3,1)),np.zeros((3,1))]
    with open(inp_file) as f:
        lines = f.readlines()
        Cellflag=False
        for l in lines:
            if len(l.split())>=1:
                if l.split()[0]=='&CELL':
                    Cellflag=True
                if l.split()[0]=='END' and l.split()[1]=='CELL':
                    Cellflag=False
                if Cellflag:
                    if l.split()[0]=='ABC':
                        cellvectors[0]=np.array([float(l.split()[1]),0.0,0.0])
                        cellvectors[1]=np.array([0.0,float(l.split()[2]),0.0])
                        cellvectors[2]=np.array([0.0,0.0,float(l.split()[3])])
                    if l.split()[0]=='A':
                        cellvectors[0]=np.array([float(l.split()[1]),float(l.split()[2]),float(l.split()[3])])
                    if l.split()[0]=='B':
                        cellvectors[1]=np.array([float(l.split()[1]),float(l.split()[2]),float(l.split()[3])])
                    if l.split()[0]=='C':
                        cellvectors[2]=np.array([float(l.split()[1]),float(l.split()[2]),float(l.split()[3])])
    #check cell volume 
    det=np.linalg.det(np.array(cellvectors))
    if np.abs(det)<10**(-3):
        ValueError("Cell Vectors do not span a unit cell!")
    return cellvectors
def centerMolecule(path="./"):
    ## Function to center the molecule/collection of atoms in the unit cell 
    ## input: (opt.)   path   path to the folder of the calculation         (string)
    ## output:  -                                                           (void)

    cellcoordinates=getCellSize(path)
    print(cellcoordinates)
    #Check if the cell is orthorhombic, otherwise throw error:
    for cellvector1 in cellcoordinates:
        for cellvector2 in cellcoordinates:
            absscalarproduct =np.abs(np.dot(cellvector1,cellvector2))
            abscrossproduct = np.linalg.norm((np.cross(cellvector1,cellvector2)))
            if absscalarproduct >10**(-10) and abscrossproduct>10**(-10):
                ValueError("Centering of Molecule in Unit Cell makes only sense for Non-Periodic Calculations! Use Orthorhombic Unit cells for this case!")

    #Compute center of Cell (assuming orthogonal basis vectors)
    cellcenter=0.5*cellcoordinates[0]+0.5*cellcoordinates[1]+0.5*cellcoordinates[2]
    xyzcoordinates,_,atomicsym=getCoordinatesAndMasses(path)
    geometric_mean=np.array([0.0,0.0,0.0])
    for coordinate in xyzcoordinates:
        geometric_mean+=coordinate
    geometric_mean/=len(xyzcoordinates)
    centerofcellCoordinates=[]
    for coordinate in xyzcoordinates:
        centerofcellCoordinates.append(coordinate+cellcenter-geometric_mean)
    #Compute maximum distance of atom from coordinatecenter
    maxdistx=0
    maxdisty=0
    maxdistz=0
    for coordinates in xyzcoordinates:
        distx=np.abs(coordinates[0])
        disty=np.abs(coordinates[1])
        distz=np.abs(coordinates[2])
        if distx > maxdistx:
            maxdistx=distx
        if disty > maxdisty:
            maxdisty=disty
        if distz > maxdistz:
            maxdistz=distz
    #Add cutoff to maximum distance
    cutoff=2
    mincellsizex=cutoff+maxdistx
    mincellsizey=cutoff+maxdisty
    mincellsizez=cutoff+maxdistz
    #Check if molecule fits into box otherwise shift coordinates to center of cell
    
    if mincellsizex<=0.5*distx:
        ValueError("Increase cell size in z direction to at least ",2*mincellsizez)
    if mincellsizey<=0.5*disty:
        ValueError("Increase cell size in y direction to at least ",2*mincellsizey)
    if mincellsizez<=0.5*distz:
        ValueError("Increase cell size in z direction to at least ",2*mincellsizez)
    
    writexyzfile(atomicsym,centerofcellCoordinates,path)
#-------------------------------------------------------------------------
def getPrincipleAxisCoordinates(path="./"):
    ## Function to center the molecule in the unit cell & align the principle Axis 
    ## and orient priciple axis along x y z
    ## input: (opt.)   path   path to the folder of the calculation         (string)
    ## output:  -                                                           (void)
    cellcoordinates=getCellSize(path)
    #Check if the cell is orthorhombic, otherwise throw error:
    for cellvector1 in cellcoordinates:
        for cellvector2 in cellcoordinates:
            absscalarproduct =np.abs(np.dot(cellvector1,cellvector2))
            abscrossproduct = np.linalg.norm((np.cross(cellvector1,cellvector2)))
            if absscalarproduct >10**(-10) and abscrossproduct>10**(-10):
                ValueError("Centering of Molecule in Unit Cell makes only sense for Non-Periodic Calculations! Use Orthorhombic Unit cells for this case!")

    #Compute center of Cell (assuming orthogonal basis vectors)
    xyzcoordinates,masses,atomicsym=getCoordinatesAndMasses(path)


    #Compute the Intertia Tensor
    I=getInertiaTensor(xyzcoordinates,masses)
    centerofmasscoordinates,_=ComputeCenterOfMassCoordinates(xyzcoordinates,masses)
    #Get the principle Axis
    _,principleAxis=np.linalg.eigh(I)
    #Check that the principle axis are orthorgonal (Could be not the case in case of degeneracy)
    for it1 in range(3):
        for it2 in range(3):
            if it1!=it2 and np.abs(np.dot(principleAxis[:,it1],principleAxis[:,it2]))>10**(-12):
                ValueError("Principle Axis of Molecules are not orthorgonal! Implement the Degenerate Case!")
            if it1==it2 and np.abs(np.dot(principleAxis[:,it1],principleAxis[:,it2])-1.0)>10**(-12):
                ValueError("Principle Axis are not normalized! This leads to Errorious results!")
    #Represent the centerofMassCoordinates in terms of the principle axis frame
    #Fix Orientation of Principle Axis 
    v1=principleAxis[:,0]
    v2=principleAxis[:,1]
    v3=principleAxis[:,2]
    if v1[0]<0.0:
        v1*=-1.0
    if v2[0]<0.0:
        v2*=-1.0
    #This makes Principle Axis righthanded
    B=np.zeros((3,3))
    B[:,0]=v1
    B[:,1]=v2
    B[:,2]=v3
    determinant=np.linalg.det(B)
    if determinant<0:
        v3*=-1.0
    principleaxiscoordinates=[]
    for coordinate in centerofmasscoordinates:
        v1coordinate=np.dot(coordinate,v1)
        v2coordinate=np.dot(coordinate,v2)
        v3coordinate=np.dot(coordinate,v3)
        principleaxiscoordinates.append(np.array([v1coordinate,v2coordinate,v3coordinate]))
    writexyzfile(atomicsym,principleaxiscoordinates,path)
def getCenterOFMassCoordinates(path="./"):
    ## Function to center the origin in the center of mass
    ## input: (opt.)   path   path to the folder of the calculation         (string)
    ## output:  -                                                           (void)

    #Compute center of Cell (assuming orthogonal basis vectors)
    xyzcoordinates,masses,atomicsym=getCoordinatesAndMasses(path)
    comc,_=ComputeCenterOfMassCoordinates(xyzcoordinates,masses)
    writexyzfile(atomicsym,comc,path)
def writexyzfile(atomicsym,coordinates,readpath="./",writepath="./",filename="dummyname",append=False):
    xyzfilename=getxyzfilename(readpath)
    xyzfilename=xyzfilename.split("/")[-1]
    # generate new xyz file
    xyzinput=[]
    with open(readpath+"/"+xyzfilename) as g:
        lines = g.readlines()
        xyzinput.append(lines[0])
        xyzinput.append(lines[1])
    g.close()
    for it,xyz in enumerate(coordinates):
        xyzinput.append(atomicsym[it]+' '+str(xyz[0])+' '+str(xyz[1])+' '+str(xyz[2])+'\n')
    #open the old xyz file
    
    #Check if the file exist in the write directory
    inp_files = [f for f in os.listdir(writepath) if f.endswith('.xyz')]
    if len(inp_files) == 0:
        os.system("touch "+xyzfilename)
    if filename=="dummyname":
        filename=xyzfilename
    g=open(writepath+"/"+filename,'a')
    if not append:
        # kill its content
        g.truncate(0)
    #generate the new content
    for line in xyzinput:
        g.write(line)
    g.close()
def getNewXYZ(path='.'):
    ## Script to readout the last iteration of the GeoOpt file and 
    ## generate new oldxyzfile_opt.xyz file for Vibrational Analysis
    ## input: (opt.)   path   path to the folder of the geoOpt calculation         (string)
    ## output:  -                                                                  (void)


    #get the Projectname
    inp_files = [f for f in os.listdir(path) if f.endswith('.inp')]
    if len(inp_files) != 1:
        raise ValueError('InputError: There should be only one inp file in the current directory')
    filename = path+"/"+inp_files[0]
    f = open(filename, "r")
    projectname='NoName'
    for line in f.readlines():
        if len(line.split())>0:
            if line.split()[0]=="PROJECT":
                projectname=line.split()[1]
    f.close()
    xyz_files = [f for f in os.listdir(path) if f.endswith('.xyz')]
    #remove the projectname+'-pos-1.xyz' file from xyz files
    xyz_files.remove(projectname+'-pos-1.xyz')
    if len(xyz_files) != 1:
        raise ValueError('InputError: There should be only one old xyz file in the current directory')
    xyzfilename=path+"/"+xyz_files[0]
    iter2=0
    if projectname!='NoName':
        numiter=0
        buffer=[]
        getflag=False
        f=open(path+'/'+projectname+'-pos-1.xyz')
        for line in f.readlines():
            if len(line.split())>0:
                if line.split()[0]=='i':
                    numiter+=1
        f.close()
        f=open(path+'/'+projectname+'-pos-1.xyz')
        for line in f.readlines():
            if getflag:
                buffer.append(line.split())
            if len(line.split())>0:
                if line.split()[0]=='i':
                    iter2+=1
                if iter2==numiter:
                    getflag=True
            else:
                continue
        f.close()
        f=open(xyzfilename,'r')
        header=f.readlines()[0:2]
        f.close()
        g=open(xyzfilename[:-4]+'_opt.xyz','w')
        g.write(header[0])
        g.write(header[1])
        for line in buffer:
            for element in line:
                g.write(element)
                g.write(' ')
            g.write('\n')
        g.close()
#-------------------------------------------------------------------------
def changeConfiguration(folderlabel,vector,delta,sign,path_xyz='./',path_to="./"):
    ## Reads in the equilibrium configuration xyz file and changes 
    ## position of the atoms in the xyz file in direction of the unit vector vector
    ## either:
    ## by delta [units of corresponding xyzfile or ], if the rescaleflag is False
    ##
    ## input:   folderlabel name of the folder to create                     (string)
    ##          vector      normalized vector of the size of 3 x atoms       (np.array)
    ##          delta       change of the atomic configuration               (float)
    ##          sign                                                         (0,1)
    ## (opt.)   rescaleflag flag to control the rescaling of delta           (bool)
    ## (opt.)   path   path to the folder of the VibAna calculation          (string)
    ## output:  -               (void)

    #Get the .xyz file
    xyz_files = [f for f in os.listdir(path_xyz) if f.endswith('.xyz')]
    if len(xyz_files) != 1:
        raise ValueError('InputError: There should be only one xyz file in the current directory')
    xyzfilename = xyz_files[0]
    xyzcoordinates=[]
    atomtypes=[]
    with open(path_xyz+"/"+xyzfilename) as g:
        lines = g.readlines()
        for l in lines[2:]:
            if len(l.split())>=4:
                xyzcoordinates.append(float(l.split()[1]))
                xyzcoordinates.append(float(l.split()[2]))
                xyzcoordinates.append(float(l.split()[3]))
                atomtypes.append(l.split()[0])
    xyzcoordinates=np.array(xyzcoordinates)
    xyzcoordinates=xyzcoordinates+(-1)**(sign)*delta*vector
    if sign==0:
        symbolsign='+'
    if sign==1:
        symbolsign='-'
    foldername=path_to+folderlabel+"sign="+symbolsign
    os.mkdir(foldername)
    f=open(foldername+"/"+xyzfilename,"w")
    f.write(lines[0])
    f.write(lines[1])
    for iter in range(int(len(xyzcoordinates)/3)):
        atom=atomtypes[iter]
        xcoord=xyzcoordinates[3*iter]
        ycoord=xyzcoordinates[3*iter+1]
        zcoord=xyzcoordinates[3*iter+2]
        f.write(atom+' '+str(xcoord)+' '+str(ycoord)+' '+str(zcoord))
        f.write("\n")
    f.close()
    return
#-------------------------------------------------------------------------
def changeR_Cutoff(origin,source,RCutoff):
    inp_files = [f for f in os.listdir(origin) if f.endswith('.inp')]
    if len(inp_files) != 1:
        raise ValueError('InputError: There should be only one inp file in the current directory')
    filename = inp_files[0]
    with open(filename, "r") as f:
        with open(source,"w") as g:
            for line in f.readlines():
                if len(line.split())>=1:
                    if line.split()[0]=="CUTOFF_RADIUS":
                        line= "\tCUTOFF_RADIUS "+str(RCutoff)+"\n"
                g.write(line)
#-------------------------------------------------------------------------
def R_CutoffTest_inputs(RCutoffs,parentpath="./",binaryloc=pathtobinaries,binary="cp2k.popt"):
    #get the Projectname: 
    inp_files = [f for f in os.listdir(parentpath) if f.endswith('.inp')]
    if len(inp_files) != 1:
        raise ValueError('InputError: There should be only one .inp file in the current directory')
    inpfilename = inp_files[0]
    Projectname='emptyString'
    with open(inpfilename,'r') as f:
        for lines in f:
            if len(lines.split())>0:
                if lines.split()[0]=="PROJECT":
                    Projectname=lines.split()[1]
    if Projectname=='emptyString':
        raise ValueError('InputError: Projectname not found!')
    #get xyzfile
    xyz_files = [f for f in os.listdir(parentpath) if f.endswith('.xyz')]
    if len(xyz_files) != 1:
        raise ValueError('InputError: There should be exactly one .xyz file in the current directory')
    xyzfilename = xyz_files[0]
    Restart_files = [f for f in os.listdir(parentpath) if f.endswith('-RESTART.wfn')]
    if len(Restart_files) != 1:
        raise ValueError('InputError: There should be exactly one Restart file in the current directory')
    Restart_filename = Restart_files[0]
    if Restart_filename!=Projectname+'-RESTART.wfn':
        raise ValueError('InputError: Project- and Restartfilename differ! Reconsider your input.')
    for R_Cutoff in RCutoffs:
        work_dir="Cutoff_"+str(R_Cutoff)+"A"
        if not os.path.isdir(parentpath+work_dir):
            os.mkdir(parentpath+work_dir)
        else:
            filelist = [ f for f in os.listdir(parentpath+work_dir)]
            for f in filelist:
                os.remove(parentpath+work_dir+"/"+f)
        changeR_Cutoff(parentpath,parentpath+work_dir+"/input_file.inp",R_Cutoff)
        os.system("cp "+parentpath+xyzfilename+" "+parentpath+work_dir)
        os.system("cp "+parentpath+Restart_filename+" "+parentpath+work_dir)
        centerMolecule(parentpath+work_dir)
        os.system("ln -s "+binaryloc+"/"+binary+" "+parentpath+work_dir+"/")
#-------------------------------------------------------------------------
def changeRelCutoff(origin,source,RelCutoff):
    inp_files = [f for f in os.listdir(origin) if f.endswith('.inp')]
    if len(inp_files) != 1:
        raise ValueError('InputError: There should be only one inp file in the current directory')
    filename = inp_files[0]
    with open(filename, "r") as f:
        with open(source,"w") as g:
            for line in f.readlines():
                if len(line.split())>=1:
                    if line.split()[0]=="REL_CUTOFF":
                        line= "\tREL_CUTOFF "+str(RelCutoff)+"\n"
                g.write(line)
#-------------------------------------------------------------------------
def RelCutoffTest_inputs(RelCutoffs,parentpath="./",binaryloc=pathtobinaries,binary="cp2k.popt"):
    #get the Projectname: 
    inp_files = [f for f in os.listdir(parentpath) if f.endswith('.inp')]
    if len(inp_files) != 1:
        raise ValueError('InputError: There should be only one .inp file in the current directory')
    inpfilename = inp_files[0]
    Projectname='emptyString'
    with open(inpfilename,'r') as f:
        for lines in f:
            if len(lines.split())>0:
                if lines.split()[0]=="PROJECT":
                    Projectname=lines.split()[1]
    if Projectname=='emptyString':
        raise ValueError('InputError: Projectname not found!')
    #get xyzfile
    xyz_files = [f for f in os.listdir(parentpath) if f.endswith('.xyz')]
    if len(xyz_files) != 1:
        raise ValueError('InputError: There should be exactly one .xyz file in the current directory')
    xyzfilename = xyz_files[0]
    Restart_files = [f for f in os.listdir(parentpath) if f.endswith('-RESTART.wfn')]
    if len(Restart_files) != 1:
        raise ValueError('InputError: There should be exactly one Restart file in the current directory')
    Restart_filename = Restart_files[0]
    if Restart_filename!=Projectname+'-RESTART.wfn':
        raise ValueError('InputError: Project- and Restartfilename differ! Reconsider your input.')
    for RelCutoff in RelCutoffs:
        work_dir="Rel_Cutoff_"+str(RelCutoff)+"Ry"
        if not os.path.isdir(parentpath+work_dir):
            os.mkdir(parentpath+work_dir)
        else:
            filelist = [ f for f in os.listdir(parentpath+work_dir)]
            for f in filelist:
                os.remove(parentpath+work_dir+"/"+f)
        changeRelCutoff(parentpath,parentpath+work_dir+"/input_file.inp",RelCutoff)
        os.system("cp "+parentpath+xyzfilename+" "+parentpath+work_dir)
        os.system("cp "+parentpath+Restart_filename+" "+parentpath+work_dir)
        centerMolecule(parentpath+work_dir)
        os.system("ln -s "+binaryloc+"/"+binary+" "+parentpath+work_dir+"/")
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def changeCutoff(origin,source,Cutoff):
    inp_files = [f for f in os.listdir(origin) if f.endswith('.inp')]
    if len(inp_files) != 1:
        raise ValueError('InputError: There should be only one inp file in the current directory')
    filename = inp_files[0]
    with open(filename, "r") as f:
        with open(source,"w") as g:
            for line in f.readlines():
                if len(line.split())>=1:
                    if line.split()[0]=="CUTOFF":
                        line= "\tCUTOFF "+str(Cutoff)+"\n"
                g.write(line)
#-------------------------------------------------------------------------
def CutoffTest_inputs(Cutoffs,parentpath="./",binaryloc=pathtobinaries,binary="cp2k.popt"):
    #get the Projectname: 
    inp_files = [f for f in os.listdir(parentpath) if f.endswith('.inp')]
    if len(inp_files) != 1:
        raise ValueError('InputError: There should be only one .inp file in the current directory')
    inpfilename = inp_files[0]
    Projectname='emptyString'
    with open(inpfilename,'r') as f:
        for lines in f:
            if len(lines.split())>0:
                if lines.split()[0]=="PROJECT":
                    Projectname=lines.split()[1]
    if Projectname=='emptyString':
        raise ValueError('InputError: Projectname not found!')
    #get xyzfile
    xyz_files = [f for f in os.listdir(parentpath) if f.endswith('.xyz')]
    if len(xyz_files) != 1:
        raise ValueError('InputError: There should be exactly one .xyz file in the current directory')
    xyzfilename = xyz_files[0]
    Restart_files = [f for f in os.listdir(parentpath) if f.endswith('-RESTART.wfn')]
    if len(Restart_files) != 1:
        raise ValueError('InputError: There should be exactly one Restart file in the current directory')
    Restart_filename = Restart_files[0]
    if Restart_filename!=Projectname+'-RESTART.wfn':
        raise ValueError('InputError: Project- and Restartfilename differ! Reconsider your input.')
    for Cutoff in Cutoffs:
        work_dir="Cutoff_"+str(Cutoff)+"Ry"
        if not os.path.isdir(parentpath+work_dir):
            os.mkdir(parentpath+work_dir)
        else:
            filelist = [ f for f in os.listdir(parentpath+work_dir)]
            for f in filelist:
                os.remove(parentpath+work_dir+"/"+f)
        changeCutoff(parentpath,parentpath+work_dir+"/input_file.inp",Cutoff)
        os.system("cp "+parentpath+xyzfilename+" "+parentpath+work_dir)
        os.system("cp "+parentpath+Restart_filename+" "+parentpath+work_dir)
        centerMolecule(parentpath+work_dir)
        os.system("ln -s "+binaryloc+"/"+binary+" "+parentpath+work_dir+"/")
#-------------------------------------------------------------------------
def changeCellSize(origin,source,CellX,CellY,CellZ):
    inp_files = [f for f in os.listdir(origin) if f.endswith('.inp')]
    if len(inp_files) != 1:
        raise ValueError('InputError: There should be only one inp file in the current directory')
    filename = inp_files[0]
    with open(filename, "r") as f:
        with open(source,"w") as g:
            for line in f.readlines():
                if len(line.split())>=1:
                    if line.split()[0]=="ABC":
                        line= "\tABC "+str(CellX)+" "+str(CellY)+" "+str(CellZ)+"\n"
                g.write(line)
#-------------------------------------------------------------------------
def CellSizeTest_inputs(Cell_Dims,parentpath="./",binaryloc=pathtobinaries,binary="cp2k.popt"):
    #get xyzfile
    xyz_files = [f for f in os.listdir(parentpath) if f.endswith('.xyz')]
    if len(xyz_files) != 1:
        raise ValueError('InputError: There should be only one inp file in the current directory')
    xyzfilename = xyz_files[0]
    
    for celldim in Cell_Dims:
        work_dir="CellDim_"+str(celldim[0])+"x"+str(celldim[1])+"x"+str(celldim[2])+"A"
        if not os.path.isdir(parentpath+work_dir):
            os.mkdir(parentpath+work_dir)
        else:
            filelist = [ f for f in os.listdir(parentpath+work_dir)]
            for f in filelist:
                os.remove(parentpath+work_dir+"/"+f)
        Restart_files = [f for f in os.listdir(parentpath+"/") if f.endswith('-RESTART.wfn')]
        if len(Restart_files) != 1:
            raise ValueError('InputError: There should be exactly one Restart file in the current directory')
        Restart_filename = Restart_files[0]
        changeCellSize(parentpath,parentpath+work_dir+"/input_file.inp",celldim[0],celldim[1],celldim[2])
        os.system("cp "+parentpath+xyzfilename+" "+parentpath+work_dir)
        os.system("cp "+parentpath+Restart_filename+" "+parentpath+work_dir)
        centerMolecule(parentpath+work_dir)
        os.system("ln -s "+binaryloc+"/"+binary+" "+parentpath+work_dir+"/")
#-------------------------------------------------------------------------
def progressbar(it, prefix="", size=60, out=sys.stdout): # Python3.3+
    count = len(it)
    def show(j):
        x = int(size*j/count)
        print("{}[{}{}] {}/{}".format(prefix, u"█"*x, "."*(size-x), j, count), 
                end='\r', file=out, flush=True)
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    print("\n", flush=True, file=out)
def Vib_Ana_inputs(delta,vectors=[],linktobinary=True,binary="cp2k.popt",parentpath="./",binaryloc=pathtobinaries):
    ConFactors=ConversionFactors()
    #get the Projectname: 
    inp_files = [f for f in os.listdir(parentpath) if f.endswith('.inp')]
    if len(inp_files) != 1:
        raise ValueError('InputError: There should be only one .inp file in the current directory')
    inpfilename = inp_files[0]
    CheckinpfileforVib_Ana(parentpath)
    Projectname='emptyString'
    with open(inpfilename,'r') as f:
        for lines in f:
            if len(lines.split())>0:
                if lines.split()[0]=="PROJECT":
                    Projectname=lines.split()[1]
    if Projectname=='emptyString':
        raise ValueError('InputError: Projectname not found!')
    #get xyzfile
    xyz_files = [f for f in os.listdir(parentpath) if f.endswith('.xyz')]
    if len(xyz_files) != 1:
        raise ValueError('InputError: There should be only one .xyz file in the current directory')
    xyzfilename = xyz_files[0]
    Restart_files = [f for f in os.listdir(parentpath) if f.endswith('-RESTART.wfn')]
    if len(Restart_files) != 1:
        raise ValueError('InputError: There should be only one Restart file in the current directory')
    Restart_filename = Restart_files[0]
    if Restart_filename!=Projectname+'-RESTART.wfn':
        raise ValueError('InputError: Project- and Restartfilename differ! Reconsider your input.')
    atomorder=[]
    with open(parentpath+xyzfilename) as f:
        lines=f.readlines()
        numberofatoms=int(lines[0])
        for line in lines[2:]:
            if len(line.split())>0:
                atomorder.append(line.split()[0])
    if  not bool(vectors):
        for it in range(3*numberofatoms):
            vector=np.zeros(3*numberofatoms)
            vector[it]=1.0
            vectors.append(vector)
    else: 
        if vectors!=3*numberofatoms:
            print('InputError: Not enough vectors given for the Normal Mode Analysis!')
        if np.linalg.matrix_rank(vectors)!=len(vectors):
            #do a second check based on SVD 
            print('InputError: The set of vectors given do not form a basis!')

    with open(parentpath+"BasisHessian","w+") as g:
        g.write("delta="+str(delta)+"\n")
        g.write("unit=Bohr"+"\n")
        g.write("Basis Vectors in which the Hessian is represented expressed in the Standard Basis:\n")
        g.write("Format:     Atom:       x:             y:           z:\n")
        num=1
        for vector in vectors:
            g.write("Basisvector    "+str(num)+"\n")
            for it in range(numberofatoms):
                g.write(atomorder[it]+" "+str(vector[3*it])+" "+str(vector[3*it+1])+" "+str(vector[3*it+2])+"\n")
            num+=1

        
    os.mkdir(parentpath+"Equilibrium_Geometry")
    os.system("cp "+parentpath+inpfilename+" "+parentpath+"Equilibrium_Geometry")
    os.system("cp "+parentpath+xyzfilename+" "+parentpath+"Equilibrium_Geometry")
    os.system("cp "+parentpath+Restart_filename+" "+parentpath+"Equilibrium_Geometry")
    if linktobinary:
        os.system("ln -s "+binaryloc+"/"+binary+" "+parentpath+"Equilibrium_Geometry"+"/")
    for it in range(3*numberofatoms):
        folderlabel='vector='+str(it+1)
        for sign in [0,1]:
            if sign==0:
                symbolsign='+'
            if sign==1:
                symbolsign='-'
            work_dir=folderlabel+"sign="+symbolsign
            vec=vectors[it]
            changeConfiguration(folderlabel,vec,delta*ConFactors['a.u.->A'],sign,parentpath)
            os.system("cp "+parentpath+inpfilename+" "+parentpath+work_dir)
            os.system("cp "+parentpath+Restart_filename+" "+parentpath+work_dir)
            if linktobinary:
                os.system("ln -s "+binaryloc+"/"+binary+" "+parentpath+work_dir+"/")
def CheckinpfileforVib_Ana(parentpath):
    #Checks the integrity of the inp file for a VibrationalAnalysis
    inp_files = [f for f in os.listdir(parentpath) if f.endswith('.inp')]
    if len(inp_files) != 1:
        raise ValueError('InputError: There should be only one .inp file in the '+parentpath+' directory')
    #Check if the inp files contain the necessary printing options
    inpfile=inp_files[0]
    #The Flags for the Sections
    AOSection=False
    SCFSection=False
    ForcesSection=False
    NDIGITSKSFlag=False
    KS_Flag=False
    OL_Flag=False
    KS_FilenameFlag=False
    RestartFlag=False
    NDIGITSForcesFlag=False
    Forces_FilenameFlag=False
    with open(inpfile,'r') as f:
        lines=f.readlines()
        for line in lines:
            if len(line.split())>0:
                #Check if AO Matrices are printed properly
                if line.split()[0]=="&AO_MATRICES":
                    if len(line.split())>1 and line.split()[1]!="ON":
                        ValueError("Reconsider the AO_Section in the .inp file! Write '&AO_MATRICES ON' !")
                    else:
                        AOSection=True
                if line.split()[0]=="&FORCES":
                    if len(line.split())>1 and line.split()[1]!="ON":
                        ValueError("Reconsider the &FORCES in the .inp file! Write '&FORCES ON' !")
                    else:
                        ForcesSection=True
                if line.split()[0]=="&SCF":
                    SCFSection=True
                if len(line.split())>1:
                    if line.split()[0]=="&END" and line.split()[1]=="AO_MATRICES":
                        AOSection=False
                    if line.split()[0]=="&END" and line.split()[1]=="SCF":
                        SCFSection=False
                    if line.split()[0]=="&END" and line.split()[1]=="FORCES":
                        ForcesSection=False
                if AOSection:
                    if len(line.split())>1:
                        if line.split()[0]=="KOHN_SHAM_MATRIX" and line.split()[1]==".TRUE.":
                            KS_Flag=True
                        if line.split()[0]=="NDIGITS" and int(line.split()[1])>=10:
                            NDIGITSKSFlag=True
                        if line.split()[0]=="OVERLAP" and line.split()[1]==".TRUE.":
                            OL_Flag=True
                        if line.split()[0]=="FILENAME" and line.split()[1]=="=KSHamiltonian":
                            KS_FilenameFlag=True
                if SCFSection:
                    if len(line.split())>1:
                        if line.split()[0]=="&RESTART" and line.split()[1]=="ON":
                            RestartFlag=True
                if ForcesSection:
                    if len(line.split())>1:
                        if line.split()[0]=="NDIGITS" and int(line.split()[1])>=10:
                            NDIGITSForcesFlag=True
                        if line.split()[0]=="FILENAME" and line.split()[1]=="=Forces":
                            Forces_FilenameFlag=True
    if not KS_Flag:
        ValueError("Reconsider the AO_Section in the .inp file! Write 'KOHN_SHAM_MATRIX .TRUE.' !")
    if not NDIGITSKSFlag:
        ValueError("Reconsider the AO_Section in the .inp file! Write 'NDIGITS 15' !")
    if not OL_Flag:
        ValueError("Reconsider the AO_Section in the .inp file! Write 'OVERLAP .TRUE.' !")
    if not KS_FilenameFlag:
        ValueError("Reconsider the AO_Section in the .inp file! Write 'FILENAME =KSHamiltonian' !")
    if not RestartFlag:
        ValueError("Reconsider the SCF Section in the .inp file! Print Restart files !")
    if not NDIGITSForcesFlag:
        ValueError("Reconsider the FORCES Section in the .inp file! Write 'NDIGITS 15' !")
    if not Forces_FilenameFlag:
        ValueError("Reconsider the AO_Section in the .inp file! Write 'FILENAME =Forces' !")

#-------------------------------------------------------------------------
def readinVibrations(parentfolder):
    try:
        VibrationalFrequencies=np.load("Normal-Mode-Energies.npy")
        CarthesianDisplacements=np.load("normalized-Carthesian-Displacements.npy")
        normfactors=np.load("Norm-Factors.npy")
    except:   
        #Open the Molden file
        mol_files = [f for f in os.listdir(parentfolder) if f.endswith('.mol')]
        if len(mol_files) != 1:
            raise ValueError('InputError: There should be only one mol file in the current directory')
        f = open(parentfolder+"/"+mol_files[0], "r")
        Frequencyflag=False
        Vibrationflag=False
        Normfactorsflag=False
        numvib=0
        CarthesianDisplacements=[]
        normfactors=[]
        mode=[]
        VibrationalFrequencies=[]
        for line in f.readlines():
            if line.split()[0]=="[FREQ]":
                Frequencyflag=True
            if line.split()[0]=="[NORM-FACTORS]":
                Normfactorsflag=True
            if line.split()[0][0]=="[" and Frequencyflag and line.split()[0]!="[FREQ]":
                Frequencyflag=False
            if line.split()[0][0]=="[" and Normfactorsflag and line.split()[0]!="[NORM-FACTORS]":
                Normfactorsflag=False
            if line.split()[0]=="[FR-NORM-COORD]":
                Vibrationflag=True
            if Frequencyflag and line.split()[0]!="[FREQ]":
                VibrationalFrequencies.append(float(line.split()[0]))
            if Normfactorsflag and line.split()[0]!="[NORM-FACTORS]":
                normfactors.append(float(line.split()[0]))
            if Vibrationflag and line.split()[0]!="[FR-NORM-COORD]":
                if line.split()[0]=="vibration":
                    numvib=int(line.split()[1])
                    if numvib!=1:
                        CarthesianDisplacements.append(np.array(mode))
                    mode=[]
                else:
                    mode.append(float(line.split()[0]))
                    mode.append(float(line.split()[1]))
                    mode.append(float(line.split()[2]))
            if line.split()[0][0]=="[" and Vibrationflag and line.split()[0]!="[FR-NORM-COORD]":
                Vibrationflag=False
        f.close()
        CarthesianDisplacements.append(np.array(mode))
    return VibrationalFrequencies,CarthesianDisplacements,normfactors
#-------------------------------------------------------------------------
#########################################################################
## END Cp2k input changer functions
#########################################################################

#########################################################################
## Cp2k functions for convergence ect.
#########################################################################
def CheckConvergence(quantity,path='./'):
    ##Function to check the convergence of DFT total energie w.r.t. different 
    ##numerical input parameters
    ## input:   Quantity                                                    (string,'PW_Cutoff','rel_Cutoff','Cutoff_Rad',Cell_Size)
    ## (opt.)   path   path to the folder of the cooresponding calculation  (string)
    ## output:  -               (void)

    
    if quantity=='PW_Cutoff':
        Cutoff_dirs = [f for f in os.listdir(".") if f.endswith('Ry')]
        Cutoffs=[]
        Energies=[]
        xyzfile= [f for f in os.listdir(path) if f.endswith('.xyz')]
        if len(xyzfile) != 1:
            raise ValueError('InputError: There should be only one inp file in the current directory')
        with open(path+"/"+xyzfile[0]) as g:
            lines=g.readlines()
            numofatoms=int(lines[0])
        for dirs in Cutoff_dirs:
            currentpath=path+"/"+dirs
            inpfile= [f for f in os.listdir(currentpath) if f.endswith('.inp')]
            if len(inpfile) != 1:
                raise ValueError('InputError: There should be only one inp file in the current directory')
            with open(path+"/"+dirs+"/"+inpfile[0]) as g:
                lines=g.readlines()
                for line in lines:
                    if len(line.split())>0:
                        if line.split()[0]=='CUTOFF' or line.split()[0]=='Cutoff':
                            Cutoffs.append(float(line.split()[1]))
            g.close()
            
            outfile= [f for f in os.listdir(currentpath) if f.endswith('.out')]
            if len(outfile) != 1:
                raise ValueError('InputError: There should be only one out file in the current directory')
            with open(path+"/"+dirs+"/"+outfile[0]) as g:
                lines=g.readlines()
                for line in lines:
                    if len(line.split())>0:
                        if (line.split()[0]=='ENERGY|' and line.split()[1]=='Total' and line.split()[2]=='FORCE_EVAL' and line.split()[3]=='('):
                            Energies.append(float(line.split()[8]))
                            print(line.split())
            g.close()
        sorted_indices=np.argsort(Cutoffs)
        Cutoffs=np.array(Cutoffs)[sorted_indices]
        Energies=np.array(Energies)[sorted_indices] 
        diffofE=np.array([np.abs(Energies[it]-(Energies[-1]))/numofatoms for it in range(len(Energies)-1) ])*2.72113838565563E+04
        plt.scatter(Cutoffs[0:-1],diffofE, s=70)
        plt.yscale('log')
        plt.ylabel(r'$\vert E_{\text{tot}}(\Delta)-E_{ \text{tot}}($'+str(Cutoffs[-1])+r'$\text{Ry})\vert $ $[\text{meV}]$',fontsize=30)
        plt.xlabel(r'Plane-Wave-Cutoff $\Delta$ [\text{Ry}]',fontsize=30)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.grid()
        plt.show()
    elif quantity=='Rel_Cutoff':
        Cutoff_dirs = [f for f in os.listdir(".") if f.endswith('Ry')]
        Cutoffs=[]
        Energies=[]
        xyzfile= [f for f in os.listdir(path) if f.endswith('.xyz')]
        if len(xyzfile) != 1:
            raise ValueError('InputError: There should be only one inp file in the current directory')
        with open(path+"/"+xyzfile[0]) as g:
            lines=g.readlines()
            numofatoms=int(lines[0])
        for dirs in Cutoff_dirs:
            currentpath=path+"/"+dirs
            inpfile= [f for f in os.listdir(currentpath) if f.endswith('.inp')]
            if len(inpfile) != 1:
                raise ValueError('InputError: There should be only one inp file in the current directory')
            with open(path+"/"+dirs+"/"+inpfile[0]) as g:
                lines=g.readlines()
                for line in lines:
                    if len(line.split())>0:
                        if line.split()[0]=='REL_CUTOFF' or line.split()[0]=='Rel_Cutoff':
                            Cutoffs.append(float(line.split()[1]))
            g.close()
            
            outfile= [f for f in os.listdir(currentpath) if f.endswith('.out')]
            if len(outfile) != 1:
                raise ValueError('InputError: There should be only one out file in the current directory')
            with open(path+"/"+dirs+"/"+outfile[0]) as g:
                lines=g.readlines()
                for line in lines:
                    if len(line.split())>0:
                        if (line.split()[0]=='ENERGY|' and line.split()[1]=='Total' and line.split()[2]=='FORCE_EVAL' and line.split()[3]=='('):
                            Energies.append(float(line.split()[8]))
                            print(line.split())
            g.close()
        sorted_indices=np.argsort(Cutoffs)
        Cutoffs=np.array(Cutoffs)[sorted_indices]
        Energies=np.array(Energies)[sorted_indices] 
        diffofE=np.array([np.abs(Energies[it]-(Energies[-1]))/numofatoms for it in range(len(Energies)-1) ])*2.72113838565563E+04
        plt.scatter(Cutoffs[0:-1],diffofE, s=70)
        plt.yscale('log')
        plt.ylabel(r'$\vert E_{\text{tot}}(\Delta)-E_{ \text{tot}}($'+str(Cutoffs[-1])+r'$\text{Ry})\vert $ $[\text{meV}]$',fontsize=30)
        plt.xlabel(r'Relative Cutoff $\Delta$ [\text{Ry}]',fontsize=30)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.grid()
        plt.show()
    elif quantity=='Cutoff_Radius':
        Cutoff_dirs = [f for f in os.listdir(".") if f.endswith('A')]
        Cutoffs=[]
        Energies=[]
        xyzfile= [f for f in os.listdir(path) if f.endswith('.xyz')]
        if len(xyzfile) != 1:
            raise ValueError('InputError: There should be only one .xyz file in the current directory')
        with open(path+"/"+xyzfile[0]) as g:
            lines=g.readlines()
            numofatoms=int(lines[0])
        for dirs in Cutoff_dirs:
            currentpath=path+"/"+dirs
            inpfile= [f for f in os.listdir(currentpath) if f.endswith('.inp')]
            if len(inpfile) != 1:
                raise ValueError('InputError: There should be only one inp file in the current directory')
            with open(path+"/"+dirs+"/"+inpfile[0]) as g:
                lines=g.readlines()
                for line in lines:
                    if len(line.split())>0:
                        if line.split()[0]=='CUTOFF_RADIUS':
                            Cutoffs.append(float(line.split()[1]))
            g.close()
            
            outfile= [f for f in os.listdir(currentpath) if f.endswith('.out')]
            if len(outfile) != 1:
                raise ValueError('InputError: There should be only one .out file in the current directory')
            with open(path+"/"+dirs+"/"+outfile[0]) as g:
                lines=g.readlines()
                for line in lines:
                    if len(line.split())>0:
                        if (line.split()[0]=='ENERGY|' and line.split()[1]=='Total' and line.split()[2]=='FORCE_EVAL' and line.split()[3]=='('):
                            Energies.append(float(line.split()[8]))
                            print(line.split())
            g.close()
        sorted_indices=np.argsort(Cutoffs)
        Cutoffs=np.array(Cutoffs)[sorted_indices]
        Energies=np.array(Energies)[sorted_indices] 
        diffofE=np.array([np.abs(Energies[it]-(Energies[-1]))/numofatoms for it in range(len(Energies)-1) ])*2.72113838565563E+04
        plt.scatter(Cutoffs[0:-1],diffofE, s=70)
        plt.yscale('log')
        plt.ylabel(r'$\vert E_{\text{tot}}(R_c)-E_{ \text{tot}}($'+str(Cutoffs[-1])+r'$\textup{~\AA})\vert $ $[\text{meV}]$',fontsize=30)
        plt.xlabel(r'$R_c$ $[\textup{~\AA}]$',fontsize=30)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.grid()
        plt.show()
    elif quantity=='Cell_Size':
        folders= [f for f in os.listdir('.') if f.endswith('A')]
        cellsizes=[]
        absre=[]
        SameSizeFlag=False
        OverlapMatrixFlag=False
        for folder in folders:
            print(folder)
            strcut=folder[8:]
            if 'x' not in strcut[:-1]:
                SameSizeFlag=True
                cellsize=float(strcut[:-1])
                cellsizes.append(cellsize)
                _,OLM=readinMatrices(path+folder)
                print("Reading in Overlapmatrix -> Done")
                Atoms=getAtomicCoordinates(path+folder)
                print("Reading in Atomic Coordinates -> Done")
                Basis=getBasis(path+folder)
                print("Construct carthesian Basis and spherical to cartesian Transformations -> Done")
                if OverlapMatrixFlag==False:
                    Overlapmatrix=getTransformationmatrix(Atoms,Atoms,Basis)
                    OverlapMatrixFlag=True
                diff=np.abs(Overlapmatrix-OLM)
                absre.append(np.max(np.max(diff)))
            else:
                cellsize=(float(strcut[0:2]),float(strcut[3:5]),float(strcut[6:-1]))
                cellsizes.append(cellsize)
                _,OLM=readinMatrices(path+folder)
                print("Reading in Overlapmatrix -> Done")
                Atoms=getAtomicCoordinates(path+folder)
                print("Reading in Atomic Coordinates -> Done")
                Basis=getBasis(path+folder)
                print("Construct carthesian Basis and spherical to cartesian Transformations -> Done")
                if OverlapMatrixFlag==False:
                    Overlapmatrix=getTransformationmatrix(Atoms,Atoms,Basis)
                    OverlapMatrixFlag=True
                diff=np.abs(Overlapmatrix-OLM)
                absre.append(np.max(np.max(diff)))
        if SameSizeFlag:
            plt.scatter(cellsizes,absre,marker="x",s=125)
            plt.yscale('log')
            plt.ylabel(r'$\displaystyle\text{max}_{i,j}\{\vert S_{i,j}-S_{i,j}^{\text{cp2k}}\vert\} $ ',fontsize=50)
            plt.xlabel(r'cell dimension $a$ [$\mathring{\text{A}}$]',fontsize=50)
            plt.xticks(fontsize=35)
            plt.yticks(fontsize=35)
            plt.grid()
            plt.show()
        else:
            fig,ax = plt.subplots()
            ax.scatter(range(len(absre)),absre)
            ax.set_xticks(range(len(absre)))
            ax.set_xticklabels([str(item) for item in cellsizes],fontsize=25)
            ax.set_yscale('log')
            ax.set_ylabel(r'$\displaystyle\text{max}_{i,j}\{\vert S_{i,j}-S_{i,j}^{\text{cp2k}}\vert\} $ ',fontsize=50)
            plt.grid()
            plt.show()
    elif quantity=='Geo_Opt':
        # Read in the out file
        Forces=[]
        Forceflag=False
        with open(path+"/Forces") as g:
                lines=g.readlines()
                for line in lines:
                    if len(line.split())>=1:
                        if line.split()[0]=='SUM' and line.split()[1]=='OF' and line.split()[2]=='ATOMIC':
                            Forceflag=False
                            Forces.append(np.array(Forcecontainer))
                        elif Forceflag:
                            Forcecontainer.append(np.array([float(line.split()[3]),float(line.split()[4]),float(line.split()[5])]))
                        elif line.split()[0]=='#' and line.split()[1]=='Atom' and line.split()[2]=='Kind':
                            Forceflag=True
                            Forcecontainer=[]
        Forces=np.array(Forces)
        maxforces=[]
        rmsforces=[]
        iterations=[]
        iter=0
        for fo in Forces:
            iterations.append(iter)
            maxforces.append(np.max(np.max(fo)))
            rmsforces.append(np.sqrt(np.mean(np.square(fo))))
            iter+=1
        plt.scatter(iterations,maxforces)
        #plt.scatter(iterations,rmsforces)
        plt.yscale('log')
        plt.ylabel(r'$\text{max}_{s,\alpha}\{F_{s,\alpha}(n)\}$ ',fontsize=30)
        plt.xlabel(r'Iteration $n$',fontsize=30)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.grid()
        plt.show()
    else:
        print("Option not recognized! Use  'PW_Cutoff','Rel_Cutoff','Cutoff_Radius','Cell_Size' or 'Geo_Opt' as input.")
#########################################################################
## END Cp2k functions for convergence ect.
#########################################################################

#########################################################################
## Parser and Basis Construction functions 
#########################################################################
def getxyzfilename(path="./"):
    ##returns the xyz filename in path returns error, when in this folder are two xyz files
    ## input:
    ## (opt.)   folder              path to the folder of the .xyz file         (string)
    ## output:  fileanem            list of sublists. 

    #get the Projectname
    xyz_files = [f for f in os.listdir(path) if f.endswith('.xyz')]
    if len(xyz_files) != 1:
        raise ValueError('InputError: There should be only one *.xyz file in the current directory')
    filename = path+"/"+xyz_files[0]
    return filename

def getAtomicCoordinates(folder="./"):
    ##Reads in the atomic coordinates from a provided xyz file (these coordinates are independent from cell vectors)! 
    ## input:
    ## (opt.)   folder              path to the folder of the .xyz file         (string)
    ## output:  Atoms               list of sublists. 
    ##                              Each of the sublists has five elements. 
    ##                              Sublist[0] contains the atomorder as a int.
    ##                              Sublist[1] contains the symbol of the atom.
    ##                              Sublist[2:] containst the x y z coordinates.
    filename=getxyzfilename(folder)
    Atoms=[]
    with open(filename) as f:
        lines=f.readlines()
        it=1
        for l in lines[2:]:
            Atoms.append([it,l.split()[0],float(l.split()[1]),float(l.split()[2]),float(l.split()[3])])
            it+=1
    f.close()
    return Atoms
#-------------------------------------------------------------------------
def getMatricesfromfile_mulOne(parentfolder="./",filename="KSHamiltonian"):
    ##Reads in the overlapmatrix and the KSHamiltonian in case of Spin multiplicity 1
    ## input:
    ## (opt.)   folder              path to the folder of the KSHamiltonian file         (string)
    ## output:  KSlines             list of strings of the converges KS Hamiltonian
    ##          OLMlines            list of strings of the overlapmatrix
    ##          NumBasisfunctions   int number of (spherical) Basisfunctions in Basis set
    with open(parentfolder+"/"+filename,'r+') as f:
        OLMlines=[]
        OLMFlag=False
        Niter=0
        Nlines=0
        for line in f:
            if len(line.split())>=1:
                if line.split()[0]=="KOHN-SHAM" and  line.split()[1]=="MATRIX":
                    OLMFlag=False
                    Niter+=1
                if line.split()[0]=="OVERLAP" and  line.split()[1]=="MATRIX" and OLMFlag:
                    OLMFlag=False
                if OLMFlag:
                    OLMlines.append(line)
                elif line.split()[0]=="OVERLAP" and  line.split()[1]=="MATRIX":
                    OLMFlag=True
            Nlines+=1
    NumBasisfunctions=0
    with open(parentfolder+"/"+filename,'r+') as f:
        KSlines=[]
        KSFlag=False
        Niter2=0
        Nlines2=0
        for line in f:
            if len(line.split())>=1:
                if line.split()[0]=="OVERLAP" and  line.split()[1]=="MATRIX":
                    KSFlag=False
                if line.split()[0]=="KOHN-SHAM" and  line.split()[1]=="MATRIX":
                    Niter2+=1
                    KSFlag=False
                if KSFlag:
                    KSlines.append(line)
                if line.split()[0]=="KOHN-SHAM" and  line.split()[1]=="MATRIX" and Niter2==Niter:
                    KSFlag=True
                if Nlines2==Nlines-3:
                    NumBasisfunctions=int(line.split()[0])
            Nlines2+=1
    return KSlines,OLMlines,NumBasisfunctions
def getMatricesfromfile_mulTwo(parentfolder,filename="KSHamiltonian"):
    with open(parentfolder+"/"+filename,'r+') as f:
        OLMlines=[]
        OLMFlag=False
        Niter0=0
        Niter1=0
        Nlines=0
        for line in f:
            if len(line.split())>=1:
                if line.split()[0]=="KOHN-SHAM" and  line.split()[1]=="MATRIX":
                    if line.split()[2]=="FOR" and  line.split()[3]=="ALPHA":
                        OLMFlag=False
                        Niter0+=1
                if line.split()[0]=="KOHN-SHAM" and  line.split()[1]=="MATRIX":
                    if line.split()[2]=="FOR" and  line.split()[3]=="BETA":
                        OLMFlag=False
                        Niter1+=1
                if line.split()[0]=="OVERLAP" and  line.split()[1]=="MATRIX" and OLMFlag:
                    OLMFlag=False
                if OLMFlag:
                    OLMlines.append(line)
                elif line.split()[0]=="OVERLAP" and  line.split()[1]=="MATRIX":
                    OLMFlag=True
            Nlines+=1
    NumBasisfunctions=0
    with open(parentfolder+"/"+filename,'r+') as f:
        KSlines_alpha=[]
        KSFlag1=False
        Niter2=0
        Nlines2=0
        for line in f:
            if len(line.split())>=1:
                if line.split()[0]=="OVERLAP" and  line.split()[1]=="MATRIX":
                    KSFlag1=False
                if line.split()[0]=="KOHN-SHAM" and  line.split()[1]=="MATRIX":
                    if line.split()[2]=="FOR" and  line.split()[3]=="BETA":
                        KSFlag1=False
                if line.split()[0]=="KOHN-SHAM" and  line.split()[1]=="MATRIX":
                    if line.split()[2]=="FOR" and  line.split()[3]=="ALPHA":
                        KSFlag1=False
                        Niter2+=1
                if KSFlag1:
                    KSlines_alpha.append(line)
                if line.split()[0]=="KOHN-SHAM" and  line.split()[1]=="MATRIX":
                    if line.split()[2]=="FOR" and  line.split()[3]=="ALPHA" and Niter2==Niter0:
                        KSFlag1=True
                if Nlines2==Nlines-3:
                    NumBasisfunctions=int(line.split()[0])
            Nlines2+=1
    with open(parentfolder+"/"+filename,'r+') as f:
        KSlines_beta=[]
        KSFlag1=False
        Niter2=0
        Nlines2=0
        for line in f:
            if len(line.split())>=1:
                if line.split()[0]=="OVERLAP" and  line.split()[1]=="MATRIX":
                    KSFlag1=False
                if line.split()[0]=="KOHN-SHAM" and  line.split()[1]=="MATRIX":
                    if line.split()[2]=="FOR" and  line.split()[3]=="ALPHA":
                        KSFlag1=False
                if line.split()[0]=="KOHN-SHAM" and  line.split()[1]=="MATRIX":
                    if line.split()[2]=="FOR" and  line.split()[3]=="BETA":
                        KSFlag1=False
                        Niter2+=1
                if KSFlag1:
                    KSlines_beta.append(line)
                if line.split()[0]=="KOHN-SHAM" and  line.split()[1]=="MATRIX":
                    if line.split()[2]=="FOR" and  line.split()[3]=="BETA" and Niter2==Niter1:
                        KSFlag1=True
            Nlines2+=1
    return KSlines_alpha,KSlines_beta,OLMlines,NumBasisfunctions
def checkforSpinMultiplicity(path="./"):
    ##opens the .inp file in the directory and checks, if Multiplicity is 1
    ## input:
    ## (opt.)   folder              path to the folder of the .inp file         (string)
    ## output:  mul    (int)        the multiplicity of the system. 

    #get the Projectname
    inp_files = [f for f in os.listdir(path) if f.endswith('.inp')]
    if len(inp_files) != 1:
        raise ValueError('InputError: There should be only one *.inp file in the current directory')
    pathtofile = path+"/"+inp_files[0]
    mul=1
    with open(pathtofile,"r") as f:
        for line in f:
            if len(line.split())>1:
                if line.split()[0]=="MULTIPLICITY":
                    mul=int(line.split()[1])
    return mul
#-------------------------------------------------------------------------
def readinMatrices(parentfolder="./",filename='KSHamiltonian'):
    ## Reads the Kohn-Sham Hamiltonian (for spin-species alpha/-beta) and the overlapmatrix from a provided file
    ## input:
    ## (opt.)   filename            path to the Hamiltonian file        (string)
    ## output:  KSHamiltonian       Kohn-Sham Hamiltonian   symmetric np.array(NumBasisfunctions,Numbasisfunction)
    ##                              In case of Multiplicity 2 KSHamiltonian is the KS_Hamiltonian of the alpha spin!
    ##          OLM                 Overlapmatrix           symmetric np.array(NumBasisfunctions,Numbasisfunction)
    mul=checkforSpinMultiplicity(parentfolder)
    if mul==1:
        try:
            KSlines,OLMlines,NumBasisfunctions=getMatricesfromfile_mulOne(parentfolder,filename)
            KSHamiltonian=np.zeros((NumBasisfunctions,NumBasisfunctions))
            OLM=np.zeros((NumBasisfunctions,NumBasisfunctions))
            for l in KSlines:
                if len(l.split())<5:
                    jindices=[int(j)-1 for j in l.split()]
                else:
                    iindix=int(l.split()[0])-1
                    iter=4
                    for jindex in jindices:
                        KSHamiltonian[iindix][jindex]=float(l.split()[iter])
                        iter+=1
            for l in OLMlines:
                if len(l.split())<5:
                    jindices=[int(j)-1 for j in l.split()]
                else:
                    iindix=int(l.split()[0])-1
                    iter=4
                    for jindex in jindices:
                        OLM[iindix][jindex]=float(l.split()[iter])
                        iter+=1
            np.save(parentfolder+"/"+"KSHamiltonian",KSHamiltonian)
            np.save(parentfolder+"/"+"OLM",OLM)
            os.remove(parentfolder+"/"+"KSHamiltonian")
        except:
            KSHamiltonian=np.load(parentfolder+"/KSHamiltonian.npy")
            OLM=np.load(parentfolder+"/OLM.npy")
    elif mul==2:
        try:
            KSlines_alpha,KSlines_beta,OLMlines,NumBasisfunctions=getMatricesfromfile_mulTwo(parentfolder,filename)
            KSHamiltonian_alpha=np.zeros((NumBasisfunctions,NumBasisfunctions))
            KSHamiltonian_beta=np.zeros((NumBasisfunctions,NumBasisfunctions))
            OLM=np.zeros((NumBasisfunctions,NumBasisfunctions))
            for l in KSlines_alpha:
                if len(l.split())<5:
                    jindices=[int(j)-1 for j in l.split()]
                else:
                    iindix=int(l.split()[0])-1
                    iter=4
                    for jindex in jindices:
                        KSHamiltonian_alpha[iindix][jindex]=float(l.split()[iter])
                        iter+=1
            for l in KSlines_beta:
                if len(l.split())<5:
                    jindices=[int(j)-1 for j in l.split()]
                else:
                    iindix=int(l.split()[0])-1
                    iter=4
                    for jindex in jindices:
                        KSHamiltonian_beta[iindix][jindex]=float(l.split()[iter])
                        iter+=1
            for l in OLMlines:
                if len(l.split())<5:
                    jindices=[int(j)-1 for j in l.split()]
                else:
                    iindix=int(l.split()[0])-1
                    iter=4
                    for jindex in jindices:
                        OLM[iindix][jindex]=float(l.split()[iter])
                        iter+=1
            np.save(parentfolder+"/"+"KSHamiltonian_alpha",KSHamiltonian_alpha)
            np.save(parentfolder+"/"+"KSHamiltonian_beta",KSHamiltonian_beta)
            np.save(parentfolder+"/"+"OLM",OLM)
            KSHamiltonian=KSHamiltonian_alpha
            os.remove(parentfolder+"/"+"KSHamiltonian")
        except:
            KSHamiltonian_alpha=np.load(parentfolder+"/KSHamiltonian_alpha.npy")
            KSHamiltonian_beta=np.load(parentfolder+"/KSHamiltonian_beta.npy")
            OLM=np.load(parentfolder+"/OLM.npy")
            KSHamiltonian=KSHamiltonian_alpha
    return KSHamiltonian,OLM
def readinMos(parentfolder="./"):
    ## Reads the Molecular Orbitals from a provided file
    ## input:
    ## (opt.)   filename            path to the MOs file        (string)
    ## output:  MOs                symmetric np.array(NumBasisfunctions,Numbasisfunction)       Expansion coefficients of the MOs in terms of AO's 
    ## Example: MOs[:,0] are the expansion coefficients of the MO 0 in the canonically ordered atomic Basis
    try:
        lastMOstart=0
        with open(parentfolder+"/MOs") as f:
            lineiter=0
            for line in f:
                if len(line.split())>6:
                    if line.split()[0]=="MO|" and line.split()[1]=="EIGENVALUES," and line.split()[2]=="OCCUPATION" and line.split()[3]=="NUMBERS," and line.split()[4]=="AND" and line.split()[5]=="SPHERICAL" and line.split()[6]=="EIGENVECTORS":
                        lastMOstart=lineiter
                lineiter+=1
        MOstring=[]
        BasisFKTIndex=-10**(-20)
        with open(parentfolder+"/MOs") as f:
            lineiter=0
            for line in f:
                if lineiter>=lastMOstart:
                    MOstring.append(line)
                if len(line.split())>5:
                    if line.split()[1]=='E(Fermi):':
                        BasisFKTIndex=lineiter-2
                lineiter+=1
        BasisFKTIndex-=lastMOstart
        NUM_BASIS_FKT=int(MOstring[BasisFKTIndex].split()[1])
        MOs=np.zeros((NUM_BASIS_FKT,NUM_BASIS_FKT))
        Basenumber=0
        for line in MOstring:
            splited_line=line.split()[1:]
            if len(splited_line)>=5:
                if splited_line[0].isdigit() and splited_line[1].isdigit() and splited_line[2].isalpha():
                    aoBasisindex=int(splited_line[0])-1
                    iterator=0
                    for number_string in splited_line[4:]:
                        number=float(number_string)
                        moindex=Basenumber+iterator
                        MOs[aoBasisindex,moindex]=number
                        iterator+=1
                    if aoBasisindex==NUM_BASIS_FKT-1:
                        Basenumber+=4
        np.save("MOs",MOs)
        os.remove(parentfolder+"/"+"MOs")
    except:
        MOs=np.load("MOs.npy")
    return MOs
def getPhaseOfMO(MO):
    ## Definition of the phase convention
    ## input:   MO                 np.array(NumBasisfunctions)       Expansion coefficients of the MO in terms of AO's 
    ## output:  MOphases            list of integers            (list)       
    ## Example: MOphases[mo_index] is the phase (in +/- 1) defined by the function below (convention)
    
    #The first non-vanishing element
    numberofpositivephases=0
    numberofnegativephases=0
    for it in range(len(MO)):
        if np.abs(MO[it])>10**(-10) and MO[it]>0:
            numberofpositivephases+=1
        if np.abs(MO[it])>10**(-10) and MO[it]<0:
            numberofnegativephases+=1
    if numberofpositivephases-numberofnegativephases>10**(-14):
        phase=1.0
    else:
        phase=-1.0
    return phase
def getMOsPhases(filename="./"):
    ## Reads the Molecular Orbitals from a provided file
    ## input:   MOs                 np.array(NumBasisfunctions,Numbasisfunction)       Expansion coefficients of the MOs in terms of AO's (index 1 AO index, index2 MO index)
    ## (opt.)   filename            path to the MOs file        (string)
    ## output:  MOphases            list of integers            (list)       
    ## Example: MOphases[mo_index] is the phase (in +/- 1) defined by the function below (convention)
    MOs=readinMos(filename)
    MOphases=[]
    for moindex in range(np.shape(MOs)[1]):
        MOphases.append(getPhaseOfMO(MOs[:,moindex]))
    return MOphases
def compressKSfile(parentfolder="./"):
    _,_=readinMatrices(parentfolder)
#-------------------------------------------------------------------------
def readinForces(folder):
    ## Reads in the Forces on atoms from a provided file
    ## input:
    ##          folder       path to the folder       (string)
    ## output:  Forces       Force on the atoms list of subslists
    ##                       list[s] are the components of the force
    ##                       on atom s+1 (atomnumbering) in x y z dir.
    ##                       unit E_h/a_0 (Hartree energy/Bohr radius)             
    f=open(folder+"/"+"Forces")
    lines=f.readlines()
    readinflag=False
    Forces=[]
    for line in lines:
        if len(line.split())>=2:
            if line.split()[0]=="SUM" and line.split()[1]=="OF":
                readinflag=False
            if readinflag:
                Forces.append(float(line.split()[3]))
                Forces.append(float(line.split()[4]))
                Forces.append(float(line.split()[5]))
            if line.split()[0]=="#" and line.split()[1]=="Atom":
                readinflag=True
            
    f.close()
    return Forces
def readinXYZ(parentfolder="./"):
    ConFactor=ConversionFactors()
    ##################
    #Get the .xyz file
    ##################
    xyz_files = [f for f in os.listdir(parentfolder) if f.endswith('.xyz')]
    if len(xyz_files) != 1:
        raise ValueError('InputError: There should be only one xyz file in the current directory')
    ##########################################
    #Get the number of atoms from the xyz file
    ##########################################
    xyzfilename=xyz_files[0]
    EqCoordinates=[]
    with open(parentfolder+"/"+xyzfilename) as g:
        lines=g.readlines()
        atomorder=[]
        for line in lines[2:]:
            if len(line.split())>0:
                atom=line.split()[0]
                atomorder.append(atom)
                EqCoordinates.append([float(line.split()[1]),float(line.split()[2]),float(line.split()[3])])
    return atomorder,EqCoordinates 
def readinBasisVectors(parentfolder="./"):
    ## Reads in the normalized Basis vectors in which the Hessian is 
    ## represented. These are the directions, in which the atoms have
    ## been displaced. Also returns the displacementfactor and the unit
    ## of the displacementfactor
    ## input: 
    ## (opt.)   parentfolder    path to the folder of the BasisHessian file         (string)
    ## output:  
    ##          BasisVectors    normalized displaced vectors                        (list of np.arrays)     
    ##          delta           displacementfactor                                  (float)
    ##          unit            unit of the displacementfactor                      (string, either 'Bohr' or 'sqrt(u)*Bohr') 

    #Get the .xyz file
    xyz_files = [f for f in os.listdir(parentfolder) if f.endswith('.xyz')]
    if len(xyz_files) != 1:
        raise ValueError('InputError: There should be only one xyz file in the current directory')
    #Get the number of atoms from the xyz file
    xyzfilename=xyz_files[0]
    ##########################################
    #Get the number of atoms from the xyz file
    ##########################################
    xyzfilename=xyz_files[0]
    numofatoms=0
    with open(parentfolder+"/"+xyzfilename) as g:
        lines=g.readlines()
        numofatoms=int(lines[0])
    delta=0
    #Read in the Basis Vectors:
    BasisVectors=[]
    with open(parentfolder+"BasisHessian") as g:
        lines=g.readlines()
        if lines[0].split("=")[0]=="delta":
            delta=float(lines[0].split("=")[1])
        else:
            raise ValueError('InputError: delta not given!')
        if lines[1].split("=")[0]=="unit":
            unit=lines[1].split("=")[1][:-1]
            if unit !="Bohr" and unit !="sqrt(u)*Bohr":
                raise ValueError("InputError: No proper unit given! Either 'Bohr' or 'sqrt(u)*Bohr' ")
        else:
            raise ValueError('InputError: Give unit of displacement!')
        for line in lines[4:]:
            if len(line)>0:
                if line.split()[0]=="Basisvector" and int(line.split()[1])==1:
                    basevector=np.zeros(3*numofatoms)
                    it=0
                elif line.split()[0]=="Basisvector" and int(line.split()[1])!=1:
                    BasisVectors.append(basevector)
                    basevector=np.zeros(3*numofatoms)
                    it=0
                else:
                    basevector[it]=float(line.split()[1])
                    basevector[it+1]=float(line.split()[2])
                    basevector[it+2]=float(line.split()[3])
                    it+=3
        BasisVectors.append(basevector) #append the last base vector
        return BasisVectors,delta
#-------------------------------------------------------------------------
def getcs():
    #returns the representation of a given solid harmonics (l,m) in terms of homogenious monomials
    #input:     l positive integer
    #           m out of [-l,-l+1,...,l-1,l] 
    #format: 
    # cs=getcs(l,m)
    # is=cs[it][0] is the representation of the monomial in terms of 
    # x^is[0]y^is[1]z^is[2]
    # and cs[it][1] the corresponding prefactor (consistent with CP2K convention)
    cs={}

    cs['s']=[[[0,0,0],0.5/np.sqrt(np.pi)]]

    cs['py']=[[[0,1,0],np.sqrt(3./(4.0*np.pi))]]
    cs['pz']=[[[0,0,1],np.sqrt(3./(4.0*np.pi))]]
    cs['px']=[[[1,0,0],np.sqrt(3./(4.0*np.pi))]]

    cs['d-2']=[[[1,1,0],0.5*np.sqrt(15./np.pi)]]
    cs['d-1']=[[[0,1,1],0.5*np.sqrt(15./np.pi)]]
    cs['d0']=[[[2,0,0],-0.25*np.sqrt(5./np.pi)],[[0,2,0],-0.25*np.sqrt(5./np.pi)],[[0,0,2],0.5*np.sqrt(5./np.pi)]]
    cs['d+1']=[[[1,0,1],0.5*np.sqrt(15./np.pi)]]
    cs['d+2']=[[[2,0,0],0.25*np.sqrt(15./np.pi)],[[0,2,0],-0.25*np.sqrt(15./np.pi)]]

    cs['f-3']=[[[2,1,0],0.75*np.sqrt(35./2./np.pi)],[[0,3,0],-0.25*np.sqrt(35./2./np.pi)]]
    cs['f-2']=[[[1,1,1],0.5*np.sqrt(105./np.pi)]]
    cs['f-1']=[[[0,1,2],np.sqrt(21./2./np.pi)],[[2,1,0],-0.25*np.sqrt(21./2./np.pi)],[[0,3,0],-0.25*np.sqrt(21./2./np.pi)]]
    cs['f0']=[[[0,0,3],0.5*np.sqrt(7./np.pi)],[[2,0,1],-0.75*np.sqrt(7/np.pi)],[[0,2,1],-0.75*np.sqrt(7/np.pi)]]
    cs['f+1']=[[[1,0,2],np.sqrt(21./2./np.pi)],[[1,2,0],-0.25*np.sqrt(21./2./np.pi)],[[3,0,0],-0.25*np.sqrt(21./2./np.pi)]]
    cs['f+2']=[[[2,0,1],0.25*np.sqrt(105./np.pi)],[[0,2,1],-0.25*np.sqrt(105./np.pi)]]
    cs['f+3']=[[[3,0,0],0.25*np.sqrt(35./2./np.pi)],[[1,2,0],-0.75*np.sqrt(35./2./np.pi)]]

    cs['g-4']=[[[3,1,0],0.75*np.sqrt(35./np.pi)],[[1,3,0],-0.75*np.sqrt(35./np.pi)]] 
    cs['g-3']=[[[2,1,1],9.0*np.sqrt(35./(2*np.pi))/4.0],[[0,3,1],-0.75*np.sqrt(35./(2.*np.pi))]] 
    cs['g-2']=[[[1,1,2],18.0*np.sqrt(5./(np.pi))/4.0],[[3,1,0],-3.*np.sqrt(5./(np.pi))/4.0],[[1,3,0],-3.*np.sqrt(5./(np.pi))/4.0]] 
    cs['g-1']=[[[0,1,3],3.0*np.sqrt(5./(2*np.pi))],[[2,1,1],-9.0*np.sqrt(5./(2*np.pi))/4.0],[[0,3,1],-9.0*np.sqrt(5./(2*np.pi))/4.0]] 
    cs['g0']=[[[0,0,4],3.0*np.sqrt(1./(np.pi))/2.0],[[4,0,0],9.0*np.sqrt(1./(np.pi))/16.0],[[0,4,0],9.0*np.sqrt(1./(np.pi))/16.0],[[2,0,2],-9.0*np.sqrt(1./np.pi)/2.0],[[0,2,2],-9.0*np.sqrt(1./np.pi)/2.0],[[2,2,0],9.0*np.sqrt(1./np.pi)/8.0]]
    cs['g+1']=[[[1,0,3],3.0*np.sqrt(5./(2*np.pi))],[[1,2,1],-9.0*np.sqrt(5./(2*np.pi))/4.0],[[3,0,1],-9.0*np.sqrt(5./(2*np.pi))/4.0]]
    cs['g+2']=[[[2,0,2],18.0*np.sqrt(5./(np.pi))/8.0],[[0,2,2],-18.*np.sqrt(5./(np.pi))/8.0],[[0,4,0],3.*np.sqrt(5./(np.pi))/8.0],[[4,0,0],-3.*np.sqrt(5./(np.pi))/8.0]]
    cs['g+3']=[[[1,2,1],-9.0*np.sqrt(35./(2*np.pi))/4.0],[[3,0,1],0.75*np.sqrt(35./(2.*np.pi))]]
    cs['g+4']=[[[4,0,0],3.0*np.sqrt(35./np.pi)/16.0],[[2,2,0],-18.0*np.sqrt(35./np.pi)/16.0],[[0,4,0],3.0*np.sqrt(35./np.pi)/16.0]]

    return cs
#-------------------------------------------------------------------------
def getAngularMomentumString(l,m):
    ## Transforms the angular momentum notation (l,m) into the 's', 'py','pz','px' ect. notation
    ## input:   l                       angular momentum quantum number            (int)
    ##          m                       magnetic quantum number                    (int)
    ## output:  s                       the s- notation for the (l,m) pair         (string)  
    if l==0:
        s='s'
    elif l==1 and m==-1:
        s='py'
    elif l==1 and m==0:
        s='pz'
    elif l==1 and m==1:
        s='px'
    elif l==2:
        if m>0:
            s='d'+'+'+str(m)
        else:
            s='d'+str(m)
    elif l==3:
        if m>0:
            s='f'+'+'+str(m)
        else:
            s='f'+str(m)
    elif l==4:
        if m>0:
            s='g'+'+'+str(m)
        else:
            s='g'+str(m)
    elif l==5:
        if m>0:
            s='h'+'+'+str(m)
        else:
            s='h'+str(m)
    else:
        print("Higher order not yet implemented")
    return s

#-------------------------------------------------------------------------
def getNormalizationfactor(alpha,l):
    ## Transformationfactor between the normalized contracted cartesian Basis set from the data directory,
    ## and the not normalized Basis set used in the QS routines of cp2k. 
    ## This means the contraction coefficients c_dd from the data directory are connected with
    ## those used in the QS routines c_QS via c_QS(alpha,l)=Output(alpha,l)*c_dd(alpha,l), 
    ## where Output is the output of this function.
    ## input:   alpha                       the exponent of the Gaussian            (float)
    ##          l                           the angular momentum quantum number     (int)
    ## output:  The transformation factor            
    return alpha**(0.5*l+0.75)*2**(l)*(2.0/np.pi)**(0.75)
#-------------------------------------------------------------------------
def getBasisSetName(path,cp2kpath=pathtocp2k):
    ## Reads in from the .inp file in path the Basis sets used. Parses the corresponding 
    ## data from the cp2kpath/data/Basis_Set file and returns this parsed data as a list.
    ## Each element in this list is a string of the corresponding line in the Basis set file
    ## input:   path                path to the folder of the .inp file         (string)
    ## (opt.)   cp2kpath            path to the cp2k folder                     (string)
    ## output:  BasisInfoReadin                                                 (list of strings)       

    #open the .inp file
    inpfile= [f for f in os.listdir(path) if f.endswith('.inp')]
    if len(inpfile) != 1:
        raise ValueError('InputError: There should be only one inp file in the current directory')
    atoms=[]
    BasisSetNames=[]
    BasisSetFileName='empty'
    BasisSetNameFlag=False
    with open(path+"/"+inpfile[0],'r') as g:
        lines=g.readlines()
        for line in lines:
            if len(line.split())>0:
                if line.split()[0]=="BASIS_SET_FILE_NAME":
                    BasisSetFileName=line.split()[1]
                if line.split()[0]=="&KIND":
                    atoms.append(line.split()[1])
                    BasisSetNameFlag=True
                if line.split()[0]=="END" and line.split()[1]=="&KIND":
                    BasisSetNameFlag=False
                if BasisSetNameFlag and line.split()[0]=="BASIS_SET":
                    BasisSetNames.append(line.split()[1])
    atomStrings=['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr']
    print(BasisSetFileName)
    BasisInfoReadin=[]
    ReadinFlag=False
    for it in range(len(atoms)):
        with open(cp2kpath+'/data/'+BasisSetFileName,'r') as g:
            for l in g:
                if len(l.split())>=1:
                    numericflag=l.split()[0][0].isnumeric()
                if len(l.split())==2 and not numericflag:
                    if l.split()[0]==atoms[it] and l.split()[1]==BasisSetNames[it]:
                        ReadinFlag=True
                    if (l.split()[0] in atomStrings and l.split()[0] not in atoms)or(l.split()[1] !=BasisSetNames[it]):
                        ReadinFlag=False
                elif len(l.split())>2 and not numericflag:
                    if (l.split()[0]==atoms[it] and l.split()[1]==BasisSetNames[it]) or (l.split()[0]==atoms[it] and l.split()[2]==BasisSetNames[it]):
                        ReadinFlag=True
                    elif bool(l.split()[0] in atomStrings and l.split()[0] not in atoms) or (bool(l.split()[1] !=BasisSetNames[it]) or bool(l.split()[2] !=BasisSetNames[it])) :
                        ReadinFlag=False
                if ReadinFlag:
                    BasisInfoReadin.append(l)
    for it in range(len(BasisInfoReadin)):
        item=BasisInfoReadin[it]
        item=item[:-1]
        BasisInfoReadin[it]=item
    return BasisInfoReadin
#-------------------------------------------------------------------------
def getBasis(filename):
    ##Constructs the (non-orthorgonal) Basis used in the CP2K calculation 
    ## input:
    ## (opt.)   filename            path to the calculation folder                   (string)
    ## output:  Basis               dic. of lists of sublists. The keys of the dic. are
    ##                              the atomic symbols.
    ##                              list contains sublist, where each Basisfunction of the 
    ##                              considered atom corresponds the one sublist.
    ##                              sublist[0] contains the set index as a string. 
    ##                              sublist[1] contains the shell index as a string
    ##                              sublist[2] contains the angular momentum label 
    ##                              as a string (e.g. shellindex py ect.)
    ##                              sublist[3:] are lists with two elements.
    ##                              The first corresponds the the exponent of the Gaussian
    ##                              The second one corresponds to the contraction coefficient
    BasisInfoReadin=getBasisSetName(filename)
    atoms=['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr']
    BasisSet={}
    atom='NotDefined'
    Atombasis=[]
    newAtomSetflag=False
    newSetflag=False
    readinBasisfunctionsflag=False
    setcounter=1
    linecounter=0
    NumberofExponents=0
    numberofBasissets=0
    for line in BasisInfoReadin:
        splitedline=line.split()
        if len(splitedline)>0:
            firstcaracter=splitedline[0]
            if readinBasisfunctionsflag:
                if linecounter==NumberofExponents+1:
                    readinBasisfunctionsflag=False
                    newSetflag=True
                    setcounter+=1
                else:
                    exponent=float(splitedline[0])
                    coefficientiter=1
                    it=0
                    for it1 in range(len(ls)):
                        l=ls[it1]
                        for it2 in range(NumofangularmomentumBasisfunctions[it1]):
                            coefficient=float(splitedline[coefficientiter])
                            for m in range(-l,l+1):
                                Basisfunctions[it].append([exponent,getNormalizationfactor(exponent,l)*coefficient]) #
                                it+=1
                            coefficientiter+=1
                    if linecounter==NumberofExponents:
                        Atombasis.append(Basisfunctions)
                    linecounter+=1
            if newSetflag and setcounter<=numberofBasissets:
                minprincipalQuantumnumber=int(splitedline[0])
                lmin=int(splitedline[1])
                lmax=int(splitedline[2])
                ls=np.array([lmin+it for it in range(0,lmax-lmin+1)])
                NumberofExponents=int(splitedline[3])
                NumofangularmomentumBasisfunctions=[]
                for split in splitedline[4:]:
                    NumofangularmomentumBasisfunctions.append(int(split))
                if (lmax-lmin+1)!=len(NumofangularmomentumBasisfunctions):
                    ValueError("Number of Basis functions does not fit!")
                NumofangularmomentumBasisfunctions=np.array(NumofangularmomentumBasisfunctions)
                readinBasisfunctionsflag=True
                newSetflag=False
                shell=0
                Basisfunctions=[]
                for it1 in range(len(ls)):
                    l=ls[it1]
                    for it2 in range(NumofangularmomentumBasisfunctions[it1]):
                        shell+=1
                        for m in range(-l,l+1):
                            Basisfunctions.append([str(setcounter),str(shell),str(minprincipalQuantumnumber+shell-1)+getAngularMomentumString(l,m)])
                linecounter=1
            if  newAtomSetflag:
                numberofBasissets=int(firstcaracter)
                setcounter=1
                newAtomSetflag=False
                newSetflag=True
            if setcounter==numberofBasissets and linecounter==NumberofExponents+1:
                Basis=[]
                for it1 in range(len(Atombasis)):
                    for it2 in range(len(Atombasis[it1])):
                        Basis.append(Atombasis[it1][it2])
                BasisSet[atom]=Basis
            if firstcaracter in atoms:
                Atombasis=[]
                atom=firstcaracter
                newAtomSetflag=True
    #Normalize the Basis
    for key in BasisSet.keys():
        for it in range(len(BasisSet[key])):
            lm1=BasisSet[key][it][2][1:]
            lm2=lm1
            dalpha1=BasisSet[key][it][3:]
            dalpha2=dalpha1
            #Check normalization
            R1=np.array([0.0,0.0,0.0])
            R2=np.array([0.0,0.0,0.0])
            normfactor=1.0/np.sqrt(getoverlap(R1,lm1,dalpha1,R2,lm2,dalpha2))
            for it2 in range(len(BasisSet[key][it])-3):
                BasisSet[key][it][it2+3][1]*=normfactor
    return BasisSet

def readinExcitedStatesCP2K(path,minweight=0.01):
    ## Parses the excited states from a TDDFPT calculation done in CP2K  
    ## input:
    ## (opt.)   minweight           minimum amplitude to consider
    ## output: 
	#get the output file 
	out_files = [f for f in os.listdir(path) if f.endswith('.out')]
	if len(out_files) != 0:
        #Check if the calculation has run through
		if len(out_files)>1:
			raise ValueError('InputError: There should be at most .out file in the '+path+' directory')
		readflag=False
		energies=[]
		with open(path+"/"+out_files[0],'r') as f:
			for line in f:
				if len(line.split())>4:
					if line.split()[0]=="TDDFPT" and line.split()[1]==":" and line.split()[2]=="CheckSum" and line.split()[3]=="=":
						readflag=False
					if readflag:
						energy=line.split()[2]
						energies.append(float(energy))
					if line.split()[0]=="number" and line.split()[1]=="energy" and line.split()[2]=="(eV)" and line.split()[3]=="x" and line.split()[4]=="y":
						readflag=True
		states=[]
		stateiterator=[]
		stateiteratorflag=False
		with open(path+"/"+out_files[0],'r') as f:
			for line in f:
				if len(line.split())>0 and stateiteratorflag:
					if line.split()[0]=="-------------------------------------------------------------------------------":
						stateiteratorflag=False
						states.append([energies[it],stateiterator])
				if stateiteratorflag and len(line.split())==3 and represents_int(line.split()[1]):
					if abs(float(line.split()[2]))>=minweight:
						stateiterator.append([int(line.split()[0])-getHOMOId(path)-1,int(line.split()[1])-getHOMOId(path)-1,float(line.split()[2])])
				if len(line.split())==3:
					if represents_int(line.split()[0]) and not represents_int(line.split()[1]) and line.split()[2]=="eV":
						if int(line.split()[0])<=len(energies):
							it=int(line.split()[0])-1
							if np.abs(energies[it]-float(line.split()[1]))<0.00001:
								if stateiterator:
									states.append([energies[it-1],stateiterator])
									stateiterator=[]
								stateiteratorflag=True
	return states
def getUniqueExcitedStates(minweight=0.05,pathToExcitedStates="./",pathtoMO="./"):
    ## Parses the excited states from a TDDFPT calculation done in CP2K  
    ## determines the eta coefficients, with respect to the MO basis, which has 
    ## all positive phases! (see getMOsPhases for the definition of the phase)
    ## input:
    ## (opt.)   minweight               minimum amplitude to consider
    ##          Delta                   Energy scale around the first excited state to consider [eV]
    ##          pathToExcitedStates     path to the output file of the Cp2k 
    ## output:  eta                     excited states represented as a NBasis x NBasis x NStates numpy array
    ##                                  NBasis is the number of used Basis functions 
    ##                                  NStates is the number of computed excited states
    ##          Energies                Excited state energies as a np array unit [eV]
	#get the output file 
    ConFactors=ConversionFactors()
    try:
        eta=np.load("eta.npy")
        Energies=np.load("ExcitedStateEnergies.npy")
    except:
        states=readinExcitedStatesCP2K(pathToExcitedStates,minweight)
        KSEnergies=getElectronicCouplings(pathToExcitedStates)
        KSEnergies*=ConFactors["a.u.->eV"]
        MOs=readinMos(pathtoMO)
        eta=np.zeros((np.shape(MOs)[0],np.shape(MOs)[1],len(states)))
        MOsphases=getMOsPhases(pathtoMO)
        homoid=getHOMOId(pathToExcitedStates)
        Energies=[]
        for it in range(len(states)):
            exitedstateenergy=states[it][0]
            Energies.append(exitedstateenergy)
            for composition in states[it][1]:
                holeState=composition[0]+homoid
                particleState=composition[1]+homoid
                unique_weight=composition[2]*MOsphases[homoid+composition[0]]*MOsphases[homoid+composition[1]]
                eta[particleState,holeState,it]=unique_weight
        #Choose the global phase of the excited state to be positive 
        for it in range(len(states)):
                index=np.argmax(np.abs(eta[:,:,it]))
                tupel=np.unravel_index(index,(np.shape(MOs)[0],np.shape(MOs)[1]))
                eta[:,:,it]*=np.sign(eta[tupel[0],tupel[1],it])
        #Normalize the State 
        for it in range(len(states)):
            normalization=np.trace(np.transpose(eta[:,:,it])@eta[:,:,it])
            print(normalization)
            eta[:,:,it]/=np.sqrt(normalization)
        np.save(pathToExcitedStates+"/"+"eta",eta)
        np.save(pathToExcitedStates+"/"+"ExcitedStateEnergies",Energies)
    return Energies,eta

        
#########################################################################
## END Parser and Basis Construction functions 
#########################################################################

#########################################################################
## Functions for Computing Overlap and Basis-Transformation Matrices
#########################################################################
def gamma(alpha,n):
    ## computes the analytical value of the integral int_{-\inf}^{inf}x^ne^{-alphax^2}
    ## (see Manuscript gamma function)
    ## input:       alpha       gaussian exponent                   (float)
    ##              n           power of x in the integral          (int)
    ## output:      value       the actual value of the integral    (float)
    def doublefactorial(n):
        if n == -1:
            return 1
        elif n==0:
            return 1
        else:
            return n * doublefactorial(n-2)
    value=0.0
    if n%2==0:
        value=(doublefactorial(n-1)*np.sqrt(np.pi))/(2**(0.5*n)*alpha**(0.5*n+0.5))
    return value
#-------------------------------------------------------------------------
def Kcomponent(Y1k,Y2k,ik,jk,alpha):
    ## computes the analytical value of the integral int_{-\inf}^{inf}dx (x-Y1k)^ik (x-Y2k)^jk e^{-alpha x^2}
    ## for one component k
    ## (see Manuscript K-Function function)
    ## input:       alpha        gaussian exponent                  (float)
    ##              Y1k          displacement Y1k                   (float)
    ##              Y2k          displacement Y2k                   (float)
    ##              ik           power of x-Y1k in the integral     (int)
    ##              jk           power of x-Y2k in the integral     (int)
    ## output:      sum          the  value of the integral         (float)
    def binom(n,k):
        return np.math.factorial(n)/np.math.factorial(k)/np.math.factorial(n-k)
    sum=0.0
    if Y1k==0.0 or Y2k==0.0:
        sum=gamma(alpha,ik+jk)
    else:
        for o in range(ik+1):
            for p in range(jk+1):
                if ik==o and jk==p:
                    sum+=gamma(alpha,o+p)
                else:
                    sum+=gamma(alpha,o+p)*binom(ik,o)*binom(jk,p)*(-Y1k)**(ik-o)*(-Y2k)**(jk-p)
    return sum
#-------------------------------------------------------------------------
def KFunction(Y1,Y2,iis,jjs,alpha):
    # the full K function iis=(ix,iy,iz) jjs=(jx,jy,jz)
    ## Computes the analytical value of full K function:
    ##  K=\prod_{k=1}^3int_{-\inf}^{inf}dx (x-Y1k)^ik (x-Y2k)^jk e^{-alpha x^2}
    ## (see Manuscript K-Function function)
    ## input:       alpha           gaussian exponent                  (float)
    ##              Y1              displacement Y1                    (np.array)
    ##              Y2k             displacement Y2                    (np.array)
    ##              iis=(ix,iy,iz)  monomial decomposition             (np.array)
    ##              jjs=(jx,jy,jz)  monomial decomposition             (np.array)
    ## output:      output             the  value of the integral      (float)
    return np.prod([Kcomponent(Y1[it],Y2[it],iis[it],jjs[it],alpha) for it in range(len(Y1))])
#-------------------------------------------------------------------------
def JInt(X,lm1,lm2,A1,A2):
    # computes the J integral using the monomial decomposition of the 
    # solid harmonics.
    #Input: X (numpy.array) of the difference vector R1-R2
    #A1: positive numerical
    #A2: positive numerical


    ###############################################################################################################################
    ###############################################################################################################################
    #Define the cs hash map for the coefficients of the solid harmonics to homigenious monomials
    #returns the representation of a given solid harmonics (l,m) in terms of homogenious monomials
    #input:    
    #format: 
    # cs
    # is=cs[it][0] is the representation of the monomial in terms of 
    # x^is[0]y^is[1]z^is[2]
    # and cs[it][1] the corresponding prefactor (consistent with CP2K convention)
    cs={}

    cs['s']=[[[0,0,0],0.5/np.sqrt(np.pi)]]

    cs['py']=[[[0,1,0],np.sqrt(3./(4.0*np.pi))]]
    cs['pz']=[[[0,0,1],np.sqrt(3./(4.0*np.pi))]]
    cs['px']=[[[1,0,0],np.sqrt(3./(4.0*np.pi))]]

    cs['d-2']=[[[1,1,0],0.5*np.sqrt(15./np.pi)]]
    cs['d-1']=[[[0,1,1],0.5*np.sqrt(15./np.pi)]]
    cs['d0']=[[[2,0,0],-0.25*np.sqrt(5./np.pi)],[[0,2,0],-0.25*np.sqrt(5./np.pi)],[[0,0,2],0.5*np.sqrt(5./np.pi)]]
    cs['d+1']=[[[1,0,1],0.5*np.sqrt(15./np.pi)]]
    cs['d+2']=[[[2,0,0],0.25*np.sqrt(15./np.pi)],[[0,2,0],-0.25*np.sqrt(15./np.pi)]]

    cs['f-3']=[[[2,1,0],0.75*np.sqrt(35./2./np.pi)],[[0,3,0],-0.25*np.sqrt(35./2./np.pi)]]
    cs['f-2']=[[[1,1,1],0.5*np.sqrt(105./np.pi)]]
    cs['f-1']=[[[0,1,2],np.sqrt(21./2./np.pi)],[[2,1,0],-0.25*np.sqrt(21./2./np.pi)],[[0,3,0],-0.25*np.sqrt(21./2./np.pi)]]
    cs['f0']=[[[0,0,3],0.5*np.sqrt(7./np.pi)],[[2,0,1],-0.75*np.sqrt(7/np.pi)],[[0,2,1],-0.75*np.sqrt(7/np.pi)]]
    cs['f+1']=[[[1,0,2],np.sqrt(21./2./np.pi)],[[1,2,0],-0.25*np.sqrt(21./2./np.pi)],[[3,0,0],-0.25*np.sqrt(21./2./np.pi)]]
    cs['f+2']=[[[2,0,1],0.25*np.sqrt(105./np.pi)],[[0,2,1],-0.25*np.sqrt(105./np.pi)]]
    cs['f+3']=[[[3,0,0],0.25*np.sqrt(35./2./np.pi)],[[1,2,0],-0.75*np.sqrt(35./2./np.pi)]]

    cs['g-4']=[[[3,1,0],0.75*np.sqrt(35./np.pi)],[[1,3,0],-0.75*np.sqrt(35./np.pi)]] 
    cs['g-3']=[[[2,1,1],9.0*np.sqrt(35./(2*np.pi))/4.0],[[0,3,1],-0.75*np.sqrt(35./(2.*np.pi))]] 
    cs['g-2']=[[[1,1,2],18.0*np.sqrt(5./(np.pi))/4.0],[[3,1,0],-3.*np.sqrt(5./(np.pi))/4.0],[[1,3,0],-3.*np.sqrt(5./(np.pi))/4.0]] 
    cs['g-1']=[[[0,1,3],3.0*np.sqrt(5./(2*np.pi))],[[2,1,1],-9.0*np.sqrt(5./(2*np.pi))/4.0],[[0,3,1],-9.0*np.sqrt(5./(2*np.pi))/4.0]] 
    cs['g0']=[[[0,0,4],3.0*np.sqrt(1./(np.pi))/2.0],[[4,0,0],9.0*np.sqrt(1./(np.pi))/16.0],[[0,4,0],9.0*np.sqrt(1./(np.pi))/16.0],[[2,0,2],-9.0*np.sqrt(1./np.pi)/2.0],[[0,2,2],-9.0*np.sqrt(1./np.pi)/2.0],[[2,2,0],9.0*np.sqrt(1./np.pi)/8.0]]
    cs['g+1']=[[[1,0,3],3.0*np.sqrt(5./(2*np.pi))],[[1,2,1],-9.0*np.sqrt(5./(2*np.pi))/4.0],[[3,0,1],-9.0*np.sqrt(5./(2*np.pi))/4.0]]
    cs['g+2']=[[[2,0,2],18.0*np.sqrt(5./(np.pi))/8.0],[[0,2,2],-18.*np.sqrt(5./(np.pi))/8.0],[[0,4,0],3.*np.sqrt(5./(np.pi))/8.0],[[4,0,0],-3.*np.sqrt(5./(np.pi))/8.0]]
    cs['g+3']=[[[1,2,1],-9.0*np.sqrt(35./(2*np.pi))/4.0],[[3,0,1],0.75*np.sqrt(35./(2.*np.pi))]]
    cs['g+4']=[[[4,0,0],3.0*np.sqrt(35./np.pi)/16.0],[[2,2,0],-18.0*np.sqrt(35./np.pi)/16.0],[[0,4,0],3.0*np.sqrt(35./np.pi)/16.0]]
    ###############################################################################################################################
    ###############################################################################################################################
    Y1=A2*X/(A1+A2)
    Y2=-A1*X/(A1+A2)
    Z1=cs[lm1]
    Z2=cs[lm2]
    integral=np.sum([Z1[it1][1]*Z2[it2][1]*KFunction(Y1,Y2,Z1[it1][0],Z2[it2][0],A1+A2) for it1 in range(len(Z1)) for it2 in range(len(Z2))])
    return integral
#-------------------------------------------------------------------------
def IInt(R1,A1,lm1,R2,A2,lm2):
    # computes the I integral using the J integral and the Gaussian prefactor
    # solid harmonics.
    #input: 
    #R1:    (numpy.array)       position of nucleii 1
    #R2     (numpy.array)       position of nucleii 2
    #lm1:    (string='s','py','pz','px','d-2'...)           angular momentum label for phi_s1,n1,l1,m1(R_s1)
    #lm2:    (string='s','py','pz','px','d-2'...)           angular momentum label for phi_s2,n2,l2,m2(R_s2) 
    #A1:    (positive real)     exponent gaussian of function 1
    #A2:    (positive real)     exponent of gaussian of function 2
    X=R1-R2
    Jintegral=JInt(X,lm1,lm2,A1,A2)
    A12red=-A1*A2/(A1+A2)
    Exponent=A12red*np.dot(X,X)
    gaussianPrefactor=np.exp(Exponent)
    integral=gaussianPrefactor*Jintegral
    return integral
#------------------------------------------------------------------------- 
def getoverlap(R1,lm1,dalpha1,R2,lm2,dalpha2):
    #Compute overlap of two basis functions <phi_s1,n1,l1,m1(R_s1)|phi_s2,n2,l2,m2(R_s2)>
    #input: 
    #R1:    (numpy.array)                                   position of nucleii 1
    #R2     (numpy.array)                                   position of nucleii 2
    #lm1:    (string='s','py','pz','px','d-2'...)           angular momentum label for phi_s1,n1,l1,m1(R_s1) 
    #dalpha1: (list of list)    specifies the first Gaussian type of wave function 
    #lm2:    (string='s','py','pz','px','d-2'...)           angular momentum label for phi_s2,n2,l2,m2(R_s2) 
    #dalpha2: (list of list)    specifies the second Gaussian type of wave function 
    overlap=0.0
    overlap=np.sum([dalpha1[it1][1]*dalpha2[it2][1]*IInt(R1,dalpha1[it1][0],lm1,R2,dalpha2[it2][0],lm2) for it1 in range(len(dalpha1)) for it2 in range(len(dalpha2))])
    return overlap
#-------------------------------------------------------------------------
def getNeibouringCellVectors(path,neighbours=1):
    ConFactors=ConversionFactors()
    cellvectors=[np.array([0.0,0.0,0.0])]
    cell=getCellSize(path)
    if neighbours==1:
        for cellvector in cell:
            for sign in [-1,1]:
                cellvectors.append(sign*cellvector*ConFactors["A->a.u."])
    if neighbours==2:
        for cellvector in cell:
            for sign in [-1,1]:
                cellvectors.append(sign*cellvector*ConFactors["A->a.u."])
        for cellvector1 in cell:
            for cellvector2 in cell:
                if np.linalg.norm(cellvector1-cellvector2)>10**(-10):
                    for sign1 in [-1,1]:
                        for sign2 in [-1,1]:
                            cellvectors.append(sign1*cellvector1*ConFactors["A->a.u."]+sign2*cellvector2*ConFactors["A->a.u."])
    return cellvectors
#-------------------------------------------------------------------------
def getTransformationmatrix(Atoms1, Atoms2, Basis, cell_vectors=[0.0, 0.0, 0.0], pathtolib=pathtocpp_lib):
    ##Compute the overlap & transformation matrix of the Basis functions with respect to the conventional basis ordering
    ##input: Atoms1              atoms of the first index
    ##                           list of sublists. 
    ##                           Each of the sublists has five elements. 
    ##                           Sublist[0] contains the atomorder as a int.
    ##                           Sublist[1] contains the symbol of the atom.
    ##                           Sublist[2:] containst the x y z coordinates.
    ##                                       unit: Angstroem
    ##
    ##       Atoms2              Atoms of the second index
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
    ##
    ##     cellvectors (opt.)    different cell vectors to take into account in the calculation the default implements open boundary conditions
    ##output:   Overlapmatrix    The Transformation matrix as a numpy array
    
    # Load the shared library
    lib = cdll.LoadLibrary(pathtolib)
    
    # Conversion factors and other initialization
    ConFactors = ConversionFactors()

    # Initialize the python lists for Basis Set 1
    atoms_set1 = []
    positions_set1 = []
    alphas_lengths_set1 = []
    alphas_set1 = []
    contr_coef_set1 = []
    lms_set1 = []

    # Create Python lists for input (Set 1)
    for itAtom1 in range(len(Atoms1)):
        Atom_type1 = Atoms1[itAtom1][1]
        B1 = Basis[Atom_type1]
        for itBasis1 in range(len(Basis[Atom_type1])):
            atoms_set1.append(Atom_type1)
            R1 = np.array(Atoms1[itAtom1][2:]) * ConFactors['A->a.u.']  # conversion from angstroem to atomic units
            for it1 in range(len(R1)):
                positions_set1.append(R1[it1])
            state1 = B1[itBasis1]
            dalpha1 = state1[3:]
            alphas_lengths_set1.append(len(dalpha1))
            for it2 in range(len(dalpha1)):
                alphas_set1.append(dalpha1[it2][0])
                contr_coef_set1.append(dalpha1[it2][1])
            lm1 = state1[2][1:]
            lms_set1.append(lm1)

    contr_coef_lengths_set1 = alphas_lengths_set1  # Lengths of contr_coef for each basis function in Set 1

    # Initialize the python lists for Basis Set 2
    atoms_set2 = []
    positions_set2 = []
    alphas_lengths_set2 = []
    alphas_set2 = []
    contr_coef_set2 = []
    lms_set2 = []

    # Fill Python lists for input (Set 2)
    for itAtom2 in range(len(Atoms2)):
        Atom_type2 = Atoms2[itAtom2][1]
        B2 = Basis[Atom_type2]
        for itBasis2 in range(len(Basis[Atom_type2])):
            atoms_set2.append(Atom_type2)
            R2 = np.array(Atoms2[itAtom2][2:]) * ConFactors['A->a.u.']  # conversion from angstroem to atomic units
            for it1 in range(len(R2)):
                positions_set2.append(R2[it1])
            state2 = B2[itBasis2]
            dalpha2 = state2[3:]
            alphas_lengths_set2.append(len(dalpha2))
            for it2 in range(len(dalpha2)):
                alphas_set2.append(dalpha2[it2][0])
                contr_coef_set2.append(dalpha2[it2][1])
            lm2 = state2[2][1:]
            lms_set2.append(lm2)

    contr_coef_lengths_set2 = alphas_lengths_set2  # Lengths of contr_coef for each basis function in Set 1

    # Define the function signature
    get_T_Matrix = lib.get_T_Matrix
    get_T_Matrix.restype = POINTER(c_double)
    get_T_Matrix.argtypes = [POINTER(c_char_p),
                             POINTER(c_double),
                             POINTER(c_double),
                             POINTER(c_int),
                             POINTER(c_double),
                             POINTER(c_int),
                             POINTER(c_char_p),
                             c_int,
                             POINTER(c_char_p),
                             POINTER(c_double),
                             POINTER(c_double),
                             POINTER(c_int),
                             POINTER(c_double),
                             POINTER(c_int),
                             POINTER(c_char_p),
                             c_int,
                             POINTER(c_double),
                             c_int]

    freeArray = lib.freeOLPasArray_ptr
    freeArray.argtypes = [POINTER(c_double)]

    # Convert Python lists to pointers
    atoms_set1_ptr = (c_char_p * len(atoms_set1))(*[s.encode("utf-8") for s in atoms_set1])
    positions_set1_ptr = (c_double * len(positions_set1))(*positions_set1)
    alphas_set1_ptr = (c_double * len(alphas_set1))(*alphas_set1)
    alphas_lengths_set1_ptr = (c_int * len(alphas_lengths_set1))(*alphas_lengths_set1)
    contr_coef_set1_ptr = (c_double * len(contr_coef_set1))(*contr_coef_set1)
    contr_coef_lengths_set1_ptr = (c_int * len(contr_coef_lengths_set1))(*contr_coef_lengths_set1)
    lms_set1_ptr = (c_char_p * len(lms_set1))(*[s.encode("utf-8") for s in lms_set1])

    atoms_set2_ptr = (c_char_p * len(atoms_set2))(*[s.encode("utf-8") for s in atoms_set2])
    positions_set2_ptr = (c_double * len(positions_set2))(*positions_set2)
    alphas_set2_ptr = (c_double * len(alphas_set2))(*alphas_set2)
    alphas_lengths_set2_ptr = (c_int * len(alphas_lengths_set2))(*alphas_lengths_set2)
    contr_coef_set2_ptr = (c_double * len(contr_coef_set2))(*contr_coef_set2)
    contr_coef_lengths_set2_ptr = (c_int * len(contr_coef_lengths_set2))(*contr_coef_lengths_set2)
    lms_set2_ptr = (c_char_p * len(lms_set2))(*[s.encode("utf-8") for s in lms_set2])

    cell_vectors_ptr = (c_double * len(cell_vectors))(*cell_vectors)

    # Call the C++ function
    OLP_array_ptr = get_T_Matrix(atoms_set1_ptr, positions_set1_ptr, alphas_set1_ptr, alphas_lengths_set1_ptr,
                                  contr_coef_set1_ptr, contr_coef_lengths_set1_ptr, lms_set1_ptr, len(atoms_set1),
                                  atoms_set2_ptr, positions_set2_ptr, alphas_set2_ptr, alphas_lengths_set2_ptr,
                                  contr_coef_set2_ptr, contr_coef_lengths_set2_ptr, lms_set2_ptr, len(atoms_set2),
                                  cell_vectors_ptr, len(cell_vectors))

    array_data = np.ctypeslib.as_array(OLP_array_ptr, shape=(len(atoms_set1) * len(atoms_set2),))
    array_list = deepcopy(array_data)
    freeArray(OLP_array_ptr)

    OLP = np.array(array_list).reshape((len(atoms_set1), len(atoms_set2)))

    return OLP
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
    ConFactors=ConversionFactors()
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
def ComputeCenterOfMassCoordinates(coordinates,masses):
    """
    Computes the center of mass of a molecule with respect to the basis of the coordinate file.

    Parameters:
    - coordinates (Nx3 numpy.array): Coordinates of the atoms with respect to some basis (arbitrary origin).
    - masses (N numpy.array): Masses of the atoms as a numpy array, requires the same ordering as the coordinates.

    Returns:
    - centerofmasscoordinates (Nx3 numpy.array): Coordinates of the atoms with respect to some basis, origin at the center of mass.
    - centerofmass (3x1 numpy.array): Coordinates of the center of mass with respect to the old frame.

    Notes:
    - The coordinates provided have an arbitrary origin.
    - The returned centerofmasscoordinates have their origin at the center of mass.
    - The returned centerofmass is the coordinates of the center of mass with respect to the old frame.
    - The center of mass is computed based on the provided masses and coordinates.
    """
    MassofMolecule=0.0
    centerofmass=np.array([0.,0.,0.])
    #Compute the center of mass: 
    for it,coords in enumerate(coordinates):
        mass=masses[it]
        MassofMolecule+=mass
        centerofmass+=mass*np.array([coords[0],coords[1],coords[2]])
    centerofmass/=MassofMolecule
    
    centerofmasscoordinates=[]
    #get the center of mass coordinates
    for coords in coordinates:
        centerofmasscoordinates.append(coords-centerofmass)
    return centerofmasscoordinates,centerofmass
def getInertiaTensor(coordinates,masses):
    """
    Computes the inertia tensor of a molecule with respect to the basis of the coordinate file.
    Note that the inertia tensor is only a physically meaningful object if the coordinates have
    the center of mass as an origin.

    Parameters:
    - coordinates (Nx3 numpy.array): Coordinates of the atoms with respect to some basis.
    - masses (N numpy.array): Masses of the atoms as a numpy array, requires the same ordering as the coordinates.

    Returns:
    - I (3x3 numpy.array): Moment of inertia tensor of the molecule.

    Notes:
    - This function relies on the `ComputeCenterOfMassCoordinates` function.
    - The coordinates provided should have the center of mass as the origin.
    - The returned inertia tensor is a 3x3 numpy array.
    - The moments of inertia are computed along the principal axes x, y, and z.
    """

    #Compute the coordinates in the frame where the origin is the center of mass
    centerofmasscoordinates,_=ComputeCenterOfMassCoordinates(coordinates,masses)
    #get the moments of inertia tensor
    Ixx=0.0
    Iyy=0.0
    Izz=0.0
    Ixy=0.0
    Ixz=0.0
    Iyz=0.0
    for it,coords in enumerate(centerofmasscoordinates):
        Ixx+=masses[it]*(coords[1]**2+coords[2]**2)
        Iyy+=masses[it]*(coords[0]**2+coords[2]**2)
        Izz+=masses[it]*(coords[0]**2+coords[1]**2)
        Ixy+=(-1)*masses[it]*coords[0]*coords[1]
        Ixz+=(-1)*masses[it]*coords[0]*coords[2]
        Iyz+=(-1)*masses[it]*coords[1]*coords[2]
    #The moment of inertia tensor
    I=np.array([[Ixx,Ixy,Ixz],[Ixy,Iyy,Iyz],[Ixz,Iyz,Izz]])
    return I
def getCoordinatesAndMasses(pathtoxyz="./Equilibrium_Geometry/"):
    """
    Helper routine to parse coordinates and masses from an xyz file.

    Parameters:
    - pathtoxyz (str, optional): Path to the corresponding xyz file where the data is saved.

    Returns:
    - coordinates (Nx3 numpy.array): Coordinates of the atoms with respect to some basis.
    - masses (N numpy.array): Masses of the atoms as a numpy array, in the same ordering as the coordinates.
    - atomicsymbols (list): Atomic symbols corresponding to each atom in the same ordering as the coordinates.

    Notes:
    - This function relies on the `getAtomicCoordinates` function and the `StandardAtomicWeights` class.
    - The `getAtomicCoordinates` function is assumed to return a list of atomic coordinates in the xyz file format.
    - The `StandardAtomicWeights` class is assumed to provide standard atomic weights for element symbols.
    - The returned coordinates are in the format of a numpy array with shape (number of atoms, 3).
    - The returned masses are in the format of a numpy array with shape (number of atoms,).
    - The returned atomicsymbols is a list of atomic symbols in the same order as the coordinates.
    """
    SaW=StandardAtomicWeights()
    #Get the Atomic coordinates 
    AtomicCoordinates=getAtomicCoordinates(pathtoxyz)
    #Parse the Coordinates & the Masses
    coordinates=[]
    masses=[]
    atomicsymbols=[]
    for coords in AtomicCoordinates:
        mass=SaW[coords[1]]
        atomicsymbols.append(coords[1])
        masses.append(mass)
        coordinates.append(np.array([coords[2],coords[3],coords[4]]))
    return coordinates,masses,atomicsymbols
def getTransAndRotEigenvectors(pathtoEquilibriumxyz="./Equilibrium_Geometry/",rescale=True):
    """
    Computes the translational and rotational eigenvectors according to "Vibrational Analysis in Gaussian,"
    Joseph W. Ochterski (1999).
    Reference: https://gaussian.com/wp-content/uploads/dl/vib.pdf

    Version: 01.07.2023

    Parameters:
    - pathtoEquilibriumxyz (string, optional): Path to the folder of the Equilibrium_Geometry calculation.
    - rescale (bool, optional): Flag to rescale the eigenvectors based on atom masses. Default is True.

    Returns:
    - Transeigenvectors (list of np.arrays): Translational eigenvectors in the rescaled Cartesian Basis |\tilde{s,alpha}>.
    - Roteigenvectors (list of np.arrays): Rotational eigenvectors in the rescaled Cartesian Basis |\tilde{s,alpha}>.

    Notes:
    - The function relies on the `getCoordinatesAndMasses`, `getInertiaTensor`, and `ComputeCenterOfMassCoordinates` functions.
    - The rotational eigenvectors are generated based on the principle axis obtained from the inertia tensor.
    - The translational eigenvectors are generated along each Cartesian axis.
    - The generated eigenvectors can be rescaled based on atom masses if the `rescale` flag is set to True.
    - All generated eigenvectors are normalized.
    """

    coordinates,masses,_=getCoordinatesAndMasses(pathtoEquilibriumxyz)
    #Compute the Intertia Tensor
    I=getInertiaTensor(coordinates,masses)
    centerofmasscoordinates,_=ComputeCenterOfMassCoordinates(coordinates,masses)

    #Get the principle Axis
    _,principleAxis=np.linalg.eigh(I)
    numofatoms=int(len(masses))
    Roteigenvectors=[]
    factor=1.0
    for it in [0,1,2]:
        #Define the RotEigenvector
        RotEigenvector=np.zeros(3*numofatoms)
        #generate the vector X along the principle axis
        X=principleAxis[:,it]
        for s in range(numofatoms):
            rvector=np.cross(X,centerofmasscoordinates[s])
            if rescale:
                factor=np.sqrt(masses[s])
            else:
                factor=1.0
            RotEigenvector[3*s]=rvector[0]*factor
            RotEigenvector[3*s+1]=rvector[1]*factor
            RotEigenvector[3*s+2]=rvector[2]*factor
        #Normalize the generated Eigenvector
        RotEigenvector/=np.linalg.norm(RotEigenvector)
        Roteigenvectors.append(RotEigenvector)
    Transeigenvectors=[]
    for it in [0,1,2]:
        #Define the TransEigenvector
        TransEigenvector=np.zeros(3*numofatoms)
        for s in range(numofatoms):
            if rescale:
                factor=np.sqrt(masses[s])
            else:
                factor=1.0
            TransEigenvector[3*s+it]=factor
        #Normalize the generated Eigenvector
        TransEigenvector/=np.linalg.norm(TransEigenvector)
        Transeigenvectors.append(TransEigenvector)
    
    return Transeigenvectors,Roteigenvectors

#######################################################################################################
#Reads in Forces and compute the Hessian
#######################################################################################################
def getHessian(parentfolder="./",writeMolFile=True):
    '''   
    input:   
        (opt.) parentfolder:  (string) absolute/relative path, where the geometry optimized .xyz file lies 
                               in the subfolders; electronic structure at displaced geometries is found there
        (opt.) writeMoleFile: (bool)   flag indicating whether to write a Mole file with computed results

    This function reads forces from subdirectories, computes the Hessian matrix, and extracts normal modes.

    Parameters:
        - parentfolder (string, optional): Path to the directory containing the geometry-optimized .xyz file.
                                           Default is the current directory.
        - writeMolFile (bool, optional):   Flag to control writing of Mole file. Default is True.

    Returns:
        None

    Saves:
        - "Normal-Mode-Energies.npy":      NumPy array containing normal mode energies in 1/cm.
        - "normalized-Cartesian-Displacements.npy": NumPy array containing normalized Cartesian displacements
        - "Norm-Factors.npy":              NumPy array containing normalization factors.

    If writeMolFile is True, also writes a Mole file with normal mode information.

    Note: This function assumes the existence of specific subdirectories and files in the parentfolder.

    Usage:
        getHessian(parentfolder="./", writeMolFile=True)
    '''        
    global binaryloc
    ##########################################
    #Reading in nessicary Data
    ##########################################
    #get Conversionfactors & atomic masses
    ConFactor=ConversionFactors()
    atomicmasses=StandardAtomicWeights()
    #Get the number of atoms & the order from the xyz file
    xyz_files = [f for f in os.listdir(parentfolder) if f.endswith('.xyz')]
    if len(xyz_files) != 1:
        raise ValueError('InputError: There should be only one xyz file in the current directory')
    xyzfilename=xyz_files[0]
    numofatoms=0
    with open(parentfolder+"/"+xyzfilename) as g:
        lines=g.readlines()
        numofatoms=int(lines[0])
        atomorder=[]
        for line in lines[2:]:
            if len(line.split())>0:
                atom=line.split()[0]
                atomorder.append(atom)
    #The sqrtM matrix in terms of carthesian basis
    sqrtM=np.zeros((3*numofatoms,3*numofatoms))
    for it in range(3*numofatoms):
        atomnum=int(np.floor((it/3)))
        sqrtM[it][it]=1./(np.sqrt(atomicmasses[atomorder[atomnum]]))
    #Get the carthesian displacement eigenvectors (not rescaled) of Translation & Rotation:
    Transeigenvectors,Roteigenvectors=getTransAndRotEigenvectors(parentfolder+"/Equilibrium_Geometry/",False)
    #Check if Rotations should also be projected out, this applies to Molecules/Clusters
    Rotations_Projector_String=input("Project out Rotations? Molecule/Cluster [Y] Crystal [N]:")
    if Rotations_Projector_String=="Y" or Rotations_Projector_String=="y":
        Rotations_Projector_Flag=True
    elif Rotations_Projector_String=="N" or Rotations_Projector_String=="n":
        Rotations_Projector_Flag=False
    else:
        print("Have not recognized Input! Exiting function")
        exit()
    
    Orthogonalprojector=np.identity(3*numofatoms)
    
    for translation in Transeigenvectors:
        Orthogonalprojector-=np.outer(translation,translation)
    if Rotations_Projector_Flag:
        for rotation in Roteigenvectors:
            Orthogonalprojector-=np.outer(rotation,rotation)
    
    #Read in the basis vectors of the finite displacements:
    BasisVectors,delta=readinBasisVectors(parentfolder+"/")
    #Built the T matrix from b basis 
    T=np.zeros((3*numofatoms,3*numofatoms))
    for salpha in range(3*numofatoms):
        for lambd in range(3*numofatoms):
            T[salpha][lambd]=BasisVectors[lambd][salpha]
    #invert the Tmatrix:
    Tinv=np.linalg.inv(T)
    ##########################################
    #get the subdirectories
    ##########################################
    Hessian=np.zeros((3*numofatoms,3*numofatoms))
    subdirectories=[f for f in os.listdir(parentfolder) if f.startswith('vector=')]
    if len(subdirectories)!=6*numofatoms:
        raise ValueError('InputError: Number of subdirectories does not match the number of needed Ones!')
    
    partialFpartialY=np.zeros((3*numofatoms,3*numofatoms))
    for lambd in range(3*numofatoms):
        folderplus='vector='+str(lambd+1)+'sign=+'
        folderminus='vector='+str(lambd+1)+'sign=-'
        #Project out the Force components into direction of rotation and translation D.O.F.
        try:
            Fplus=np.dot(Orthogonalprojector,np.array(readinForces(parentfolder+"/"+folderplus)))
        except:
            print("Error in folder: "+parentfolder+"/"+folderplus)
            exit()
        try:
            Fminus=np.dot(Orthogonalprojector,np.array(readinForces(parentfolder+"/"+folderminus)))
        except:
            print("Error in folder: "+parentfolder+"/"+folderminus)
            exit()
        diffofforces=(Fplus-Fminus)/(2*delta)
        for s1alpha1 in range(3*numofatoms):
            partialFpartialY[s1alpha1][lambd]=diffofforces[s1alpha1]
    Hessian=-partialFpartialY@Tinv
    #Symmetrize the Hessian
    Hessian=0.5*(Hessian+np.transpose(Hessian)) 
    #built the rescaled Hessian
    rescaledHessian=sqrtM@Hessian@sqrtM 
    # transform to units 1/cm
    rescaledHessian*=(10**(3)/1.8228884842645)*(2.19474631370540E+02)**2 
    # Diagonalize the rescaled Hessian
    Lambda,Normalmodes=np.linalg.eigh(rescaledHessian)

    #Standard Values for the Translation and Rotational overlaps with Vibrations
    threshhold_string=input("Maximally allowed Weight of Translations with Numerical Normal Modes [float between 0 and 1 or std for Standard Value]:")
    if is_number(threshhold_string):
        threshhold_trans=float(threshhold_string)
    elif threshhold_string=="std":
        threshhold_trans=0.999
        threshhold_rot=0.999
    else:
        print("Have not recognized Input! Continuing with Standard values. [0.999]")
        threshhold_trans=0.999
        threshhold_rot=0.999 
    if Rotations_Projector_Flag:
        threshhold_string=input("Maximally allowed Weight of Rotations with Numerical Normal Modes [float between 0 and 1 or std for Standard Value]:")
        if is_number(threshhold_string):
            threshhold_rot=float(threshhold_string)
        elif threshhold_string=="std":
            threshhold_rot=0.999
        else:
            print("Have not recognized Input! Continuing with Standard values. [0.999]")
            threshhold_rot=0.999
    #Get the rescaled eigenvectors of Translation & Rotation:
    Transeigenvectors,Roteigenvectors=getTransAndRotEigenvectors(parentfolder+"/Equilibrium_Geometry/",True)
    Orthogonalprojector_trans=np.identity(3*numofatoms)
    for translation in Transeigenvectors:
        Orthogonalprojector_trans-=np.outer(translation,translation)
    if Rotations_Projector_Flag:
        Orthogonalprojector_rot=np.identity(3*numofatoms)
        for rotation in Roteigenvectors:
            Orthogonalprojector_rot-=np.outer(rotation,rotation)
    # in rescaled Basis
    orthogonalized_Vibrations=[]
    Vibrational_eigenvalues=[]
    translational_subspace=[]
    rotational_subspace=[]
    for it in range(3*numofatoms):
        projection_trans=np.dot(Orthogonalprojector_trans,Normalmodes[:,it])
        weight_trans=np.sqrt(np.dot(projection_trans,projection_trans))
        normofprojection_trans=np.sqrt(weight_trans)
        if Rotations_Projector_Flag:
            projection_rot=np.dot(Orthogonalprojector_rot,Normalmodes[:,it])
            weight_rot=np.sqrt(np.dot(projection_rot,projection_rot))
        else:
            weight_rot=1.0
        if weight_trans > threshhold_trans: 
            if Rotations_Projector_Flag:
                if weight_rot>threshhold_rot:
                    orthogonalized_Vibrations.append(projection_trans/normofprojection_trans)
                    Vibrational_eigenvalues.append(Lambda[it])
            else:
                orthogonalized_Vibrations.append(projection_trans/normofprojection_trans)
                Vibrational_eigenvalues.append(Lambda[it])
        elif weight_trans<=threshhold_trans:
            if Rotations_Projector_Flag:
                if weight_rot>threshhold_rot:
                    print("Mode with Frequency="+str(np.sign(Lambda[it])*np.sqrt(np.abs(Lambda[it])))+" 1/cm "+"has trans weight "+str(weight_trans))
                    translational_subspace.append(it)
                else:
                    print("Mode with Frequency="+str(np.sign(Lambda[it])*np.sqrt(np.abs(Lambda[it])))+" 1/cm "+"has trans weight "+str(weight_trans)+" and "+"has rot weight "+str(weight_rot))
            else:
                print("Mode with Frequency="+str(np.sign(Lambda[it])*np.sqrt(np.abs(Lambda[it])))+" 1/cm "+"has trans weight "+str(weight_trans))
                translational_subspace.append(it)
        elif weight_trans>threshhold_trans and weight_rot<=threshhold_rot:
            print("Mode with Frequency="+str(np.sign(Lambda[it])*np.sqrt(np.abs(Lambda[it])))+" 1/cm "+"has rot weight "+str(weight_rot))
            rotational_subspace.append(it)
    normalmodeEnergies_preliminary=np.array(Vibrational_eigenvalues)*np.sqrt(np.abs(np.array(Vibrational_eigenvalues)))/np.abs(np.array(Vibrational_eigenvalues))
    normalmodes_prelimiary=orthogonalized_Vibrations
    
    
    #Check the correct number of the identified Translational and evtl. Rotational Subspace
    if len(translational_subspace)!=3:
        print("Translational Subspace has Dimension "+str(len(translational_subspace))+"!")
        exit()
    if Rotations_Projector_Flag:
        if len(rotational_subspace)!=3:
            print("Rotational Subspace has Dimension "+str(len(translational_subspace))+"!")
            exit()
    print("Normal-Mode-Energies [1/cm]: \n",normalmodeEnergies_preliminary)
    normalmodes=[]
    normalmodeenergies=[]
    for it in range(len(normalmodeEnergies_preliminary)):
        normalmodes.append(normalmodes_prelimiary[it])
        normalmodeenergies.append(normalmodeEnergies_preliminary[it])    
        
    
    carthesianDisplacements=[]
    normfactors=[]
    for vvector in normalmodes:
        #represent the normal modes, the Transeigenvector and the RotEigenvectors in terms in cartesian components & normalize it
        vvector=sqrtM@vvector
        normfactor=np.sqrt(np.dot(vvector,vvector))
        normfactors.append(normfactor)
        carthesianDisplacements.append(vvector/normfactor)
    np.save("Normal-Mode-Energies",normalmodeenergies)
    np.save("normalized-Carthesian-Displacements",carthesianDisplacements)
    np.save("Norm-Factors",normfactors)
    if writeMolFile:
        writemolFile(normalmodeenergies,carthesianDisplacements,normfactors,parentfolder)
def CorrectNormalModeEnergies_Input(delta=0.1,parentfolder="./"):
    '''
    Perform corrections on normal mode energies and generate input files for each correction.

    Parameters:
        - delta (float, optional):        Displacement magnitude for corrections. Default is 0.1.
        - parentfolder (string):         Absolute/relative path to the directory containing input files.

    Returns:
        None

    Creates a new directory named "Correction_Calc" within the specified parentfolder.
    For each negative normal mode energy, a subdirectory is created within "Correction_Calc" 
    with input files adjusted for corrections. The subdirectories are named "Correction_Mode_x" 
    where x is the index of the corrected normal mode.

    Note: This function assumes the existence of specific files and directories in the parentfolder.

    Usage:
        CorrectNormalModeEnergies_Input(delta=0.1, parentfolder="./")
    '''
    os.mkdir(parentfolder+"/"+"Correction_Calc/")
    ConFactors=ConversionFactors()
    normalmodeenergies,normalizedcarthesiandisplacements,_=readinVibrations(parentfolder)
    vectorsToCorrect=[]
    for it,modeenergy in enumerate(normalmodeenergies):
        if modeenergy<0.0:
            vectorsToCorrect.append(it)

    for it in vectorsToCorrect:
        normalizedCartesianDisplacement=normalizedcarthesiandisplacements[it]
        inp_files = [f for f in os.listdir(parentfolder) if f.endswith('.inp')]
        if len(inp_files) != 1:
            raise ValueError('InputError: There should be only one .inp file in the current directory')
        inpfilename = inp_files[0]
        Projectname='emptyString'
        with open(inpfilename,'r') as f:
            for lines in f:
                if len(lines.split())>0:
                    if lines.split()[0]=="PROJECT":
                        Projectname=lines.split()[1]
        if Projectname=='emptyString':
            raise ValueError('InputError: Projectname not found!')
        Restart_files = [f for f in os.listdir(parentfolder) if f.endswith('-RESTART.wfn')]
        if len(Restart_files) != 1:
            raise ValueError('InputError: There should be only one Restart file in the current directory')
        Restart_filename = Restart_files[0]
        if Restart_filename!=Projectname+'-RESTART.wfn':
            raise ValueError('InputError: Project- and Restartfilename differ! Reconsider your input.')
        foldername="Correction_Mode_"+str(it)
        # Check if folder exists 
        fo = [f for f in os.listdir(parentfolder+"/"+"Correction_Calc/") if f==foldername]
        if len(fo)==0:
            os.mkdir(parentfolder+"/"+"Correction_Calc/"+foldername)
        folderpath=parentfolder+"/"+"Correction_Calc/"+"/"+foldername+"/"
        #Check which subdirectories exist
        for displacementnumber in range(1,2):
            for sign in [0,1]:
                if sign==0:
                    symbolsign='+'
                if sign==1:
                    symbolsign='-'
                name=str(displacementnumber)+"sign="+symbolsign
                fo = [f for f in os.listdir(folderpath) if f==name]
                if len(fo)==0:
                    changeConfiguration(str(displacementnumber),normalizedCartesianDisplacement,displacementnumber*delta*ConFactors['a.u.->A'],sign,parentfolder,folderpath)
                    work_dir=parentfolder+"/"+"Correction_Calc"+"/"+foldername+"/"+name+"/"
                    os.system("cp "+parentfolder+inpfilename+" "+work_dir)
                    os.system("cp "+parentfolder+Restart_filename+" "+work_dir)
                    os.system("ln -s "+pathtobinaries+"/"+"cp2k.popt"+" "+work_dir)
def CorrectNormalModeEnergies_Output(delta=0.1,path_to_original_data="./",path_to_correctiondata="./Correction_Calc/"):
    '''
    Perform corrections on normal mode energies based on the provided correction data.

    Parameters:
        - delta (float, optional):        Displacement magnitude for corrections. Default is 0.1.
        - path_to_original_data (string): Absolute/relative path to the directory containing original data.
        - path_to_correctiondata (string): Absolute/relative path to the directory containing correction data.

    Returns:
        None

    Corrects normal mode energies based on the provided correction data and updates output files.

    Note: This function assumes the existence of specific files and directories in the specified paths.

    Usage:
        CorrectNormalModeEnergies_Output(delta=0.1, path_to_original_data="./", path_to_correctiondata="./Correction_Calc/")
    '''
    VibrationalFrequencies,NormCarthesianDisplacements,normfactors=readinVibrations(path_to_original_data)

    # Iterate over correction directories
    for overdir in os.listdir(path_to_correctiondata):
        itmode=int(overdir[-1])
        numofatoms=int((len(VibrationalFrequencies)+6)/3)

        # Get translational and rotational eigenvectors
        Transeigenvectors,Roteigenvectors=getTransAndRotEigenvectors(path_to_original_data,True)
        # Calculate orthogonal projector
        Orthogonalprojector=np.identity(3*numofatoms)
        for translation in Transeigenvectors:
            Orthogonalprojector-=np.outer(translation,translation)
        for rotation in Roteigenvectors:
            Orthogonalprojector-=np.outer(rotation,rotation)
        # Calculate overlap matrix
        OLP=np.zeros((len(NormCarthesianDisplacements)-1,len(NormCarthesianDisplacements)-1))
        iterator1=0
        iterator2=0
        for it1,mode1 in enumerate(NormCarthesianDisplacements):
            for it2,mode2 in enumerate(NormCarthesianDisplacements):
                if it1!=itmode and it2!=itmode:
                    OLP[iterator1][iterator2]=np.dot(mode1,mode2)
                    iterator2+=1
            if it1!=itmode and it2!=itmode:
                iterator2=0
                iterator1+=1
        # Invert the overlap matrix
        OLPinv=np.linalg.inv(OLP)
        iterator1=0
        iterator2=0
        Projector=np.zeros((3*numofatoms,3*numofatoms))
        # Calculate projector based on inverted overlap matrix
        for it1,mode1 in enumerate(NormCarthesianDisplacements):
            for it2,mode2 in enumerate(NormCarthesianDisplacements):
                if it1!=itmode and it2!=itmode:
                    Projector+=OLPinv[iterator1][iterator2]*np.outer(mode1,mode2)
                    iterator2+=1
            if it1!=itmode and it2!=itmode:
                iterator2=0
                iterator1+=1

        subfolders_in_folders=[x[0] for x in os.walk(path_to_correctiondata+"/"+overdir)][2:]
        displacements=[]
        Forces=[]
        # Iterate over subdirectories in the correction data
        for dir in subfolders_in_folders:
            splitedline=dir.split("/")[-1]
            displacement=int(splitedline.split("sign=")[0])*delta
            sign_symbol=splitedline.split("sign=")[-1]
            sign=1
            if sign_symbol =="+":
                sign=1
            elif sign_symbol=="-":
                sign=-1
            displacement*=sign
            displacements.append(displacement)
            currentpath=dir
            # Read forces from correction data
            outfile= [f for f in os.listdir(currentpath) if f.endswith('.out')]
            if len(outfile) != 1:
                raise ValueError('InputError: There should be only one out file in the current directory')
            Force=readinForces(dir)
            Forces.append(np.dot(NormCarthesianDisplacements[itmode],np.dot(Orthogonalprojector,Force)))
        # Calculate second derivative of energy with respect to displacement
        dEbydXsquared=0.0
        for it,displacement in enumerate(displacements):
            if np.abs(displacement-delta)<10**(-12):
                prefactor=+1.0
            elif np.abs(displacement+delta)<10**(-12):
                prefactor=-1.0
            else:
                prefactor=0.0
            dEbydXsquared+=prefactor*Forces[it]*8.2387234983E-8/(5.29177210903*10**(-11))
        dEbydXsquared/=-(2*delta)
        # Calculate corrected mode energy based on corrected second derivative
        Energy=1.05457182*10**(-34)*np.sign(dEbydXsquared)*np.sqrt(np.abs(dEbydXsquared)/1.660539)*normfactors[itmode]*10**(13.5)/(1.602*10**(-22))
        Energy*=8.065610
        # Update the VibrationalFrequencies array with corrected energy
        VibrationalFrequencies[itmode]=Energy
    sortedIndices=np.argsort(VibrationalFrequencies)
    # Save the corrected data
    np.save("Normal-Mode-Energies",VibrationalFrequencies[sortedIndices])
    np.save("normalized-Carthesian-Displacements",NormCarthesianDisplacements[sortedIndices])
    np.save("Norm-Factors",normfactors[sortedIndices])
    # Write a Mole file with corrected data
    writemolFile(VibrationalFrequencies[sortedIndices],NormCarthesianDisplacements[sortedIndices],normfactors[sortedIndices],path_to_original_data)

def writemolFile(normalmodeEnergies,normalmodes,normfactors,parentfolder="./"):
    '''
    Function to generate a .mol file for use with e.g., Jmol.
    
    Parameters:
    - normalmodeEnergies (np.array): The normal mode energies as a numpy array.
    - normalmodes (np.array): Normalized Cartesian displacements, i.e., vectors proportional to the Cartesian displacements.
                              v = sqrt(M^(-1)) @ X, where X are the eigenvectors of the rescaled Hessian.
    - normfactors (np.array): The normalization factors n = sqrt(dot(v, v)).
    - parentfolder (str, optional): The parent folder where the .mol file will be created. Default is "./".

    Notes:
    - The function relies on the `ConversionFactors` class.
    - The function looks for a single .xyz file in the specified directory and reads the atomic coordinates from it.
    - The .mol file is generated with sections for frequencies, atomic coordinates, normalization factors, and vibrational modes.
    - The generated .mol file is named "Vibrations.mol" and is saved in the specified parent folder.
    '''
    ConFactor=ConversionFactors()
    xyz_files = [f for f in os.listdir(parentfolder) if f.endswith('.xyz')]
    if len(xyz_files) != 1:
        raise ValueError('InputError: There should be only one xyz file in the current directory')
    ##########################################
    #Get the number of atoms from the xyz file
    ##########################################
    xyzfilename=xyz_files[0]
    numofatoms=0
    with open(parentfolder+"/"+xyzfilename) as g:
        lines=g.readlines()
        numofatoms=int(lines[0])
        moldencoordinates=[]
        for line in lines[2:]:
            if len(line.split())>0:
                atom=line.split()[0]
                moldencoordinates.append([atom,float(line.split()[1])*ConFactor['A->a.u.'],float(line.split()[2])*ConFactor['A->a.u.'],float(line.split()[3])*ConFactor['A->a.u.']])
    with open(parentfolder+"/Vibrations.mol",'w') as f:
        f.write('[Molden Format]\n')
        f.write('[FREQ]\n')
        for Frequency in normalmodeEnergies:
            f.write('   '+str(Frequency)+'\n')
        f.write('[FR-COORD]\n')
        for atoms in moldencoordinates:
            f.write(atoms[0]+'   '+str(atoms[1])+'   '+str(atoms[2])+'   '+str(atoms[3])+'\n')
        f.write('[NORM-FACTORS]\n')
        for normfactor in normfactors:
            f.write(str(normfactor)+'\n')
        f.write('[FR-NORM-COORD]\n')
        modeiter=1
        for mode in normalmodes:
            f.write('vibration      '+str(modeiter)+'\n')
            for s in range(numofatoms):
                f.write('   '+str(round(mode[3*s], 12))+'   '+str(round(mode[3*s+1], 12))+'   '+str(round(mode[3*s+2],12))+'\n')
            modeiter+=1


def WFNonGrid(id=0,N1=100,N2=100,N3=100,parentfolder='./'):
    '''Function to represent the DFT eigenstate HOMO+id on a real space grid within the unit cell with Nx,Ny,Nz grid points
       input:   id:               (int)                   specifies the Orbital, id=0 is HOMO id=1 is LUMO id=-1 is HOMO-1 ect. 
       (opt.)   parentfolder:     (str)                   path to the .inp file of the cp2k calculation to read in the cell dimensions    
                Nx,Ny,Nz:         (int)                   Number of grid points in each direction                        
       output:  f                 (Nx x Ny x Nz np.array) Wavefunction coefficients, where first index is x, second y and third z
    '''
    def ElementaryPolynomialFunction(r,Rs,csdec):
        diff=r-Rs
        dx=diff[0,:,:,:]
        dy=diff[1,:,:,:]
        dz=diff[2,:,:,:]
        res=np.zeros(np.shape(dx))
        for item in csdec:
            res+=(item[1])*dx**(item[0][0])*dy**(item[0][1])*dz**(item[0][2])
        return res
    def getElementaryBasisFunction(r,Rs,alphas,ds,csdec):
        '''Function to evaluate a elementary CP2K atom centered basis function at position r.
           input:   r:         (3x1 np.array)               position at which the wf is evaluated
                    Rs:        (3x1 np.array)               position of the atom to which this basis function is attached
                    alphas:    (np.array)                   numpy array of the exponents  of the gaussian (same order as ds)
                    ds:        (np.array)                   contraction coefficients of the elementary gaussian 
                    csdec:     ()
        '''
        diff=r-Rs
        res=np.zeros(np.shape(diff[0,:,:,:]))
        for it in range(len(alphas)):
            res+=ds[it]*np.exp(-alphas[it]*(diff[0,:,:,:]**2+diff[1,:,:,:]**2+diff[2,:,:,:]**2))
        return ElementaryPolynomialFunction(r,Rs,csdec)*res
    def getWavefunction(x,y,z,Atoms,a,Basis,cs,voxelvolume,v1,v2,v3):
        '''Function to construct the wavefunction with coefficients stored in a and CP2K atom centered basis described by 
           Atoms, Basis, cs on a orthonormal xyz grid. The lengthx,lenghty, lenghtz parameters are the dimensions of the minimal voxel.
        '''
        Luptable={}
        Luptable['s']=0
        Luptable['p']=1
        Luptable['d']=2
        Luptable['f']=3
        Luptable['g']=4
        Luptable['h']=5
        r=np.array([x*v1[0]+y*v2[0]+z*v3[0],x*v1[1]+y*v2[1]+z*v3[1],x*v1[2]+y*v2[2]+z*v3[2]])
        it=0
        res=0.0
        for atom in Atoms:
            Rs=np.array([1.88972613288564*atom[2],1.88972613288564*atom[3],1.88972613288564*atom[4]])
            Rstensor=np.zeros(np.shape(r))
            Rstensor[0,:,:,:]=Rs[0]
            Rstensor[1,:,:,:]=Rs[1]
            Rstensor[2,:,:,:]=Rs[2]
            Atomicsymbol=atom[1]
            for Bf in Basis[Atomicsymbol]:
                Coefficients=np.array(Bf[3:])
                lm=Bf[2][1:]
                #l=Bf[2][1]
                alphas=np.array([Coefficients[it][0] for it in range(len(Coefficients))])
                ds=np.array([Coefficients[it][1] for it in range(len(Coefficients))])
                csdec=cs[lm]
                value=getElementaryBasisFunction(r,Rstensor,alphas,ds,csdec)
                #check that normalized
                normfactor=np.sqrt(voxelvolume*np.sum(value**2))
                res+=a[it]*value/normfactor
                it+=1
        return res
    Homoid=getHOMOId(parentfolder)
    KSHamiltonian,OLM=readinMatrices(parentfolder)
    Sm12=LoewdinTransformation(OLM)
    S12=sci.linalg.fractional_matrix_power(OLM, 0.5)
    try:
        MOs=np.load("MOs.npy")
        a=MOs[:,id+Homoid]
        a*=getPhaseOfMO(S12@a)
    except:
        KSHorth=np.dot(Sm12,np.dot(KSHamiltonian,Sm12))
        _,A=np.linalg.eigh(KSHorth)
        A[:,id+Homoid]*=getPhaseOfMO(A[:,id+Homoid])
        a=Sm12@A[:,id+Homoid]
    Atoms=getAtomicCoordinates(parentfolder)
    cs=getcs()
    Basis=getBasis(parentfolder)
    Cellvectors=getCellSize(parentfolder)
    #Convert to atomic units
    cellvector1=Cellvectors[0]*1.88972613288564
    cellvector2=Cellvectors[1]*1.88972613288564
    cellvector3=Cellvectors[2]*1.88972613288564
    #get voxel volume in a.u.**3
    voxelvolume=np.dot(cellvector1,np.cross(cellvector2,cellvector3))/(N1*N2*N3)
    #discretization
    v1=cellvector1/np.linalg.norm(cellvector1)
    v2=cellvector2/np.linalg.norm(cellvector2)
    v3=cellvector3/np.linalg.norm(cellvector3)
    length1=np.linalg.norm(cellvector1)/N1
    length2=np.linalg.norm(cellvector2)/N2
    length3=np.linalg.norm(cellvector3)/N3
    grid1=length1*np.arange(N1)
    grid2=length2*np.arange(N2)
    grid3=length3*np.arange(N3)
    xx,yy,zz=np.meshgrid(grid1,grid2,grid3,indexing="ij")
    f=np.zeros((N1,N2,N3))
    f=getWavefunction(xx,yy,zz,Atoms,a,Basis,cs,voxelvolume,v1,v2,v3)
    print(voxelvolume*np.sum(np.sum(np.sum(f**2))))
    f/=np.sqrt(voxelvolume*np.sum(np.sum(np.sum(f**2))))
    filename=str(id)
    np.save(parentfolder+"/"+filename,f)
    return f
def writeCubeFile(data,filename='test.cube',parentfolder='./'):
    '''Function to write a .cube file that is readable by e.g. Jmol, the origin is assumed to be [0.0,0.0,0.0]
       input:   data:             (np.array)              Nx x Ny x Nz numpy array with the wave function coefficients at each gridpoint 
       (opt.)   parentfolder:     (str)                   path to the .inp file of the cp2k calculation to read in the cell dimensions                               
       output:  f                 (np.array)              Nx x Ny x Nz numpy array where first index is x, second y and third is z
    '''
    coordinates=getAtomicCoordinates(parentfolder)
    numofatoms=len(coordinates)
    conFactors=ConversionFactors()
    Nx=np.shape(data)[0]
    Ny=np.shape(data)[1]
    Nz=np.shape(data)[2]
    origin=[0.0,0.0,0.0]
    CellSizes=getCellSize(parentfolder)
    with open(parentfolder+"/"+filename,'w') as file:
        file.write("Cube File generated with Parabola\n")
        file.write("OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n")
        file.write(format(numofatoms,'5.0f')+format(origin[0],'12.6f')+format(origin[1],'12.6f')+format(origin[2],'12.6f')+"\n")
        file.write(format(Nx,'5.0f')+format(CellSizes[0][0]*conFactors['A->a.u.']/Nx,'12.6f')+format(CellSizes[0][1]*conFactors['A->a.u.']/Nx,'12.6f')+format(CellSizes[0][2]*conFactors['A->a.u.']/Nx,'12.6f')+"\n")
        file.write(format(Ny,'5.0f')+format(CellSizes[1][0]*conFactors['A->a.u.']/Ny,'12.6f')+format(CellSizes[1][1]*conFactors['A->a.u.']/Ny,'12.6f')+format(CellSizes[1][2]*conFactors['A->a.u.']/Ny,'12.6f')+"\n")
        file.write(format(Nz,'5.0f')+format(CellSizes[2][0]*conFactors['A->a.u.']/Nz,'12.6f')+format(CellSizes[2][1]*conFactors['A->a.u.']/Nz,'12.6f')+format(CellSizes[2][2]*conFactors['A->a.u.']/Nz,'12.6f')+"\n")
        for atom in coordinates:
            file.write(format(AtomSymbolToAtomnumber(atom[1]),'5.0f')+format(0.0,'12.6f')+format(conFactors['A->a.u.']*atom[2],'12.6f')+format(conFactors['A->a.u.']*atom[3],'12.6f')+format(conFactors['A->a.u.']*atom[4],'12.6f')+"\n")
        for itx in range(Nx):
            for ity in range(Ny):
                for itz in range(Nz):
                    if (itx or ity or itz) and itz%6==0:
                        file.write("\n")
                    file.write(" {0: .5E}".format(data[itx,ity,itz]))
    return
def readCubeFile(filename,parentfolder="./"):
    '''Function to write a .cube file , the origin is assumed to be [0.0,0.0,0.0]
       input:   filename:         (str)                   name of the file 
       (opt.)   parentfolder:     (str)                   path to the .inp file of the cp2k calculation to read in the cell dimensions                               
       output:  data              (np.array)              Nx x Ny x Nz numpy array where first index is x, second y and third is z
    '''
    predata=[]
    with open(parentfolder+"/"+filename,"r") as file:
        lines=file.readlines()
        it=0
        Cubedata=False
        for line in lines:
            if it==3:
                Nx=int(line.split()[0])
            if it==4:
                Ny=int(line.split()[0])
            if it==5:
                Nz=int(line.split()[0])
            if len(line.split())==6:
                Cubedata=True
            if Cubedata:
                for da in line.split():
                    predata.append(float(da))
            it+=1
    data=np.zeros((Nx,Ny,Nz))
    it=0
    for itx in range(Nx):
        for ity in range(Ny):
            for itz in range(Nz):
                data[itx][ity][itz]=predata[it]
                it+=1
    return data
                
               

def writeXSFFile(data,filename="test.xsf",parentfolder='./'):
    '''Function to write a .xsf file that is readable by e.g. Jmol, the origin is assumed to be [0.0,0.0,0.0]
       input:   data:             (np.array)              Nx x Ny x Nz numpy array with the wave function coefficients at each gridpoint 
       (opt.)   parentfolder:     (str)                   path to the .inp file of the cp2k calculation to read in the cell dimensions                               
       output:  f                 (np.array)              Nx x Ny x Nz numpy array where first index is x, second y and third is z
    '''
    import datetime
    now = datetime.datetime.now()
    coordinates=getAtomicCoordinates(parentfolder)
    numofatoms=len(coordinates)
    conFactors=ConversionFactors()
    Nx=np.shape(data)[0]
    Ny=np.shape(data)[1]
    Nz=np.shape(data)[2]
    origin=[0.0,0.0,0.0]
    CellSizes=getCellSize(parentfolder)
    with open(parentfolder+"/"+filename,'w') as file:
        file.write("#\n")
        file.write("# .xsf file generated with Parabola\n")
        file.write("# "+now.strftime("%Y-%m-%d %H:%M:%S")+"\n")
        file.write("#\n")
        file.write("CRYSTAL\n")
        file.write("PRIMVEC\n")
        file.write(format(CellSizes[0][0],'12.6f')+format(CellSizes[0][1],'12.6f')+format(CellSizes[0][2],'12.6f')+"\n")
        file.write(format(CellSizes[1][0],'12.6f')+format(CellSizes[1][1],'12.6f')+format(CellSizes[1][2],'12.6f')+"\n")
        file.write(format(CellSizes[2][0],'12.6f')+format(CellSizes[2][1],'12.6f')+format(CellSizes[2][2],'12.6f')+"\n")
        file.write("CONVVEC\n")
        file.write(format(CellSizes[0][0],'12.6f')+format(CellSizes[0][1],'12.6f')+format(CellSizes[0][2],'12.6f')+"\n")
        file.write(format(CellSizes[1][0],'12.6f')+format(CellSizes[1][1],'12.6f')+format(CellSizes[1][2],'12.6f')+"\n")
        file.write(format(CellSizes[2][0],'12.6f')+format(CellSizes[2][1],'12.6f')+format(CellSizes[2][2],'12.6f')+"\n")
        file.write("PRIMCOORD\n")
        file.write(format(numofatoms,'5.0f')+format(1,'5.0f')+"\n")
        for atom in coordinates:
            file.write(atom[1]+" "+format(atom[2],'12.6f')+format(atom[3],'12.6f')+format(atom[4],'12.6f')+"\n")
        file.write("BEGIN_BLOCK_DATAGRID_3D\n")
        file.write("3D_field\n")
        file.write("BEGIN_DATAGRID_3D_UNKNOWN\n")
        file.write(format(Nx,'5.0f')+format(Ny,'5.0f')+format(Nz,'5.0f')+"\n")
        file.write(format(origin[0],'5.1f')+format(origin[1],'5.1f')+format(origin[2],'5.1f')+"\n")
        file.write(format(np.linalg.norm(CellSizes[0])*(Nx-1)/Nx,'12.6f')+format(0.0,'12.6f')+format(0.0,'12.6f')+"\n")
        file.write(format(0.0,'12.6f')+format(np.linalg.norm(CellSizes[1])*(Ny-1)/Ny,'12.6f')+format(0.0,'12.6f')+"\n")
        file.write(format(0.0,'12.6f')+format(0.0,'12.6f')+format(np.linalg.norm(CellSizes[2])*(Nz-1)/Nz,'12.6f')+"\n")
        for itz in range(Nz):
            for ity in range(Ny):
                for itx in range(Nx):
                    if (itx or ity or itz) and itx%6==0:
                        file.write("\n")
                    file.write(" {0: .5E}".format(data[itx,ity,itz]))
        file.write("END_DATAGRID_3D\n")
        file.write("END_BLOCK_DATAGRID_3D")

       
    
    return                    
def ComputeDipolmatrixElements(State1,State2,path="./"):
    '''Function to compute the Dipolematrixelement between two wavefunctions State1 and State2 on a real space grid
       input:   State1/2:         (np.array)              Nx x Ny x Nz numpy array with the wave function coefficients at each gridpoint 
       (opt.)   path:             (str)                   path to the .inp file of the cp2k calculation to read in the cell dimensions                                
       output:  d12               (float)                 the dipolematrixelement between the states unit e*Bohr=a.u. for x y and z direction
    '''
    #Check that State1 and State2 have the same dimensions
    if np.shape(State1)!=np.shape(State2):
        raise ValueError("Different Grid for the two states. Reconsider your input!")
    N1=np.shape(State1)[0]
    N2=np.shape(State1)[1]
    N3=np.shape(State1)[2]
    conFactors=ConversionFactors()
    Cellvectors=getCellSize(path)
    #Convert to atomic units
    cellvector1=Cellvectors[0]*1.88972613288564
    cellvector2=Cellvectors[1]*1.88972613288564
    cellvector3=Cellvectors[2]*1.88972613288564
    #get voxel volume in a.u.**3
    voxelvolume=np.dot(cellvector1,np.cross(cellvector2,cellvector3))/(N1*N2*N3)
    ##Set up real space grids
    #discretization
    length1=np.linalg.norm(cellvector1)/N1
    length2=np.linalg.norm(cellvector2)/N2
    length3=np.linalg.norm(cellvector3)/N3
    grid1=length1*np.arange(N1)
    grid2=length2*np.arange(N2)
    grid3=length3*np.arange(N3)
    xx,yy,zz=np.meshgrid(grid1,grid2,grid3,indexing="ij")
    transitiondensity=State1*State2
    dxint=np.sum(xx*transitiondensity)*voxelvolume
    dyint=np.sum(yy*transitiondensity)*voxelvolume
    dzint=np.sum(zz*transitiondensity)*voxelvolume
    return dxint,dyint,dzint
def getTransitionDipoleMomentsAnalytic(minweigth=0.05,pathtoMO="./",pathtoExcitedstates="./"):
    '''Function to generate a file, where the Dipolmatrixelements and the excited states are summarized
       input:   path              (string)                path to the folder, where the wavefunctions have been generated and where the .inp/outputfile of the 
                                                          TDDFPT calculation lies                                                
       output:                    (void)                  
    '''
    def OOInt(X,lm1,lm2,A1,A2,cs,k):
        # computes the J integral using the monomial decomposition of the 
        # solid harmonics.
        #Input: X (numpy.array) of the difference vector R1-R2
        #A1: positive numerical
        #A2: positive numerical
        Y1=A2*X/(A1+A2)
        Y2=-A1*X/(A1+A2)
        Z1=cs[lm1]
        Z2=cs[lm2]
        integral=0.0
        if k==0:
            for P1 in Z1:
                for P2 in Z2:
                    c1=P1[1]
                    c2=P2[1]
                    is1=P1[0]
                    is2=P2[0]
                    integral+=c1*c2*KFunction(Y1,Y2,(is1[0]+1,is1[1],is1[2]),is2,A1+A2)
        elif k==1:
            for P1 in Z1:
                for P2 in Z2:
                    c1=P1[1]
                    c2=P2[1]
                    is1=P1[0]
                    is2=P2[0]
                    integral+=c1*c2*KFunction(Y1,Y2,(is1[0],is1[1]+1,is1[2]),is2,A1+A2)
        elif k==2:
            for P1 in Z1:
                for P2 in Z2:
                    c1=P1[1]
                    c2=P2[1]
                    is1=P1[0]
                    is2=P2[0]
                    integral+=c1*c2*KFunction(Y1,Y2,(is1[0],is1[1],is1[2]+1),is2,A1+A2)
        return integral
    #-------------------------------------------------------------------------
    def OInt(R1,A1,lm1,R2,A2,lm2,cs,k):
        # computes the I integral using the J integral and the Gaussian prefactor
        # solid harmonics.
        #input: 
        #R1:    (numpy.array)       position of nucleii 1
        #R2     (numpy.array)       position of nucleii 2
        #lm1:    (string='s','py','pz','px','d-2'...)           angular momentum label for phi_s1,n1,l1,m1(R_s1)
        #lm2:    (string='s','py','pz','px','d-2'...)           angular momentum label for phi_s2,n2,l2,m2(R_s2) 
        #A1:    (positive real)     exponent gaussian of function 1
        #A2:    (positive real)     exponent of gaussian of function 2
        X=R1-R2
        Jintegral=OOInt(X,lm1,lm2,A1,A2,cs,k)
        A12red=-A1*A2/(A1+A2)
        Exponent=A12red*np.dot(X,X)
        gaussianPrefactor=np.exp(Exponent)
        integral=gaussianPrefactor*Jintegral
        return integral
    #------------------------------------------------------------------------- 
    def getContribution(R1,lm1,dalpha1,R2,lm2,dalpha2,cs,k):
        #Compute overlap of two basis functions <phi_s1,n1,l1,m1(R_s1)|phi_s2,n2,l2,m2(R_s2)>
        #input: 
        #R1:    (numpy.array)                                   position of nucleii 1
        #R2     (numpy.array)                                   position of nucleii 2
        #lm1:    (string='s','py','pz','px','d-2'...)           angular momentum label for phi_s1,n1,l1,m1(R_s1) 
        #dalpha1: (list of list)    specifies the first Gaussian type of wave function 
        #lm2:    (string='s','py','pz','px','d-2'...)           angular momentum label for phi_s2,n2,l2,m2(R_s2) 
        #dalpha2: (list of list)    specifies the second Gaussian type of wave function 
        overlap=0.0
        for obj1 in dalpha1:
            for obj2 in dalpha2:
                d1=obj1[1]
                alpha1=obj1[0]
                d2=obj2[1]
                alpha2=obj2[0]
                overlap+=d1*d2*OInt(R1,alpha1,lm1,R2,alpha2,lm2,cs,k)
        return overlap
    # Readin the excited states 
    conFactors=ConversionFactors()
    energies,eta=getUniqueExcitedStates(minweigth,pathtoExcitedstates,pathtoMO)
    id_homo=getHOMOId(pathtoExcitedstates)
    ### Generate the Overlap Contribution 
    KSHamiltonian,OLM=readinMatrices(pathtoMO)
    Sm12=LoewdinTransformation(OLM)
    S12=sci.linalg.fractional_matrix_power(OLM, 0.5)
    try:
        a=np.load("MOs.npy")
        #Fix the Phase
        for it in range(np.shape(A)[1]):
            a[:,it]*=getPhaseOfMO(S12@a[:,it])
        A=S12@a
    except:
        KSHorth=np.dot(Sm12,np.dot(KSHamiltonian,Sm12))
        _,A=np.linalg.eigh(KSHorth)
        #Fix the Phase
        for it in range(np.shape(A)[1]):
            A[:,it]*=getPhaseOfMO(A[:,it])
    Atoms=getAtomicCoordinates(pathtoExcitedstates)
    Basis=getBasis(pathtoExcitedstates)
    cs=getcs
    Rx=np.zeros(np.shape(Sm12))
    Ry=np.zeros(np.shape(Sm12))
    Rz=np.zeros(np.shape(Sm12))
    it=0
    for itAtom in range(len(Atoms)):
        Atom_type1=Atoms[itAtom][1]
        for itBasis1 in range(len(Basis[Atom_type1])):
            R1=np.array(Atoms[itAtom][2:])*conFactors['A->a.u.']
            Rx[it][it]=R1[0]
            Ry[it][it]=R1[1]
            Rz[it][it]=R1[2]
            it+=1
    overlapcontribution_x=Sm12@Rx@S12
    overlapcontribution_y=Sm12@Ry@S12
    overlapcontribution_z=Sm12@Rz@S12
    dx=np.zeros(np.shape(Sm12))
    dy=np.zeros(np.shape(Sm12))
    dz=np.zeros(np.shape(Sm12))
    it1=0
    it2=0
    for itAtom1 in range(len(Atoms)):
        Atom_type1=Atoms[itAtom1][1]
        B1=Basis[Atom_type1]
        for itBasis1 in range(len(Basis[Atom_type1])):
            R1=np.array(Atoms[itAtom1][2:])*conFactors['A->a.u.'] #conversion from angstroem to atomic units
            state1=B1[itBasis1]
            dalpha1=state1[3:]
            lm1=state1[2][1:]
            for itAtom2 in range(len(Atoms)):
                Atom_type2=Atoms[itAtom2][1]
                B2=Basis[Atom_type2]
                for itBasis2 in range(len(Basis[Atom_type2])):
                    #get the position of the Atoms
                    R2=np.array(Atoms[itAtom2][2:])*conFactors['A->a.u.'] #conversion from angstroem to atomic units
                    state2=B2[itBasis2]
                    dalpha2=state2[3:]
                    lm2=state2[2][1:]
                    dx[it1][it2]=getContribution(R1,lm1,dalpha1,R2,lm2,dalpha2,cs,0)
                    dy[it1][it2]=getContribution(R1,lm1,dalpha1,R2,lm2,dalpha2,cs,1)
                    dz[it1][it2]=getContribution(R1,lm1,dalpha1,R2,lm2,dalpha2,cs,2)
                    it2+=1
            it1+=1
            it2=0
    dx=Sm12@dx@Sm12
    dy=Sm12@dy@Sm12
    dz=Sm12@dz@Sm12
    DipoleOperator_x=dx+overlapcontribution_x
    DipoleOperator_y=dy+overlapcontribution_y
    DipoleOperator_z=dz+overlapcontribution_z
    TransitionDipolevectors=[]
    with open("ExcitedStatesAndDipoles.dat","a") as file:
        file.write("Python Convention of state labeling!\n")
    for it in range(len(energies)):
        i_index,m_index=np.where(np.abs(eta[:,:,it]>minweigth))
        #generate the T matrix 

        Energy=energies[it]
        dx=0.0
        dy=0.0
        dz=0.0
        for id,i in enumerate(i_index):
            m=m_index[id]
            amplitude=eta[i,m,it]
            StateLabel1=m
            StateLabel2=i
            edx=-np.dot(A[:,StateLabel2],DipoleOperator_x@A[:,StateLabel1])
            edy=-np.dot(A[:,StateLabel2],DipoleOperator_y@A[:,StateLabel1])
            edz=-np.dot(A[:,StateLabel2],DipoleOperator_z@A[:,StateLabel1])
            dx+=amplitude*edx
            dy+=amplitude*edy
            dz+=amplitude*edz
        TransitionDipolevectors.append(np.array([dx,dy,dz])) 
        with open("ExcitedStatesAndDipoles.dat","a") as file:
            file.write("Excited State #:"+str(it+1)+"\n")
            file.write("Energy [eV]:"+format(Energy,'12.6f')+"\n")
            file.write("Dipolematrixelements (x,y,z) [eBohr]:"+format(dx,'12.6f')+format(dy,'12.6f')+format(dz,'12.6f')+"\n")
            file.write("Dipole strength**2:"+format((dx**2+dy**2+dz**2),'12.6f')+"\n")
            file.write("Oszillator strength: "+format(Energy/(3*conFactors["a.u.->eV"]/2)*(dx**2+dy**2+dz**2),'12.6f')+"\n")
            sorted_m_index=[]
            sorted_i_index=[]
            eta_prime=cp.deepcopy(eta)
            for id in enumerate(i_index):
                imax,mmax=np.unravel_index(np.argmax(np.abs(eta_prime[:,:,it])),np.shape(eta_prime[:,:,it]))
                sorted_i_index.append(imax)
                sorted_m_index.append(mmax)
                eta_prime[imax,mmax,it]=0.0
            it1=0
            for id2,i2 in enumerate(sorted_i_index):
                m2=sorted_m_index[id2]
                amplitude2=eta[i2,m2,it]
                if it1==0:
                    file.write("Dominant state transition:"+"HOMO-"+str(int(np.abs(m2-id_homo)))+"->"+"LUMO+"+str(i2-id_homo-1)+"\n")
                    file.write("Excited State is composed of the individual particle-hole excitations:\n") 
                if np.abs(amplitude2)>0.1:
                    file.write(format(m2-id_homo,'3.0f')+" ->"+format(i2-id_homo,'3.0f')+":"+format(amplitude2,'12.6f')+"\n")
                it1+=1
    np.save("Transitiondipolevectors",TransitionDipolevectors)
def getTransitionDipoleMomentsNumerical(minweigth=0.05,Nx=100,Ny=100,Nz=100,pathtoExcitedstates="./",pathtoMO="./"):
    '''Function to generate a file, where the Dipolmatrixelements and the excited states are summarized
       input:   path              (string)                path to the folder, where the wavefunctions have been generated and where the .inp/outputfile of the 
                                                          TDDFPT calculation lies                                                
       output:                    (void)                  
    '''
    # Readin the excited states 
    conFactors=ConversionFactors()
    energies=[]
    energies,eta=getUniqueExcitedStates(minweigth,pathtoExcitedstates,pathtoMO)
    id_homo=getHOMOId(pathtoExcitedstates)
    TransitionDipolevectors=[]
    with open("ExcitedStatesAndDipoles.dat","a") as file:
        file.write("Python Convention of state labeling!\n")
    for it in range(len(energies)):
        i_index,m_index=np.where(np.abs(eta[:,:,it]>minweigth))
        Energy=energies[it]
        dx=0.0
        dy=0.0
        dz=0.0
        for id,i in enumerate(i_index):
            m=m_index[id]
            amplitude=eta[i,m,it]
            StateLabel1=m-id_homo
            StateLabel2=i-id_homo
            try:
                State1=np.load(str(StateLabel1)+".npy")    
            except:
                print("Have not found state #: "+str(StateLabel1)+"\n")
                print("Creating it from scratch. This may take a while!\n")
                State1=WFNonGrid(StateLabel1,Nx,Ny,Nz,pathtoMO)
            try:
                State2=np.load(str(StateLabel2)+".npy")
            except:
                print("Have not found state #: "+str(StateLabel2)+"\n")
                print("Creating it from scratch. This may take a while!\n")
                State2=WFNonGrid(StateLabel2,Nx,Ny,Nz,pathtoMO)
            edx,edy,edz=ComputeDipolmatrixElements(State2,State1,pathtoMO)
            dx+=-amplitude*edx
            dy+=-amplitude*edy
            dz+=-amplitude*edz
        TransitionDipolevectors.append(np.array([dx,dy,dz]))
        with open("ExcitedStatesAndDipoles.dat","a") as file:
            file.write("Excited State #:"+str(it+1)+"\n")
            file.write("Energy [eV]:"+format(Energy,'12.6f')+"\n")
            file.write("Dipolematrixelements (x,y,z) [eBohr]:"+format(dx,'12.6f')+format(dy,'12.6f')+format(dz,'12.6f')+"\n")
            file.write("Dipole strength**2:"+format((dx**2+dy**2+dz**2),'12.6f')+"\n")
            file.write("Oszillator strength: "+format(Energy/(3*conFactors["a.u.->eV"]/2)*(dx**2+dy**2+dz**2),'12.6f')+"\n")
    np.save("Transitiondipolevectors",TransitionDipolevectors)
    return TransitionDipolevectors



def getHOMOId(parentfolder):
    ##   Function to get the index of the HOMO orbital, if energyeigenvalues 0,1,2,...Homoit are ordered acendingly
    ##   input:   parentfolder:         (string)            absolute/relative path, where the geometry optimized .xyz file lies 
    ##                                                      in the subfolders there we find the electronic structure at displaced geometries                      
    ##   output:  HOMOit                (int)               the index of the HOMO orbital (python convention)
    Atoms=getAtomicCoordinates(parentfolder)
    inp_files = [f for f in os.listdir(parentfolder) if f.endswith('.inp')]
    if len(inp_files) != 1:
        raise ValueError('InputError: There should be only one .inp file in the current directory')
    filename=inp_files[0]
    #Calculate the HOMO 
    NumberOfElectrons={}
    Charge=0
    with open(parentfolder+"/"+filename,'r') as f:
        lines=f.readlines()
        for line in lines:
            if len(line.split())>=2:
                if line.split()[0]=="CHARGE":
                    Charge=int(line.split()[1])
                if line.split()[0]=="&KIND":
                    atomtype= line.split()[1]
                if line.split()[0]=="POTENTIAL":
                    PotentialName=line.split()[1]
                    splitedPotentialName=PotentialName.split('-')
                    if splitedPotentialName[0]=="GTH" or splitedPotentialName[1]=="GTH":
                        numstring=splitedPotentialName[-1]
                        numofE=int(numstring[1:])
                        NumberOfElectrons[atomtype]=numofE
                    else:
                        ValueError("Yet only GTH Potentials implemented")
    numofE=0
    for atom in Atoms:
        atomsymbol=atom[1]
        numofE+=NumberOfElectrons[atomsymbol]
    numofE+=Charge
    remainder=numofE%2
    iter=np.floor(numofE/2)
    HOMOit=iter-1+remainder
    return int(HOMOit)
def getManyBodyCouplings(eta,LCC,id_homo):
    ##   Function to obtain the Many-Body Coupling Constants from the CP2K TDDFT excited states and the DFT Coupling constants
    ##   input:   eta:         (np.array)            numpy array which encodes the states
    ##                                                  required structure: states[n] encodes the n th excited state
    ##                                                                      states[n][0] is its energy
    ##                                                                      states[n][1] is a list of lists, where each list in this list contains 
    ##                                                                      list[0] hole index list[1] particle index and list[2] the weight of this 
    ##                                                                      particle hole state
    ##                                                                      second index first component
    ##            couplingConstants (np.array)          the DFT coupling constants as outputted by "getLinearCouplingConstants"
    ##                                                                        
    ##            HOMOit                (int)           the index of the HOMO orbital (python convention)
    #generate g matrix, h matrix and k matrices
    g=LCC[:,id_homo+1:,id_homo+1:]
    h=LCC[:,:id_homo+1,:id_homo+1]*(-1) # minus 1 due to fermionic commutator!
    k=LCC[:,:id_homo+1,id_homo+1:]
    #get the number of excited States to take into account
    Num_OfExciteStates=np.shape(eta)[-1]
    Num_OfModes=np.shape(LCC)[0]
    #Normalize the eta
    for p in progressbar(range(Num_OfExciteStates),"Normalizing States:",40):
        eta[:,:,p]/=np.trace(np.transpose(eta[:,:,p])@eta[:,:,p])
    K=np.zeros((Num_OfModes,Num_OfExciteStates)) #Coupling of excited state to ground state
    for m in progressbar(range(Num_OfExciteStates),"Computing Coupling to Ground State:",40):
        etap=eta[id_homo+1:,:id_homo+1,m] #First index electrons second hole
        for lamb in range(Num_OfModes):
            klamb=k[lamb,:,:]
            K[lamb,m]=np.trace(klamb@etap)
    H=np.zeros((Num_OfModes,Num_OfExciteStates,Num_OfExciteStates)) #Coupling between the excited states
    for p in progressbar(range(Num_OfExciteStates),"Computing Coupling between excited States:",40):
        for q in range(p,Num_OfExciteStates):
            etap=eta[id_homo+1:,:id_homo+1,p] #First index electrons second hole
            etaq=eta[id_homo+1:,:id_homo+1,q]
            for lamb in range(Num_OfModes):
                glamb=g[lamb,:,:]
                hlamb=h[lamb,:,:]
                H[lamb,q,p]=np.trace(np.transpose(etaq)@glamb@etap)+np.trace(etaq@hlamb@np.transpose(etap))
                H[lamb,p,q]=H[lamb,q,p]
    np.save("H_CouplingConstants",H)
    np.save("K_CouplingConstants",K)
    return H,K

def LoewdinTransformation(S,algorithm='Schur-Pade'):
    ##  Function to compute S^(-0.5) for the Loewdin orthogonalization
    ##   input:   S         (numpy array)            the overlapmatrix 
    ##
    ##   output:  Sm12      (numpy array)            the overlapmatrix to the power 1/2                            
    if algorithm=="Diagonalization":
        e,U=np.linalg.eigh(S)
        Sm12=np.diag(e**(-0.5))
        Sm12=np.dot(U,np.dot(Sm12,np.transpose(np.conjugate(U))))
    elif algorithm=="Schur-Pade":
        Sm12=sci.linalg.fractional_matrix_power(S, -0.5)
    else:
        ValueError("Algorithm not recognized! Currently available 'Schur-Pade' and 'Diagonalization'")
    return Sm12
def getElectronicCouplings(parentfolder="./"):
    ##  Function to compute the electronic energies from the equilibrium file
    ##   input:   parentfolder:         (string)            absolute/relative path, where the geometry optimized .xyz file lies 
    ##                                                      in the subfolders there we find the electronic structure at displaced geometries                         
    try:
        E=np.load(parentfolder+"/KS-Eigenvalues.npy")
    except:
        KSHamiltonian,OLM=readinMatrices(parentfolder)
        Sm12=LoewdinTransformation(OLM)
        KSHorth=np.dot(Sm12,np.dot(KSHamiltonian,Sm12))
        E,_=np.linalg.eigh(KSHorth)
        np.save(parentfolder+"/KS-Eigenvalues",E)
    return E

#######################################################################################################
#Function to Compute the local Coupling constants g 
#######################################################################################################
def getLinearCouplingConstants(parentfolder="./"):
    ''' input:   parentfolder:         (string)            absolute/relative path, where the geometry optimized .xyz file lies 
                                                          in the subfolders there we find the electronic structure at displaced geometries        
                    
        (opt.)  spread:               (int)               compute coupling elements for orbitals HOMO-spread,HOMO-spread+1,...,LUMO+spread
    
                cleandifferentSigns   (bool)              if the coupling constants have different signs for the plus displacement and the
                                                          clean them from the file 
                                                          
       output: saves 
    '''
    ConFactors=ConversionFactors()
    #Get the .xyz file
    xyz_files = [f for f in os.listdir(parentfolder+"/"+'Equilibrium_Geometry') if f.endswith('.xyz')]
    if len(xyz_files) != 1:
        raise ValueError('InputError: There should be only one xyz file in the directory:'+parentfolder+"/"+'Equilibrium_Geometry/')


    #----------------------------------------------------------------------
    # Equilibrium configuration
    #----------------------------------------------------------------------

    #get the Equilibrium Configuration 
    Atoms_Eq=getAtomicCoordinates(parentfolder+"/Equilibrium_Geometry/")
    #Construct Basis of the Equilibrium configuration
    Basis_Eq=getBasis(parentfolder+"/Equilibrium_Geometry/")
    #Read in the KS Hamiltonian
    KSHamiltonian_Eq,S_Eq=readinMatrices(parentfolder+"/Equilibrium_Geometry/")
    #perform a Loewdin Orthogonalization
    Sm12_Eq=LoewdinTransformation(S_Eq)
    KSHorth_Eq=np.transpose(Sm12_Eq)@KSHamiltonian_Eq@Sm12_Eq
    #Diagonalize to the KS Hamiltonian in the ortonormal Basis
    E_Eq,a_orth_Eq=np.linalg.eigh(KSHorth_Eq)
    #get the normalized Eigenstates in the non-orthorgonal Basis & fix Phase
    orthorgonalEigenstates_Eq=[]
    for it in range(len(E_Eq)):
        orth_eigenstate=a_orth_Eq[:,it]
        orth_eigenstate*=getPhaseOfMO(orth_eigenstate)
        orthorgonalEigenstates_Eq.append(orth_eigenstate)
    _,delta=readinBasisVectors(parentfolder)
    #get the normal modes from the cartesian displacements
    VibrationalFrequencies,_,normfactors=readinVibrations(parentfolder)
    #Multiply by Tinv to get partialY_mu/partialX_lambda
    #This has index convention lambda,mu
    #*ConFactors['E_H/a_0*hbar/sqrt(2*m_H)->cm^(3/2)']/(VibrationalFrequencies[it])**(1.5) for it in range(len(M_salpha_timesX_salpha_lambda))
    couplingConstants=np.zeros((len(normfactors),np.shape(E_Eq)[0],np.shape(E_Eq)[0]))
    for mu in progressbar(range(len(normfactors)),"Coupling Constants:",40):
        #----------------------------------------------------------------------
        # Positively displaced 
        #----------------------------------------------------------------------
        folderplus='vector='+str(mu+1)+'sign=+'
        #Read in the KS Hamiltonian and the overlap matrix
        KSHamiltonian_Plus,OLM_Plus=readinMatrices(parentfolder+"/"+folderplus+'/')
        #Get the stompositions for the positively displaced atoms
        Atoms_Plus=getAtomicCoordinates(parentfolder+"/"+folderplus)
        Sm12_Plus=LoewdinTransformation(OLM_Plus)
        KSHorth_P=np.dot(Sm12_Plus,np.dot(KSHamiltonian_Plus,Sm12_Plus))
        EPlus,a_orth_Plus=np.linalg.eigh(KSHorth_P)
        T_Eq_Plus=getTransformationmatrix(Atoms_Eq,Atoms_Plus,Basis_Eq)
        Tmatrix_Plus=Sm12_Eq@T_Eq_Plus@Sm12_Plus
        folderminus='vector='+str(mu+1)+'sign=-'
        #Read in the KS Hamiltonian and the overlap matrix
        KSHamiltonian_Minus,OLM_Minus=readinMatrices(parentfolder+"/"+folderminus+'/')
        #Get the atom positions for the negatively displaced atoms
        Atoms_Minus=getAtomicCoordinates(parentfolder+"/"+folderminus)
        #perform a Loewdin Orthogonalization
        Sm12_Minus=LoewdinTransformation(OLM_Minus)
        KSHorth_Minus=np.dot(Sm12_Minus,np.dot(KSHamiltonian_Minus,Sm12_Minus))
        #Diagonalize the KS Hamiltonian in the orthorgonal Basis
        EMinus,a_orth_Minus=np.linalg.eigh(KSHorth_Minus)
        T_Eq_Minus=getTransformationmatrix(Atoms_Eq,Atoms_Minus,Basis_Eq)
        T_Matrix_Minus=Sm12_Eq@T_Eq_Minus@Sm12_Minus
        #get the Eigenstates in the non-orthorgonal Basis
        orthorgonalEigenstates_Plus=[]
        for it in range(len(EPlus)):
            orth_eigenstate=a_orth_Plus[:,it]
            orth_eigenstate*=getPhaseOfMO(orth_eigenstate)
            orthorgonalEigenstates_Plus.append(orth_eigenstate)
        #get the Eigenstates in the non-orthorgonal Basis
        orthorgonalEigenstates_Minus=[]
        for it in range(len(EMinus)):
            orth_eigenstate=a_orth_Minus[:,it]
            orth_eigenstate*=getPhaseOfMO(orth_eigenstate)
            orthorgonalEigenstates_Minus.append(orth_eigenstate)
        adibaticallyConnectediters_Plus=[]
        #Get the adiabtically connected eigenvalues/states
        for it0 in range(len(E_Eq)):
            maximumAbsOverlap=0.0
            maximumOverlap=0.0
            iter1=-1
            for it1 in range(len(E_Eq)):
                overlap=np.dot(orthorgonalEigenstates_Eq[it0],Tmatrix_Plus@orthorgonalEigenstates_Plus[it1])
                absoverlap=np.abs(overlap)
                if absoverlap>maximumAbsOverlap:
                    iter1=it1
                    maximumOverlap=overlap
                    maximumAbsOverlap=absoverlap
            adibaticallyConnectediters_Plus.append(iter1)
            if maximumOverlap<0:
                orthorgonalEigenstates_Plus[iter1]*=(-1.0)
            if maximumAbsOverlap<0.5:
                ValueError("Maximum Overlap small! Check your inputs!")
        #Check that each iterator is exactly once in the adibaticallyConnectediters_Plus set
        for it in range(len(E_Eq)):
            if adibaticallyConnectediters_Plus.count(it)!=1:
                ValueError("Some eigenstates appear more then once as maximum weight states! Check your inputs!")
        #Get the adiabtically connected eigenvalues/states for the negative displacement
        adibaticallyConnectediters_Minus=[]
        for it0 in range(len(E_Eq)):
            maximumAbsOverlap=0.0
            maxOverlap=0.0
            iter1=-1
            for it1 in range(len(E_Eq)):
                overlap=np.dot(orthorgonalEigenstates_Eq[it0],T_Matrix_Minus@orthorgonalEigenstates_Minus[it1])
                absoverlap=np.abs(overlap)
                if absoverlap>maximumAbsOverlap:
                    maxOverlap=overlap
                    iter1=it1
                    maximumAbsOverlap=absoverlap
            adibaticallyConnectediters_Minus.append(iter1)
            if maxOverlap<0:
                orthorgonalEigenstates_Minus[iter1]*=(-1.0)
            if maximumAbsOverlap<0.5:
                ValueError("Maximum Overlap small! Check your inputs!")
        #Check that each iterator is exactly once in the adibaticallyConnectediters set
        for it in range(len(E_Eq)):
            if adibaticallyConnectediters_Minus.count(it)!=1:
                ValueError("Some eigenstates appear more then once as maximum weight states! Check your inputs!")
        
        for it0 in range(len(E_Eq)):
            for it1 in range(len(E_Eq)):
                if it0==it1:
                    deltaE=(EPlus[adibaticallyConnectediters_Plus[it0]]-EMinus[adibaticallyConnectediters_Minus[it1]])/(2*delta)*normfactors[mu]
                    couplingConstants[mu,it0,it1]=ConFactors['E_H/a_0*hbar/sqrt(2*m_H)->cm^(3/2)']/(VibrationalFrequencies[mu])**(1.5)*deltaE
                else:
                    overlap1=np.dot(orthorgonalEigenstates_Eq[it0],Tmatrix_Plus@orthorgonalEigenstates_Plus[adibaticallyConnectediters_Plus[it1]])
                    overlap2=np.dot(orthorgonalEigenstates_Eq[it0],Tmatrix_Plus@orthorgonalEigenstates_Minus[adibaticallyConnectediters_Minus[it1]])
                    deltaE=(E_Eq[it1]-E_Eq[it0])*(overlap1-overlap2)/(2*delta)*normfactors[mu]
                    couplingConstants[mu,it0,it1]=ConFactors['E_H/a_0*hbar/sqrt(2*m_H)->cm^(3/2)']/(VibrationalFrequencies[mu])**(1.5)*deltaE
    np.save("Linear_Coupling_Constants",couplingConstants)

def Parametrize():
    _,_,_=getHessian()
    _=getElectronicCouplings("./Equilibrium_Geometry/")
    getLinearCouplingConstants()

def represents_int(s):
    try: 
        int(s)
    except ValueError:
        return False
    else:
        return True

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
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



