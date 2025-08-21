from . import Geometry
import numpy as np
from .PhysConst import ConversionFactors,StandardAtomicWeights
from . import Read
from . import Write
from . import Util
from . import Symmetry
import os
#get the environmental variable 
pathtocp2k=os.environ["cp2kpath"]
pathtobinaries=pathtocp2k+"/exe/local/"
def Vib_Ana_inputs(deltas=[],vectors=[],parentpath="./",linktobinary=True,binary="cp2k.popt",binaryloc=pathtobinaries):
    deltasflag=False
    if len(deltas)==0:
        deltasflag=True
        print("Using Standardvalue of 0.05 a_0 for Displacements!")
    cartesianflag=False
    if len(vectors)==0:
        cartesianflag=True
    #get the Projectname: 
    inp_files = [f for f in os.listdir(parentpath) if f.endswith('.inp')]
    if len(inp_files) != 1:
        raise ValueError('InputError: There should be only one .inp file in the current directory')
    inpfilename = inp_files[0]
    CheckinpfileforVib_Ana(parentpath)
    Projectname='emptyString'
    with open(parentpath+"/"+inpfilename,'r') as f:
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
    RestartfileFlag=True
    if len(Restart_files) ==0:
        print("Warning: No Restart Files detected!")
        RestartfileFlag=False
    elif len(Restart_files) >1:
        raise ValueError('InputError: More than one Restartfile detected!')
    else:
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
        if len(vectors)!=3*numberofatoms:
            print('Warning: Not enough vectors given for the Normal Mode Analysis!')
        if np.linalg.matrix_rank(vectors)!=len(vectors):
            #do a second check based on SVD 
            print('Warning: The set of vectors given do not form a basis!')
    if deltasflag:
        deltas=0.05*np.ones(3*numberofatoms)
    with open(parentpath+"BasisHessian","w+") as g:
        if cartesianflag:
            g.write("delta="+str(deltas[0])+"\n")
        else:
            g.write("delta="+str(deltas[0]))
            for it in range(1,len(deltas)-1):
                g.write(","+str(deltas[it]))
            g.write(","+str(deltas[-1])+"\n")

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
    if RestartfileFlag:
        os.system("cp "+parentpath+Restart_filename+" "+parentpath+"Equilibrium_Geometry")
    if linktobinary:
        os.system("ln -s "+binaryloc+"/"+binary+" "+parentpath+"Equilibrium_Geometry"+"/")
    for it in range(len(vectors)):
        folderlabel='vector='+str(it+1)
        for sign in [0,1]:
            if sign==0:
                symbolsign='+'
            if sign==1:
                symbolsign='-'
            work_dir=folderlabel+"sign="+symbolsign
            vec=vectors[it]
            if cartesianflag:
                delta=deltas[0]
            else:
                delta=deltas[it]
            Geometry.changeConfiguration(folderlabel=folderlabel,vector=vec,delta=delta*ConversionFactors['a.u.->A'],sign=sign,path_xyz=parentpath,path_to=parentpath)
            os.system("cp "+parentpath+inpfilename+" "+parentpath+work_dir)
            if RestartfileFlag:
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
    with open(parentpath+"/"+inpfile,'r') as f:
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


def deflectAlongModes(parentfolder="./"):
    def unit_prefactor(omega):
        return 0.5/np.sqrt(omega)
    trans_vec=np.load(parentfolder+"/Translation_Eigenvectors.npy")
    rot_vec=np.load(parentfolder+"/Rotational_Eigenvectors.npy")
    nCD=np.load(parentfolder+"/normalized-Carthesian-Displacements.npy")
    normfactors=np.load(parentfolder+"/Norm-Factors.npy")
    omegas=np.load(parentfolder+"/Normal-Mode-Energies.npy")
    vectors=[]
    deltas=[]
    vectors.append(trans_vec[0,:]);vectors.append(trans_vec[1,:]);vectors.append(trans_vec[2,:])
    deltas.append(0.1);deltas.append(0.1);deltas.append(0.1)
    if not Read.read_periodicity(parentfolder):
        vectors.append(rot_vec[0,:]);vectors.append(rot_vec[1,:]);vectors.append(rot_vec[2,:])
        deltas.append(0.1);deltas.append(0.1);deltas.append(0.1)
    for it in range(np.shape(nCD)[0]):
        if omegas[it]>15:
            delta=np.max([unit_prefactor(omegas[it])/normfactors[it],0.05])
        else:
            delta=np.max([unit_prefactor(15)/normfactors[it],0.05])
        deltas.append(np.round(delta,2))
        vectors.append(nCD[it,:])
    Vib_Ana_inputs(deltas,vectors,parentfolder)




def getTransAndRotEigenvectors(coordinates,masses):
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
    - The function relies on the 'readCoordinatesAndMasses`, `getInertiaTensor`, and `ComputeCenterOfMassCoordinates` functions.
    - The rotational eigenvectors are generated based on the principle axis obtained from the inertia tensor.
    - The translational eigenvectors are generated along each Cartesian axis.
    - The generated eigenvectors can be rescaled based on atom masses if the `rescale` flag is set to True.
    - All generated eigenvectors are normalized.
    """

    #Compute the Intertia Tensor
    I=Geometry.getInertiaTensor(coordinates,masses)
    centerofmasscoordinates,_=Geometry.ComputeCenterOfMassCoordinates(coordinates,masses)

    #Get the principle Axis
    _,principleAxis=np.linalg.eigh(I)
    numofatoms=int(len(masses))
    Roteigenvectors=[]
    
    for it in [0,1,2]:
        #Define the RotEigenvector
        RotEigenvector=np.zeros(3*numofatoms)
        #generate the vector X along the principle axis
        X=principleAxis[:,it]
        for s in range(numofatoms):
            rvector=np.cross(X,centerofmasscoordinates[s])
            RotEigenvector[3*s]=rvector[0]
            RotEigenvector[3*s+1]=rvector[1]
            RotEigenvector[3*s+2]=rvector[2]
        #Normalize the generated Eigenvector
        RotEigenvector/=np.linalg.norm(RotEigenvector)
        Roteigenvectors.append(RotEigenvector)
    Transeigenvectors=[]
    for it in [0,1,2]:
        #Define the TransEigenvector
        TransEigenvector=np.zeros(3*numofatoms)
        for s in range(numofatoms):
            TransEigenvector[3*s+it]=1
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
    atomicmasses=StandardAtomicWeights
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
    sqrtMm1=np.zeros((3*numofatoms,3*numofatoms))
    for it in range(3*numofatoms):
        atomnum=int(np.floor((it/3)))
        sqrtMm1[it][it]=1./(np.sqrt(atomicmasses[atomorder[atomnum]]))
    #Read in the basis vectors of the finite displacements:
    BasisVectors,deltas=Read.read_basis_vectors(parentfolder+"/")
    print(deltas)
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
            Fplus=np.array(Read.readinForces(parentfolder+"/"+folderplus))
            if len(Fplus)==0:
                print("Error in folder: "+parentfolder+"/"+folderplus)
        except:
            print("Error in folder: "+parentfolder+"/"+folderplus)
            exit()
        try:
            Fminus=np.array(Read.readinForces(parentfolder+"/"+folderminus))
            if len(Fminus)==0:
                print("Error in folder: "+parentfolder+"/"+folderminus)
        except:
            print("Error in folder: "+parentfolder+"/"+folderminus)
            exit()
        diffofforces=(Fplus-Fminus)/(2*deltas[lambd])
        for s1alpha1 in range(3*numofatoms):
            partialFpartialY[s1alpha1][lambd]=diffofforces[s1alpha1]
    Hessian=-partialFpartialY@Tinv
    Hessian=Symmetry.Enforce_Symmetry_On_Hessian(Hessian,parentfolder)
    #built the rescaled Hessian
    rescaledHessian=sqrtMm1@Hessian@sqrtMm1 
    # transform to units 1/cm
    rescaledHessian*=(10**(3)/1.8228884842645)*(2.19474631370540E+02)**2 
    # Diagonalize the rescaled Hessian
    Lambda,Normalmodes=np.linalg.eigh(rescaledHessian)
    #Standard Values for the Translation and Rotational overlaps with Vibrations
    print("Projecting Out Translational Eigenvectors")
    threshhold_string=input("Maximally allowed Weight of Translations with Numerical Normal Modes [float between 0 and 1 or std for Standard Value]:")
    if Util.is_number(threshhold_string):
        threshhold_trans=float(threshhold_string)
    elif threshhold_string=="std":
        threshhold_trans=0.99
    else:
        print("Continuing with Standard values. [0.99]")
        threshhold_trans=0.99
    is_periodic=bool(Read.read_periodicity(parentfolder))
    Rotations_Projector_Flag=False
    if not is_periodic:
        print("Having detected non-periodic calculation") 
    else:
        print("Having detected periodic calculation") 
    strin=input("Do you want to project out the Rotational Eigenvectors?[Y/N]")
    if strin=="N":
        Rotations_Projector_Flag=False
    else:
        Rotations_Projector_Flag=True
        print("Projecting Out Rotational Eigenvectors")
        threshhold_string=input("Maximally allowed Weight of Rotations with Numerical Normal Modes [float between 0 and 1 or std for Standard Value]:")
        if Util.is_number(threshhold_string):
            threshhold_rot=float(threshhold_string)
        elif threshhold_string=="std":
            threshhold_rot=0.9
        else:
            print("Continuing with Standard values. [0.9]")
            threshhold_rot=0.9
    #Get the rescaled eigenvectors of Translation & Rotation:
    Transeigenvectors=Symmetry.getTransEigenvectors(parentfolder+"/Equilibrium_Geometry/",True)
    Roteigenvectors=Symmetry.getRotEigenvectors(parentfolder+"/Equilibrium_Geometry/",True)
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
        weight_trans=np.abs(np.dot(Normalmodes[:,it],np.dot(Orthogonalprojector_trans,Normalmodes[:,it])))
        if Rotations_Projector_Flag:
            projection_rot=np.dot(Orthogonalprojector_rot,projection_trans)
            projection_rot/=np.linalg.norm(projection_rot)
            weight_rot=np.abs(np.dot(Normalmodes[:,it],np.dot(Orthogonalprojector_rot,Normalmodes[:,it])))
            if weight_rot>threshhold_rot and weight_trans>threshhold_trans:
                    orthogonalized_Vibrations.append(projection_rot)
                    Vibrational_eigenvalues.append(Lambda[it])
            elif weight_rot<=threshhold_rot and weight_trans>threshhold_trans:
                print("Mode "+str(it)+" with Frequency="+str(np.sign(Lambda[it])*np.sqrt(np.abs(Lambda[it])))+" 1/cm "+"has rot weight "+str(1.0-weight_rot))
                rotational_subspace.append(it)
            elif weight_rot>threshhold_rot and weight_trans<=threshhold_trans:
                print("Mode "+str(it)+" with Frequency="+str(np.sign(Lambda[it])*np.sqrt(np.abs(Lambda[it])))+" 1/cm "+"has trans weight "+str(1.0-weight_trans))
                translational_subspace.append(it)
            else:
                print("Mode "+str(it)+" with Frequency="+str(np.sign(Lambda[it])*np.sqrt(np.abs(Lambda[it])))+" 1/cm "+"has trans weight "+str(1.0-weight_trans)+" and has rot weight "+str(1.0-weight_rot)+"!")
                exit()
        else:
            normofprojection_trans=np.linalg.norm(projection_trans)
            if weight_trans>threshhold_trans:
                orthogonalized_Vibrations.append(projection_trans/normofprojection_trans)
                Vibrational_eigenvalues.append(Lambda[it])
            else:
                print("Mode"+str(it)+" with Frequency="+str(np.sign(Lambda[it])*np.sqrt(np.abs(Lambda[it])))+" 1/cm "+"has trans weight "+str(1.0-weight_trans))
                translational_subspace.append(it)
    normalmodeEnergies_preliminary=np.array(Vibrational_eigenvalues)*np.sqrt(np.abs(np.array(Vibrational_eigenvalues)))/np.abs(np.array(Vibrational_eigenvalues))
    normalmodes_prelimiary=orthogonalized_Vibrations
    
    
    #Check the correct number of the identified Translational and evtl. Rotational Subspace
    if len(translational_subspace)!=3:
        print("Translational Subspace has Dimension "+str(len(translational_subspace))+"!")
        exit()
    if Rotations_Projector_Flag:
        if len(rotational_subspace)!=3:
            print("Rotational Subspace has Dimension "+str(len(rotational_subspace))+"!")
            exit()
    normalmodes=[]
    normalmodeenergies=[]
    for it in range(len(normalmodeEnergies_preliminary)):
        normalmodes.append(normalmodes_prelimiary[it])
        normalmodeenergies.append(normalmodeEnergies_preliminary[it])    
        
    
    carthesianDisplacements=[]
    normfactors=[]
    for vvector in normalmodes:
        #represent the normal modes, the Transeigenvector and the RotEigenvectors in terms in cartesian components & normalize it
        print(vvector)
        vvector=sqrtMm1@vvector
        normfactor=np.sqrt(np.dot(vvector,vvector))
        normfactors.append(normfactor)
        carthesianDisplacements.append(vvector/normfactor)
    np.save(parentfolder+"/Normal-Mode-Energies",normalmodeenergies)
    np.save(parentfolder+"/normalized-Carthesian-Displacements",carthesianDisplacements)
    np.save(parentfolder+"/Norm-Factors",normfactors)
    Transeigenvectors_unscaled=Symmetry.getTransEigenvectors(parentfolder+"/Equilibrium_Geometry/",False)
    Roteigenvectors_unscaled=Symmetry.getRotEigenvectors(parentfolder+"/Equilibrium_Geometry/",False)
    np.save(parentfolder+"/Translation_Eigenvectors",Transeigenvectors_unscaled)
    np.save(parentfolder+"/Rotational_Eigenvectors",Roteigenvectors_unscaled)
    if writeMolFile:
        Write.write_mol_file(normalmodeenergies,carthesianDisplacements,normfactors,parentfolder)
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
    normalmodeenergies,normalizedcarthesiandisplacements,_=Read.read_vibrations(parentfolder)
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
                    Geometry.changeConfiguration(str(displacementnumber),normalizedCartesianDisplacement,displacementnumber*delta*ConversionFactors['a.u.->A'],sign,parentfolder,folderpath)
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
    VibrationalFrequencies,NormCarthesianDisplacements,normfactors=Read.read_vibrations(path_to_original_data)

    # Iterate over correction directories
    for overdir in os.listdir(path_to_correctiondata):
        itmode=int(overdir[-1])
        numofatoms=int(len(NormCarthesianDisplacements[0])/3)

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
            Force=Read.readinForces(dir)
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
    Write.write_mol_file(VibrationalFrequencies[sortedIndices],NormCarthesianDisplacements[sortedIndices],normfactors[sortedIndices],path_to_original_data)
def get_ir_transition_dipole_moments(parentfolder="./"):
    #get the normal modes from the cartesian displacements
    VibrationalFrequencies,_,normfactors=Read.read_vibrations(parentfolder)
    _,deltas=Read.read_basis_vectors(parentfolder+"/")
    subdirectories=[f for f in os.listdir(parentfolder) if f.startswith('vector=')]
    if len(subdirectories)!=2*len(normfactors):
        raise ValueError('InputError: Number of subdirectories does not match the number of needed Ones!')
    IR_TDMs=[]
    for lambd in range(len(normfactors)):
        folderplus='vector='+str(lambd+1)+'sign=+'
        folderminus='vector='+str(lambd+1)+'sign=-'
        #Project out the Force components into direction of rotation and translation D.O.F.
        try:
            GS_DPM_Plus=Read.read_GS_Dipole_Moment(parentfolder+"/"+folderplus+"/GS_DIPOLE_MOMENTS")
        except:
            print("Error in folder: "+parentfolder+"/"+folderplus)
            exit()
        try:
            GS_DPM_Minus=Read.read_GS_Dipole_Moment(parentfolder+"/"+folderminus+"/GS_DIPOLE_MOMENTS")
        except:
            print("Error in folder: "+parentfolder+"/"+folderminus)
            exit()
        ir_TDM=(GS_DPM_Plus-GS_DPM_Minus)/(2*deltas[lambd])/normfactors[lambd]/np.sqrt((2*ConversionFactors["u->a.u."]*ConversionFactors["1/cm->a.u."]*VibrationalFrequencies[lambd]))
        IR_TDMs.append(ir_TDM)
    np.save(parentfolder+"/"+"IR-Transition-Dipolemoments",np.array(IR_TDMs))
    with open(parentfolder+"/"+"IR-Transition-Dipolemoments.dat","w") as file:
        file.write("Infrared Transition Dipole Moments\n")
        file.write("File generated with Parabola\n")
        for it in range(len(normfactors)):
            file.write(format(it+1,'5.0f')+format(VibrationalFrequencies[it],'12.6f')+format(IR_TDMs[it][0],'12.6f')+format(IR_TDMs[it][1],'12.6f')+format(IR_TDMs[it][2],'12.6f')+"\n") 


