import Modules.Geometry as Geometry
import numpy as np
import Modules.PhysConst as PhysConst
import Modules.Read as Read
import Modules.Write as Write
import Modules.Util as Util
import os
#get the environmental variable 
pathtocp2k=os.environ["cp2kpath"]
pathtobinaries=pathtocp2k+"/exe/local/"
def Vib_Ana_inputs(delta,vectors=[],linktobinary=True,binary="cp2k.popt",parentpath="./",binaryloc=pathtobinaries):
    ConFactors=PhysConst.ConversionFactors()
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
            Geometry.changeConfiguration(folderlabel,vec,delta*ConFactors['a.u.->A'],sign,parentpath)
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

    coordinates,masses,_=Geometry.getCoordinatesAndMasses(pathtoEquilibriumxyz)
    #Compute the Intertia Tensor
    I=Geometry.getInertiaTensor(coordinates,masses)
    centerofmasscoordinates,_=Geometry.ComputeCenterOfMassCoordinates(coordinates,masses)

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
def getHessian(InteractiveFlag=True,Rotations_Projector_String="Y" ,parentfolder="./",writeMolFile=True):
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
    atomicmasses=PhysConst.StandardAtomicWeights()
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
    if InteractiveFlag==True:
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
    BasisVectors,delta=Read.readinBasisVectors(parentfolder+"/")
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
            Fplus=np.dot(Orthogonalprojector,np.array(Read.readinForces(parentfolder+"/"+folderplus)))
        except:
            print("Error in folder: "+parentfolder+"/"+folderplus)
            exit()
        try:
            Fminus=np.dot(Orthogonalprojector,np.array(Read.readinForces(parentfolder+"/"+folderminus)))
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
    if InteractiveFlag:
        threshhold_string=input("Maximally allowed Weight of Translations with Numerical Normal Modes [float between 0 and 1 or std for Standard Value]:")
    else:
        threshhold_string="std"
    if Util.is_number(threshhold_string):
        threshhold_trans=float(threshhold_string)
    elif threshhold_string=="std":
        threshhold_trans=0.999
        threshhold_rot=0.999
    else:
        print("Have not recognized Input! Continuing with Standard values. [0.999]")
        threshhold_trans=0.999
        threshhold_rot=0.999 
    if Rotations_Projector_Flag:
        if InteractiveFlag:
            threshhold_string=input("Maximally allowed Weight of Rotations with Numerical Normal Modes [float between 0 and 1 or std for Standard Value]:")
        else:
            threshhold_string="std"
        if Util.is_number(threshhold_string):
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
        projection_rot=np.dot(Orthogonalprojector_rot,Normalmodes[:,it])
        weight_rot=np.sqrt(np.dot(projection_rot,projection_rot))
        if Rotations_Projector_Flag:
            if weight_rot>threshhold_rot and weight_trans>threshhold_trans:
                    orthogonalized_Vibrations.append(projection_trans/normofprojection_trans)
                    Vibrational_eigenvalues.append(Lambda[it])
            elif weight_rot<=threshhold_rot and weight_trans>threshhold_trans:
                print("Mode with Frequency="+str(np.sign(Lambda[it])*np.sqrt(np.abs(Lambda[it])))+" 1/cm "+"has rot weight "+str(1.0-weight_rot))
                rotational_subspace.append(it)
            elif weight_rot>threshhold_rot and weight_trans<=threshhold_trans:
                print("Mode with Frequency="+str(np.sign(Lambda[it])*np.sqrt(np.abs(Lambda[it])))+" 1/cm "+"has trans weight "+str(1.0-weight_trans))
                translational_subspace.append(it)
            else:
                print("Mode with Frequency="+str(np.sign(Lambda[it])*np.sqrt(np.abs(Lambda[it])))+" 1/cm "+"has trans weight "+str(1.0-weight_trans)+" and has rot weight "+str(1.0-weight_rot)+"!")
                exit()
        else:
            if weight_trans>threshhold_trans:
                orthogonalized_Vibrations.append(projection_trans/normofprojection_trans)
                Vibrational_eigenvalues.append(Lambda[it])
            else:
                print("Mode with Frequency="+str(np.sign(Lambda[it])*np.sqrt(np.abs(Lambda[it])))+" 1/cm "+"has trans weight "+str(1.0-weight_trans))
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
    np.save(parentfolder+"/Normal-Mode-Energies",normalmodeenergies)
    np.save(parentfolder+"/normalized-Carthesian-Displacements",carthesianDisplacements)
    np.save(parentfolder+"/Norm-Factors",normfactors)
    if writeMolFile:
        Write.writemolFile(normalmodeenergies,carthesianDisplacements,normfactors,parentfolder)
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
    ConFactors=PhysConst.ConversionFactors()
    normalmodeenergies,normalizedcarthesiandisplacements,_=Read.readinVibrations(parentfolder)
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
                    Geometry.changeConfiguration(str(displacementnumber),normalizedCartesianDisplacement,displacementnumber*delta*ConFactors['a.u.->A'],sign,parentfolder,folderpath)
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
    VibrationalFrequencies,NormCarthesianDisplacements,normfactors=Read.readinVibrations(path_to_original_data)

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
    Write.writemolFile(VibrationalFrequencies[sortedIndices],NormCarthesianDisplacements[sortedIndices],normfactors[sortedIndices],path_to_original_data)
