import numpy as np
import os
import Modules.Util as util
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
            KSHamiltonian_alpha=KSHamiltonian
            KSHamiltonian_beta=KSHamiltonian_alpha
            os.remove(parentfolder+"/"+"KSHamiltonian")
        except:
            KSHamiltonian_alpha=np.load(parentfolder+"/KSHamiltonian.npy")
            KSHamiltonian_beta=KSHamiltonian_alpha
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
    return KSHamiltonian_alpha,KSHamiltonian_beta,OLM
def readinMos_AO(parentfolder="./"):
    ## Reads the Molecular Orbitals from a provided file in the AO basis valid for Multiplicity 1
    ## input:
    ## (opt.)   filename            path to the MOs file        (string)
    ## output:  MOs                symmetric np.array(NumBasisfunctions,Numbasisfunction)       Expansion coefficients of the MOs in terms of AO's 
    ## Example: MOs[:,0] are the expansion coefficients of the MO 0 in the canonically ordered atomic Basis
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
    HOMOid=util.getHOMOId(parentfolder)
    Basenumber=0
    MOindices=[]
    for line in MOstring:
        splited_line=line.split()[1:]
        if len(splited_line)>=5:
            if splited_line[0].isdigit() and splited_line[1].isdigit() and splited_line[2].isalpha():
                aoBasisindex=int(splited_line[0])-1
                iterator=0
                for number_string in splited_line[4:]:
                    number=float(number_string)
                    moindex=Basenumber+iterator
                    MOindices.append(moindex-HOMOid)
                    MOs[aoBasisindex,moindex]=number
                    iterator+=1
                if aoBasisindex==NUM_BASIS_FKT-1:
                    Basenumber+=4
    np.save("MOs",MOs)
    np.save("MOindices",MOindices)
    os.remove(parentfolder+"/MOs")
    return MOs,MOindices
def readinMos_RealSpace(parentfolder="./"):
    ## Reads the Molecular Orbitals from a provided file in a discretized real space basis (.cube)
    ## input:
    ## (opt.)   filename            path to the MOs file        (string)
    ## output:  MOs                symmetric np.array(NumBasisfunctions,Numbasisfunction)       Expansion coefficients of the MOs in terms of AO's 
    ## Example: MOs[:,0] are the expansion coefficients of the MO 0 in the canonically ordered atomic Basis
    spinmultiplicity=checkforSpinMultiplicity(parentfolder)
    npyfiles = [f for f in os.listdir(parentfolder) if f.endswith('.npy')]
    MOs=[]
    MOindices=[]
    if spinmultiplicity==1:
        for file in npyfiles:
            if util.represents_int(file.split(".npy")[0]):
                Monumber=int(file.split(".npy")[0])
                data=np.load(parentfolder+"/"+file)
                MOs.append(data)
                MOindices.append(Monumber)
    return np.array(MOs),MOindices
def readinMos(parentfolder="./"):
    ## Reads the Molecular Orbitals from a provided file
    ## input:
    ## (opt.)   filename            path to the MOs file        (string)
    ## output:  MOs                symmetric np.array(NumBasisfunctions,Numbasisfunction)       Expansion coefficients of the MOs in terms of AO's 
    ## Example: MOs[:,0] are the expansion coefficients of the MO 0 in the canonically ordered atomic Basis
    try:
        MOs=np.load("MOs.npy")
        MOindices=np.load("MOindices.npy")
    except:
        #check if MO file exists
        MOfiles= [f for f in os.listdir(parentfolder) if f.endswith('MOs')]
        npyfiles = [f for f in os.listdir(parentfolder) if f.endswith('.npy')]
        WFNFiles=[f for f in npyfiles if util.represents_int(f.split(".npy")[0])]
        if len(MOfiles)==1:
            MOs,MOindices=readinMos_AO(parentfolder)
        elif len(WFNFiles)>0:
            MOs,MOindices=readinMos_RealSpace(parentfolder)
    return MOs,MOindices
    
def readinCubeFile(filename,parentfolder="./"):
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
        for line in lines[2:]:
            if it==1:
                Nx=int(line.split()[0])
            if it==2:
                Ny=int(line.split()[0])
            if it==3:
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
def readinAtomicCoordinates(folder="./"):
    ##Reads in the atomic coordinates from a provided xyz file (these coordinates are independent from cell vectors)! 
    ## input:
    ## (opt.)   folder              path to the folder of the .xyz file         (string)
    ## output:  Atoms               list of sublists. 
    ##                              Each of the sublists has five elements. 
    ##                              Sublist[0] contains the atomorder as a int.
    ##                              Sublist[1] contains the symbol of the atom.
    ##                              Sublist[2:] containst the x y z coordinates.
    filename=util.getxyzfilename(folder)
    Atoms=[]
    with open(filename) as f:
        lines=f.readlines()
        it=1
        for l in lines[2:]:
            Atoms.append([it,l.split()[0],float(l.split()[1]),float(l.split()[2]),float(l.split()[3])])
            it+=1
    f.close()
    return Atoms
def readinVibrations(parentfolder="./"):
    try:
        VibrationalFrequencies=np.load(parentfolder+"/Normal-Mode-Energies.npy")
        CarthesianDisplacements=np.load(parentfolder+"/normalized-Carthesian-Displacements.npy")
        normfactors=np.load(parentfolder+"/Norm-Factors.npy")
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

def readinCellSize(path="./"):
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
				if stateiteratorflag and len(line.split())==3 and util.represents_int(line.split()[1]):
					if abs(float(line.split()[2]))>=minweight:
						stateiterator.append([int(line.split()[0])-util.getHOMOId(path)-1,int(line.split()[1])-util.getHOMOId(path)-1,float(line.split()[2])])
				if len(line.split())==3:
					if util.represents_int(line.split()[0]) and not util.represents_int(line.split()[1]) and line.split()[2]=="eV":
						if int(line.split()[0])<=len(energies):
							it=int(line.split()[0])-1
							if np.abs(energies[it]-float(line.split()[1]))<0.00001:
								if stateiterator:
									states.append([energies[it-1],stateiterator])
									stateiterator=[]
								stateiteratorflag=True
	return states
def readinDipoleMoments(path="./"):
    ## Parses the excited states & DipoleMoments from a TDDFPT calculation done in CP2K  
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
		TransitionDipolevectors=[];Oscillatorstrenghts=[]
		with open(path+"/"+out_files[0],'r') as f:
			for line in f:
				if len(line.split())>4:
					if line.split()[0]=="TDDFPT" and line.split()[1]==":" and line.split()[2]=="CheckSum" and line.split()[3]=="=":
						readflag=False
					if readflag:
						energy=line.split()[2]
						energies.append(float(energy))
						dx=line.split()[3]
						dy=line.split()[4]
						dz=line.split()[5]
						TransitionDipolevectors.append(np.array([float(dx),float(dy),float(dz)]));Oscillatorstrenghts.append(float(line.split()[6]))
					if line.split()[0]=="number" and line.split()[1]=="energy" and line.split()[2]=="(eV)" and line.split()[3]=="x" and line.split()[4]=="y":
						readflag=True
	np.save("Transitiondipolevectors",TransitionDipolevectors);np.save("Oscillatorstrenghts",Oscillatorstrenghts)
	np.save(path+"/"+"ExcitedStateEnergies",energies)
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

def getOutFileName(path="./"):
    #Get the name of the .out file from the path
    out_files = [f for f in os.listdir(path) if f.endswith('.out')]
    if len(out_files) != 1:
        raise ValueError('InputError: There should be only one .out file in the current directory')
    #Get the number of atoms from the xyz file
    outfilename=out_files[0]
    return outfilename

def readinG0W0Energies(path="./"):
    ## Reads in the G0W0 energies and the respective Orbitals
    ## input: 
    ## (opt.)   parentfolder    path to the folder of the BasisHessian file         (string)
    ## output:  
    ##          BasisVectors    normalized displaced vectors                        (list of np.arrays)     
    ##          delta           displacementfactor                                  (float)
    ##          unit            unit of the displacementfactor                      (string, either 'Bohr' or 'sqrt(u)*Bohr') 
    out_file=getOutFileName(path)
    readflag=False
    G0W0flag=False
    orbitals=[]
    E_SCF=[]
    Sig_C=[]
    Sigxmvxc=[]
    E_QP=[]
    with open(path+"/"+out_file) as f:
        lines=f.readlines()
        for line in lines:
            if len(line.split())>1:
                if line.split()[0]=="GW" and line.split()[1]=="HOMO-LUMO":
                    G0W0flag=False
                if readflag and G0W0flag:
                    orbitals.append(int(line.split()[0])-1)
                    E_SCF.append(float(line.split()[4]))
                    Sig_C.append(float(line.split()[5]))
                    Sigxmvxc.append(float(line.split()[6]))
                    E_QP.append(float(line.split()[7]))
                if line.split()[0]=="G0W0" and line.split()[1]=="results":
                    G0W0flag=True
                if line.split()[0]=="Molecular" and line.split()[1]=="orbital" and line.split()[2]=="E_SCF":
                    readflag=True
    return  orbitals,np.array(E_SCF),np.array(Sig_C),np.array(Sigxmvxc),np.array(E_QP)

def readinGSEnergy(path="./"):
    outfilename=getOutFileName(path)
    GSEnergy=0.0
    with open(path+"/"+outfilename,'r') as f:
        for line in f:
            if len(line.split())>8:
                if line.split()[0]=="ENERGY|" and line.split()[1]=="Total" and line.split()[2]=="FORCE_EVAL":
                    GSEnergy=float(line.split()[8])
    
    return GSEnergy      

    
