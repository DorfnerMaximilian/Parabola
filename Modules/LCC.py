import numpy as np
import os
import copy
import Modules.Util as Util
import Modules.PhysConst as PhysConst
import Modules.Read as Read
import Modules.AtomicBasis as AtomicBasis
import Modules.TDDFT as TDDFT
import Modules.VibAna as VibAna
pathtocp2k=os.environ["cp2kpath"]
pathtobinaries=pathtocp2k+"/exe/local/"
def LCC_inputs(delta,parentfolder="./",linktobinary=True,binary="cp2k.popt",parentpath="./",binaryloc=pathtobinaries):
    _,normCD,_=Read.readinVibrations(parentfolder)
    nCD=[]
    for it in range(np.shape(normCD)[0]):
        nCD.append(normCD[it,:])
    VibAna.Vib_Ana_inputs(delta,nCD,linktobinary,binary,parentpath,binaryloc)

def getManyBodyCouplings(eta,LCC,id_homo):
    ''' Function to obtain the Many-Body Coupling Constants from the CP2K TDDFT excited states and the DFT Coupling constants
        input:  eta:         (np.array)            numpy array which encodes the states
                                                      required structure: states[n] encodes the n th excited state
                                                                          states[n][0] is its energy
                                                                          states[n][1] is a list of lists, where each list in this list contains 
                                                                          list[0] hole index list[1] particle index and list[2] the weight of this 
                                                                          particle hole state
                                                                          second index first component
                couplingConstants (np.array)          the DFT coupling constants as outputted by "getLinearCouplingConstants"
                                                                            
                HOMOit                (int)           the index of the HOMO orbital (python convention)
    '''

    #generate g matrix, h matrix and k matrices
    g=LCC[:,id_homo+1:,id_homo+1:]
    h=LCC[:,:id_homo+1,:id_homo+1]*(-1) # minus 1 due to fermionic commutator!
    k=LCC[:,:id_homo+1,id_homo+1:]
    #get the number of excited States to take into account
    Num_OfExciteStates=np.shape(eta)[-1]
    Num_OfModes=np.shape(LCC)[0]
    #Normalize the eta
    for p in Util.progressbar(range(Num_OfExciteStates),"Normalizing States:",40):
        eta[:,:,p]/=np.trace(np.transpose(eta[:,:,p])@eta[:,:,p])
    K=np.zeros((Num_OfModes,Num_OfExciteStates)) #Coupling of excited state to ground state
    for m in Util.progressbar(range(Num_OfExciteStates),"Computing Coupling to Ground State:",40):
        etap=eta[id_homo+1:,:id_homo+1,m] #First index electrons second hole
        for lamb in range(Num_OfModes):
            klamb=k[lamb,:,:]
            K[lamb,m]=np.trace(klamb@etap)
    H=np.zeros((Num_OfModes,Num_OfExciteStates,Num_OfExciteStates)) #Coupling between the excited states
    for p in Util.progressbar(range(Num_OfExciteStates),"Computing Coupling between excited States:",40):
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
#######################################################################################################
#Helper routine to get the range of considered electronic states
#######################################################################################################
def getadiabaticallyConnectedEigenstates(orthogonalEigenstates_Eq,orthorgonalEigenstates_deflected,Tmatrix_deflected):
    ''' 
    Compute adiabatically connected eigenstates.

    Parameters:
        orthogonalEigenstates_Eq: list
            List of orthogonal eigenstates at equilibrium geometry.

        orthogonalEigenstates_deflected: list
            List of orthogonal eigenstates at deflected geometry.

        Tmatrix_deflected: numpy.ndarray
            Transition matrix at deflected geometry.

        spread: int, optional
            Compute coupling elements for orbitals HOMO-spread, HOMO-spread+1, ..., LUMO+spread.
        

    Returns:
        list
            A list containing indices of adiabatically connected eigenstates.

    Raises:
        ValueError: If maximum overlap is too small.

    Note:
        This function computes adiabatically connected eigenstates by finding the maximum overlap between eigenstates at equilibrium and deflected geometries.
        It iterates over each equilibrium eigenstate and finds the corresponding deflected eigenstate with the maximum overlap.
        If the maximum overlap is less than 0.5, it raises a ValueError.

    '''
    adibaticallyConnectediters_Plus = []

    # Precompute some values
    orthogonal_states_Plus = (np.array(orthorgonalEigenstates_deflected).T).copy()
    Tmatrix_Plus_dot_states_Plus = Tmatrix_deflected @ orthogonal_states_Plus
    # Get the adiabatically connected eigenvalues/states
    for it0 in range(np.shape(orthogonalEigenstates_Eq)[0]):
        overlaps = np.dot(orthogonalEigenstates_Eq[it0], Tmatrix_Plus_dot_states_Plus)
        iter1 = np.argmax(np.abs(overlaps))
        maximumAbsOverlap = np.abs(overlaps[iter1])
        adibaticallyConnectediters_Plus.append(iter1)
        if overlaps[iter1]<0:
            orthorgonalEigenstates_deflected[iter1]*=-1
        if maximumAbsOverlap < 0.5:
            print("Maximum Overlap: ", maximumAbsOverlap)
            raise ValueError("Maximum Overlap too small! Check your inputs!")
    return adibaticallyConnectediters_Plus


#######################################################################################################
#Function to Compute the local Coupling constants g 
#######################################################################################################
def getLinearCouplingConstants(parentfolder="./",idmin=0,idmax=-1):
    '''Compute linear coupling constants from finitly displaced geometries.
    
    Parameters:
        parentfolder (str): Absolute or relative path where the geometry optimized .xyz file lies, 
                            with electronic structure data at displaced geometries in subfolders.
                            Default is the current directory.
                                                 
       Output:
        Saves coupling constants in a numpy .npy file named "Linear_Coupling_Constants.npy" in the parentfolder.
    '''
    
    ConFactors=PhysConst.ConversionFactors()
    #Get the .xyz file
    xyz_files = [f for f in os.listdir(parentfolder+"/"+'Equilibrium_Geometry') if f.endswith('.xyz')]
    if len(xyz_files) != 1:
        raise ValueError('InputError: There should be only one xyz file in the directory:'+parentfolder+"/"+'Equilibrium_Geometry/')

    #----------------------------------------------------------------------
    # Equilibrium configuration
    #----------------------------------------------------------------------

    #get the Equilibrium Configuration 
    Atoms_Eq=Read.readinAtomicCoordinates(parentfolder+"/Equilibrium_Geometry/")
    #Construct Basis of the Equilibrium configuration
    Basis_Eq=AtomicBasis.getBasis(parentfolder+"/Equilibrium_Geometry/")
    #Read in the KS Hamiltonian
    KSHamiltonian_Eq,S_Eq=Read.readinMatrices(parentfolder+"/Equilibrium_Geometry/")
    #perform a Loewdin Orthogonalization
    Sm12_Eq=Util.LoewdinTransformation(S_Eq)
    KSHorth_Eq=np.transpose(Sm12_Eq)@KSHamiltonian_Eq@Sm12_Eq
    #Diagonalize to the KS Hamiltonian in the ortonormal Basis
    E_Eq,a_orth_Eq=np.linalg.eigh(KSHorth_Eq)
    #get the normalized Eigenstates in the non-orthorgonal Basis & fix Phase
    orthorgonalEigenstates_Eq = []
    # Precompute phases
    phases = np.array([TDDFT.getPhaseOfMO(a_orth_Eq[:, it]) for it in range(len(E_Eq))])
    #parsing the input
    Homoid=Util.getHOMOId(parentfolder+"/Equilibrium_Geometry/")
    #parsing for occupied orbitals
    index_low=0
    if idmin==0:
        index_low=0
        print("Computing coupling matrix elements for all occupied orbitals!")
    elif idmin>=Homoid:
        print("Cannot take into account more then "+str(Homoid+1)+" occupied orbitals.\n")
        print("Continuing by taking into account all occupied orbitals!")
        index_low=0
    else:
        print("Computing coupling matrix elements for the occupied orbitals HOMO, ... HOMO-"+str(idmin)+"\n")
        index_low=Homoid-idmin
    #parsing for empty orbitals
    index_up=0
    if idmax==-1:
        index_up=len(E_Eq)
        print("Computing coupling matrix elements for all empty orbitals!")
    elif idmax+Homoid+1>=len(E_Eq):
        print("Cannot take into account more then "+str(len(E_Eq)-Homoid-1)+" empty orbitals.\n")
        print("Continuing by taking into account all empty orbitals!")
        index_up=len(E_Eq)
    else:
        print("Computing coupling matrix elements for the empty orbitals LUMO, ... LUMO+"+str(idmax)+"\n")
        index_up=Homoid+idmax
    # Compute and append normalized eigenstates
    included_orbitals=[]
    for it in range(len(E_Eq)):
        orth_eigenstate = a_orth_Eq[:, it]  # Make a copy to avoid modifying the original data
        phase = phases[it]
        orth_eigenstate *= phase
        if it>=index_low and it<=index_up:
            included_orbitals.append(it)
            orthorgonalEigenstates_Eq.append(orth_eigenstate)
        
    _,delta=Read.readinBasisVectors(parentfolder)
    #get the normal modes from the cartesian displacements
    VibrationalFrequencies,_,normfactors=Read.readinVibrations(parentfolder)
    #This has index convention lambda,mu
    couplingConstants=np.zeros((len(normfactors),np.shape(E_Eq)[0],np.shape(E_Eq)[0]))
    for mu in Util.progressbar(range(len(normfactors)),"Coupling Constants:",40):
        #----------------------------------------------------------------------
        # Positively displaced 
        #----------------------------------------------------------------------
        folderplus='vector='+str(mu+1)+'sign=+'
        #Read in the KS Hamiltonian and the overlap matrix
        KSHamiltonian_Plus,OLM_Plus=Read.readinMatrices(parentfolder+"/"+folderplus+'/')
        #Get the stompositions for the positively displaced atoms
        Atoms_Plus=Read.readinAtomicCoordinates(parentfolder+"/"+folderplus)
        Sm12_Plus=Util.LoewdinTransformation(OLM_Plus)
        KSHorth_P=np.dot(Sm12_Plus,np.dot(KSHamiltonian_Plus,Sm12_Plus))
        EPlus,a_orth_Plus=np.linalg.eigh(KSHorth_P)
        T_Eq_Plus=AtomicBasis.getTransformationmatrix(Atoms_Eq,Atoms_Plus,Basis_Eq)
        T_matrix_Plus=Sm12_Eq@T_Eq_Plus@Sm12_Plus
        #----------------------------------------------------------------------
        # Negatively displaced 
        #----------------------------------------------------------------------
        folderminus='vector='+str(mu+1)+'sign=-'
        #Read in the KS Hamiltonian and the overlap matrix
        KSHamiltonian_Minus,OLM_Minus=Read.readinMatrices(parentfolder+"/"+folderminus+'/')
        #Get the atom positions for the negatively displaced atoms
        Atoms_Minus=Read.readinAtomicCoordinates(parentfolder+"/"+folderminus)
        #perform a Loewdin Orthogonalization
        Sm12_Minus=Util.LoewdinTransformation(OLM_Minus)
        KSHorth_Minus=np.dot(Sm12_Minus,np.dot(KSHamiltonian_Minus,Sm12_Minus))
        #Diagonalize the KS Hamiltonian in the orthorgonal Basis
        EMinus,a_orth_Minus=np.linalg.eigh(KSHorth_Minus)
        #obtain the Basis Transformation Matrix 
        T_Eq_Minus=AtomicBasis.getTransformationmatrix(Atoms_Eq,Atoms_Minus,Basis_Eq)
        T_Matrix_Minus=Sm12_Eq@T_Eq_Minus@Sm12_Minus

        #fix the phase of the Eigenstates
        orthorgonalEigenstates_Plus=[]
        for it in range(len(EPlus)):
            orth_eigenstate=a_orth_Plus[:,it]
            orth_eigenstate*=TDDFT.getPhaseOfMO(orth_eigenstate)
            if it in included_orbitals:
                orthorgonalEigenstates_Plus.append(orth_eigenstate)
        #fix the phase of the Eigenstates
        orthorgonalEigenstates_Minus=[]
        for it in range(len(EMinus)):
            orth_eigenstate=a_orth_Minus[:,it]
            orth_eigenstate*=TDDFT.getPhaseOfMO(orth_eigenstate)
            if it in included_orbitals:
                orthorgonalEigenstates_Minus.append(orth_eigenstate)
        #Get the adiabtically connected eigenvalues/states for the positive displacement
        adibaticallyConnectediters_Plus=getadiabaticallyConnectedEigenstates(orthorgonalEigenstates_Eq,orthorgonalEigenstates_Plus,T_matrix_Plus)
        #Check that each iterator is exactly once in the adibaticallyConnectediters_Plus set
        for it in included_orbitals:
            if adibaticallyConnectediters_Plus.count(it)!=1:
                ValueError("Some eigenstates appear more or less then once as maximum weight states! Check your inputs!")
        #Get the adiabtically connected eigenvalues/states for the negative displacement
        adibaticallyConnectediters_Minus=getadiabaticallyConnectedEigenstates(orthorgonalEigenstates_Eq,orthorgonalEigenstates_Minus,T_Matrix_Minus)
        #Check that each iterator is exactly once in the adibaticallyConnectediters set
        for it in included_orbitals:
            if adibaticallyConnectediters_Minus.count(it)!=1:
                ValueError("Some eigenstates appear more or less then once as maximum weight states! Check your inputs!")
        #Precompute the matrix products
        Tmatrix_Plus_dot_states_Plus = T_matrix_Plus@np.array(orthorgonalEigenstates_Plus).T
        Tmatrix_Plus_dot_states_Minus=T_Matrix_Minus@np.array(orthorgonalEigenstates_Minus).T
        for it0 in range(len(included_orbitals)):
            for it1 in range(len(included_orbitals)):
                if it0==it1:
                    deltaE=(EPlus[included_orbitals[adibaticallyConnectediters_Plus[it0]]]-EMinus[included_orbitals[adibaticallyConnectediters_Minus[it1]]])/(2*delta)*normfactors[mu]
                    couplingConstants[mu,included_orbitals[it0],included_orbitals[it1]]=ConFactors['E_H/a_0*hbar/sqrt(2*m_H)->cm^(3/2)']/(VibrationalFrequencies[mu])**(1.5)*deltaE
                else:
                    overlap1=np.dot(orthorgonalEigenstates_Eq[it0],Tmatrix_Plus_dot_states_Plus[:,adibaticallyConnectediters_Plus[it1]])
                    overlap2=np.dot(orthorgonalEigenstates_Eq[it0],Tmatrix_Plus_dot_states_Minus[:,adibaticallyConnectediters_Minus[it1]])
                    deltaE=(E_Eq[included_orbitals[it1]]-E_Eq[included_orbitals[it0]])*(overlap1-overlap2)/(2*delta)*normfactors[mu]
                    couplingConstants[mu,included_orbitals[it0],included_orbitals[it1]]=ConFactors['E_H/a_0*hbar/sqrt(2*m_H)->cm^(3/2)']/(VibrationalFrequencies[mu])**(1.5)*deltaE
    np.save(parentfolder+"/Linear_Coupling_Constants",couplingConstants)
