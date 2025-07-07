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
def LCC_inputs(deltas,parentfolder="./",linktobinary=True,binaryloc=pathtobinaries):
    _,normCD,_=Read.readinVibrations(parentfolder)
    nCD=[]
    for it in range(np.shape(normCD)[0]):
        nCD.append(normCD[it,:])
    VibAna.Vib_Ana_inputs(deltas,nCD,parentpath=parentfolder)

def getManyBodyCouplings(eta, LCC, id_homo):
    """
    Calculates the Many-Body Coupling Constants using vectorized NumPy operations.

    Args:
        eta (np.ndarray): 3D array encoding the excited states.
                          Shape: (orbitals, orbitals, num_excited_states).
        LCC (np.ndarray): The DFT linear coupling constants.
                          Shape: (num_modes, orbitals, orbitals).
        id_homo (int): The index of the HOMO orbital (0-indexed).

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - H (np.ndarray): Coupling matrix between excited states.
                              Shape: (num_modes, num_excited_states, num_excited_states).
            - K (np.ndarray): Coupling matrix to the ground state.
                              Shape: (num_modes, num_excited_states).
    """
    # --- Setup: No changes needed here, these are fast operations ---
    Num_OfModes = LCC.shape[0]
    Num_OfExciteStates = eta.shape[-1]
    
    # Slicing assumes the first (id_homo + 1) orbitals are holes
    # and the remaining are electrons (particles).
    num_holes = id_homo + 1
    
    g = LCC[:, num_holes:, num_holes:]        # Particle-particle block
    h = LCC[:, :num_holes, :num_holes] * -1   # Hole-hole block
    k = LCC[:, :num_holes, num_holes:]        # Hole-particle block

    # --- 1. Vectorized State Normalization ---
    # The original loop calculates the squared Frobenius norm for each state matrix
    # and divides by it. This can be done for all states at once.
    # The norm is sum of squares of all elements.
    norm_factors = np.sum(eta**2, axis=(0, 1), keepdims=True)
    norm_factors[norm_factors == 0] = 1  # Avoid division by zero for null states
    eta = eta / norm_factors
    # --- 2. Vectorized Ground State Coupling (K matrix) ---
    # The nested loops for K are replaced by a single, powerful np.einsum call.
    # 'einsum' stands for Einstein summation, a notation for tensor operations.
    # Original operation: K[lamb, m] = np.trace(k[lamb] @ etap[m])
    # einsum equivalent: K_lm = sum_{i,j} k_lij * eta_jim
    eta_slice_k = eta[num_holes:, :num_holes, :]
    K = np.einsum('lij,jim->lm', k, eta_slice_k, optimize=True)
    # --- 3. Vectorized Excited State Coupling (H matrix) ---
    # The slow triple nested loop is replaced by two einsum calls.
    eta_sub = eta[num_holes:, :num_holes, :] # particle-hole slice of all states
    # Term 1: np.trace(np.transpose(etaq) @ g @ etap)
    # einsum: H_lqp = sum_{e,f,h} eta_ehq * g_lef * eta_fhp
    term1 = np.einsum('ehq,lef,fhp->lqp', eta_sub, g, eta_sub, optimize=True)
    
    # Term 2: np.trace(etaq @ h @ np.transpose(etap))
    # einsum: H_lqp = sum_{e,i,j} eta_eiq * h_lij * eta_ejp
    term2 = np.einsum('eiq,lij,ejp->lqp', eta_sub, h, eta_sub, optimize=True)
    H = term1 + term2
    # The original code copied H[l,q,p] to H[l,p,q]. The einsum calculation
    # computes the full, symmetric matrix at once, making this unnecessary.

    # --- 4. Save and Return ---
    np.save("H_CouplingConstants", H)
    np.save("K_CouplingConstants", K)
    return H, K

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
        if maximumAbsOverlap < 0.60:
            print("Maximum Overlap: ", maximumAbsOverlap," for Orbital ", it0)
    return adibaticallyConnectediters_Plus,orthorgonalEigenstates_deflected


#######################################################################################################
#Function to Compute the local Coupling constants g 
#######################################################################################################
def getLCC_EIG(parentfolder="./",idmin=0,idmax=-1,cell_vectors=[0.0, 0.0, 0.0]):
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
    #Diagonalize to the KS Hamiltonian in the ortonormal Basis
    E_Eq,a_orth_Eq,Sm12_Eq=Util.Diagonalize_KS_Hamiltonian(parentfolder+"/Equilibrium_Geometry/")
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
        
    _,deltas=Read.readinBasisVectors(parentfolder)
    #get the normal modes from the cartesian displacements
    VibrationalFrequencies,_,normfactors=Read.readinVibrations(parentfolder)
    #This has index convention lambda,mu
    couplingConstants=np.zeros((len(normfactors),np.shape(E_Eq)[0],np.shape(E_Eq)[0]))
    for mu in Util.progressbar(range(len(normfactors)),"Coupling Constants:",40):
        #----------------------------------------------------------------------
        # Positively displaced 
        #----------------------------------------------------------------------
        folderplus='vector='+str(mu+1)+'sign=+'
        #Get the stompositions for the positively displaced atoms
        Atoms_Plus=Read.readinAtomicCoordinates(parentfolder+"/"+folderplus)
        EPlus,a_orth_Plus,Sm12_Plus=Util.Diagonalize_KS_Hamiltonian(parentfolder+"/"+folderplus+"/")
        T_Eq_Plus=AtomicBasis.getTransformationmatrix(Atoms_Eq,Atoms_Plus,Basis_Eq,cell_vectors)
        T_matrix_Plus=Sm12_Eq@T_Eq_Plus@Sm12_Plus
        #T_matrix_Plus=np.eye(np.shape(Sm12_Plus)[0])
        #----------------------------------------------------------------------
        # Negatively displaced 
        #----------------------------------------------------------------------
        folderminus='vector='+str(mu+1)+'sign=-'
        #Get the atom positions for the negatively displaced atoms
        Atoms_Minus=Read.readinAtomicCoordinates(parentfolder+"/"+folderminus)
        EMinus,a_orth_Minus,Sm12_Minus=Util.Diagonalize_KS_Hamiltonian(parentfolder+"/"+folderminus+"/")
        #obtain the Basis Transformation Matrix 
        T_Eq_Minus=AtomicBasis.getTransformationmatrix(Atoms_Eq,Atoms_Minus,Basis_Eq,cell_vectors)
        T_Matrix_Minus=Sm12_Eq@T_Eq_Minus@Sm12_Minus
        #T_Matrix_Minus=np.eye(np.shape(Sm12_Plus)[0])
        #fix the phase of the Eigenstates
        orthorgonalEigenstates_Plus=[]
        for it in range(len(EPlus)):
            orth_eigenstate=a_orth_Plus[:,it]
            if it in included_orbitals:
                orthorgonalEigenstates_Plus.append(orth_eigenstate)
        #fix the phase of the Eigenstates
        orthorgonalEigenstates_Minus=[]
        for it in range(len(EMinus)):
            orth_eigenstate=a_orth_Minus[:,it]
            if it in included_orbitals:
                orthorgonalEigenstates_Minus.append(orth_eigenstate)
        #Get the adiabtically connected eigenvalues/states for the positive displacement
        adibaticallyConnectediters_Plus,orthorgonalEigenstates_Plus=getadiabaticallyConnectedEigenstates(orthorgonalEigenstates_Eq,orthorgonalEigenstates_Plus,T_matrix_Plus)
        #Check that each iterator is exactly once in the adibaticallyConnectediters_Plus set
        for it in included_orbitals:
            if adibaticallyConnectediters_Plus.count(it)!=1:
                ValueError("Some eigenstates appear more or less then once as maximum weight states! Check your inputs!")
        #Get the adiabtically connected eigenvalues/states for the negative displacement
        adibaticallyConnectediters_Minus,orthorgonalEigenstates_Minus=getadiabaticallyConnectedEigenstates(orthorgonalEigenstates_Eq,orthorgonalEigenstates_Minus,T_Matrix_Minus)
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
                    deltaE=(EPlus[included_orbitals[adibaticallyConnectediters_Plus[it0]]]-EMinus[included_orbitals[adibaticallyConnectediters_Minus[it1]]])/(2*deltas[mu])*normfactors[mu]
                    couplingConstants[mu,included_orbitals[it0],included_orbitals[it1]]=ConFactors['E_H/a_0*hbar/sqrt(2*m_H)->cm^(3/2)']/(VibrationalFrequencies[mu])**(1.5)*deltaE
                else:
                    overlap1=np.dot(orthorgonalEigenstates_Eq[it0],Tmatrix_Plus_dot_states_Plus[:,adibaticallyConnectediters_Plus[it1]])
                    overlap2=np.dot(orthorgonalEigenstates_Eq[it0],Tmatrix_Plus_dot_states_Minus[:,adibaticallyConnectediters_Minus[it1]])
                    deltaE=(overlap1-overlap2)/(2*deltas[mu])*normfactors[mu]*(E_Eq[included_orbitals[it1]]-E_Eq[included_orbitals[it0]])
                    couplingConstants[mu,included_orbitals[it0],included_orbitals[it1]]=ConFactors['E_H/a_0*hbar/sqrt(2*m_H)->cm^(3/2)']/(VibrationalFrequencies[mu])**(1.5)*deltaE
    np.save(parentfolder+"/Linear_Coupling_Constants",couplingConstants)
def getLCC_STATES(parentfolder="./",idmin=0,idmax=-1,cell_vectors=[0.0, 0.0, 0.0]):
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
    #Diagonalize to the KS Hamiltonian in the ortonormal Basis
    E_Eq,a_orth_Eq,Sm12_Eq=Util.Diagonalize_KS_Hamiltonian(parentfolder+"/Equilibrium_Geometry/")
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
    print(np.shape(orthorgonalEigenstates_Eq))
    States=np.transpose(np.array(orthorgonalEigenstates_Eq))
    _,deltas=Read.readinBasisVectors(parentfolder)
    #get the normal modes from the cartesian displacements
    VibrationalFrequencies,_,normfactors=Read.readinVibrations(parentfolder)
    #This has index convention lambda,mu
    couplingConstants=np.zeros((len(normfactors),np.shape(States)[0],np.shape(States)[0]))
    for mu in Util.progressbar(range(len(normfactors)),"Coupling Constants:",40):
        #----------------------------------------------------------------------
        # Positively displaced 
        #----------------------------------------------------------------------
        folderplus='vector='+str(mu+1)+'sign=+'
        #Get the stompositions for the positively displaced atoms
        Atoms_Plus=Read.readinAtomicCoordinates(parentfolder+"/"+folderplus)
        KSH_alpha_Plus,KSH_beta_Plus,OLM_Plus=Read.readinMatrices(parentfolder+"/"+folderplus)
        T_Eq_Plus=AtomicBasis.getTransformationmatrix(Atoms_Eq,Atoms_Plus,Basis_Eq,cell_vectors)
        Sm1_Plus=Util.getSm1(OLM_Plus)
        T_Plus=Sm12_Eq@T_Eq_Plus@Sm1_Plus
        #Do Basis Transform
        KS_Plus=T_Plus@KSH_alpha_Plus@np.transpose(T_Plus)
        #----------------------------------------------------------------------
        # Negatively displaced 
        #----------------------------------------------------------------------
        folderminus='vector='+str(mu+1)+'sign=-'
        #Get the atom positions for the negatively displaced atoms
        Atoms_Minus=Read.readinAtomicCoordinates(parentfolder+"/"+folderminus)
        KSH_alpha_Minus,KSH_beta_Minus,OLM_Minus=Read.readinMatrices(parentfolder+"/"+folderminus)
        T_Eq_Minus=AtomicBasis.getTransformationmatrix(Atoms_Eq,Atoms_Minus,Basis_Eq,cell_vectors)
        Sm1_Minus=Util.getSm1(OLM_Minus)
        T_Minus=Sm12_Eq@T_Eq_Minus@Sm1_Minus
        #Do Basis Transform
        KS_minus=T_Minus@KSH_alpha_Minus@np.transpose(T_Minus)
        #Compute derivative
        derivativeKS=(KS_Plus-KS_minus)/(2*deltas[mu])*normfactors[mu]
        couplingConstants[mu,:,:]=ConFactors['E_H/a_0*hbar/sqrt(2*m_H)->cm^(3/2)']/(VibrationalFrequencies[mu])**(1.5)*np.transpose(States)@derivativeKS@States
    np.save(parentfolder+"/LCC_STATES",couplingConstants)
