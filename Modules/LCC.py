import numpy as np
import os
import Modules.Util as Util
import Modules.PhysConsts as PhysConst
import Modules.Read as Read
import Modules.AtomicBasis as AtomicBasis
import Modules.TDDFT as TDDFT

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
    orthorgonalEigenstates_Eq=[]
    for it in range(len(E_Eq)):
        orth_eigenstate=a_orth_Eq[:,it]
        orth_eigenstate*=TDDFT.getPhaseOfMO(orth_eigenstate)
        orthorgonalEigenstates_Eq.append(orth_eigenstate)
    _,delta=Read.readinBasisVectors(parentfolder)
    #get the normal modes from the cartesian displacements
    VibrationalFrequencies,_,normfactors=Read.readinVibrations(parentfolder)
    #Multiply by Tinv to get partialY_mu/partialX_lambda
    #This has index convention lambda,mu
    #*ConFactors['E_H/a_0*hbar/sqrt(2*m_H)->cm^(3/2)']/(VibrationalFrequencies[it])**(1.5) for it in range(len(M_salpha_timesX_salpha_lambda))
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
        Tmatrix_Plus=Sm12_Eq@T_Eq_Plus@Sm12_Plus
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
        T_Eq_Minus=AtomicBasis.getTransformationmatrix(Atoms_Eq,Atoms_Minus,Basis_Eq)
        T_Matrix_Minus=Sm12_Eq@T_Eq_Minus@Sm12_Minus
        #get the Eigenstates in the non-orthorgonal Basis
        orthorgonalEigenstates_Plus=[]
        for it in range(len(EPlus)):
            orth_eigenstate=a_orth_Plus[:,it]
            orth_eigenstate*=Util.getPhaseOfMO(orth_eigenstate)
            orthorgonalEigenstates_Plus.append(orth_eigenstate)
        #get the Eigenstates in the non-orthorgonal Basis
        orthorgonalEigenstates_Minus=[]
        for it in range(len(EMinus)):
            orth_eigenstate=a_orth_Minus[:,it]
            orth_eigenstate*=Util.getPhaseOfMO(orth_eigenstate)
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
