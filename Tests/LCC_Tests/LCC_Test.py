import unittest
import Parabola as p
import numpy as np
import time 
import os
def getadibaticallyConnectedEigenstates_original(orthorgonalEigenstates_Eq,orthorgonalEigenstates_Plus,Tmatrix_Plus):
    adibaticallyConnectediters_Plus=[]
    #Get the adiabtically connected eigenvalues/states
    for it0 in range(np.shape(orthorgonalEigenstates_Eq)[0]):
        maximumAbsOverlap=0.0
        maximumOverlap=0.0
        iter1=-1
        for it1 in range(np.shape(orthorgonalEigenstates_Eq)[0]):
            overlap=np.dot(orthorgonalEigenstates_Eq[it0],Tmatrix_Plus@orthorgonalEigenstates_Plus[it1])
            absoverlap=np.abs(overlap)
            if absoverlap>maximumAbsOverlap:
                iter1=it1
                maximumOverlap=overlap
                maximumAbsOverlap=absoverlap
        adibaticallyConnectediters_Plus.append(iter1)
        if maximumOverlap<0:
            orthorgonalEigenstates_Plus[iter1]*=(-1.0)
        if maximumAbsOverlap<0.6:
            ValueError("Maximum Overlap small! Check your inputs!")
    return adibaticallyConnectediters_Plus

class TestLCC(unittest.TestCase):
    def test_getadibaticallyConnectedEigenstates(self):
        print("Started getadibaticallyConnectedEigenstates Test")
        Atoms_Eq=p.Read.readinAtomicCoordinates("./getadibaticallyConnectedEigenstates/Pyrazine_Test/Eq_Geo/")
        Basis_Eq=p.AtomicBasis.getBasis("./getadibaticallyConnectedEigenstates/Pyrazine_Test/Eq_Geo/")
        KSHamiltonian_Eq,S_Eq=p.Read.readinMatrices("./getadibaticallyConnectedEigenstates/Pyrazine_Test/Eq_Geo/")
        Sm12_Eq=p.Util.LoewdinTransformation(S_Eq)
        KSHorth_Eq=np.transpose(Sm12_Eq)@KSHamiltonian_Eq@Sm12_Eq
        E_Eq,a_orth_Eq=np.linalg.eigh(KSHorth_Eq)
        #get the normalized Eigenstates in the non-orthorgonal Basis & fix Phase
        orthorgonalEigenstates_Eq = []

        # Precompute phases
        phases = np.array([p.TDDFT.getPhaseOfMO(a_orth_Eq[:, it]) for it in range(len(E_Eq))])

        # Compute and append normalized eigenstates
        for it in range(len(E_Eq)):
            orth_eigenstate = a_orth_Eq[:, it].copy()  # Make a copy to avoid modifying the original data
            phase = phases[it]
            orth_eigenstate *= phase
            orthorgonalEigenstates_Eq.append(orth_eigenstate)

        KSHamiltonian_Plus,OLM_Plus=p.Read.readinMatrices('./getadibaticallyConnectedEigenstates/Pyrazine_Test/deflected_Geo/')
        #Get the stompositions for the positively displaced atoms
        Atoms_Plus=p.Read.readinAtomicCoordinates('./getadibaticallyConnectedEigenstates/Pyrazine_Test/deflected_Geo/')
        Sm12_Plus=p.Util.LoewdinTransformation(OLM_Plus)
        KSHorth_P=np.dot(Sm12_Plus,np.dot(KSHamiltonian_Plus,Sm12_Plus))
        EPlus,a_orth_Plus=np.linalg.eigh(KSHorth_P)
        T_Eq_Plus=p.AtomicBasis.getTransformationmatrix(Atoms_Eq,Atoms_Plus,Basis_Eq)
        Tmatrix_Plus=Sm12_Eq@T_Eq_Plus@Sm12_Plus
        orthorgonalEigenstates_Plus=[]
        for it in range(len(EPlus)):
            orth_eigenstate=a_orth_Plus[:,it]
            orth_eigenstate*=p.TDDFT.getPhaseOfMO(orth_eigenstate)
            orthorgonalEigenstates_Plus.append(orth_eigenstate)
        start_time1 = time.time()
        adibaticallyConnectediters_original=getadibaticallyConnectedEigenstates_original(orthorgonalEigenstates_Eq,orthorgonalEigenstates_Plus,Tmatrix_Plus)
        print("Original Implementation Timings: ", time.time()-start_time1)
        start_time2 = time.time()
        adibaticallyConnectediters_Parabola=p.LCC.getadiabaticallyConnectedEigenstates(orthorgonalEigenstates_Eq,orthorgonalEigenstates_Plus,Tmatrix_Plus)
        print("New Implementation Timings: ", time.time()-start_time2)
        np.testing.assert_array_almost_equal(adibaticallyConnectediters_Parabola, adibaticallyConnectediters_original,13,"adibaticallyConnectedEigenstates Test failed!")
        print("Finished getadibaticallyConnectedEigenstates Test")
    def test_getLinearCouplingConstants(self):
        print("Started getLinearCouplingConstants Test")
        parentfolder="./getLCC/Linear_Coupling_Constants/"
        LCC_Comparison=np.load(parentfolder+"Linear_Coupling_Constants_Comparison.npy")
        p.LCC.getLinearCouplingConstants(parentfolder,0,15)
        LCC_Test=np.load(parentfolder+"Linear_Coupling_Constants.npy")
        np.testing.assert_array_almost_equal(LCC_Test,LCC_Comparison,10,"adibaticallyConnectedEigenstates Test failed!")
        os.system("rm "+parentfolder+"/Linear_Coupling_Constant.npy")
        print("Finished getLinearCouplingConstants Test")
	
if __name__ == '__main__':
    unittest.main()

