import unittest
import parabola as p
import numpy as np
import matplotlib.pyplot as plt
import scipy as sci
def getphasefromEV(eval):
    real_part=np.real(eval)
    imaginary_part=np.imag(eval)
    phase=0
    if real_part>=0:
        phase=np.arcsin(imaginary_part)
    else:
        phase=np.pi-np.arcsin(imaginary_part)
    if phase>np.pi:
        phase=phase-2*np.pi
    return phase
def determine_phases_from_Gamma_eigenstates(O,Es,T,tolerance=10**(-8)):
    tolerance=10**(-8)
    indexed_vals = sorted(enumerate(Es), key=lambda x: x[1])
    clusters = []
    current_cluster = [indexed_vals[0][0]]
    last_value = indexed_vals[0][1]

    for i in range(1, len(indexed_vals)):
        idx, val = indexed_vals[i]
        if abs(val - last_value) <= tolerance:
            current_cluster.append(idx)
        else:
            clusters.append(current_cluster)
            current_cluster = [idx]
        last_value = val
    clusters.append(current_cluster)  # Add the last group
    O_k=[]
    phases=[]
    energies=[]
    for pairs in clusters:
        TrasRep=np.zeros((len(pairs),len(pairs)))
        vectors=np.array([O[:,it] for it in pairs]).transpose()
        for it1 in range(len(pairs)):
            for it2 in range(len(pairs)):
                TrasRep[it1,it2]=np.dot(O[:,pairs[it1]],np.dot(T,O[:,pairs[it2]]))
        e,rot=np.linalg.eig(TrasRep)
        Oks=vectors@np.conj(rot)
        for it in range(np.shape(rot)[1]):
            phases.append(getphasefromEV(e[it])/np.pi)
            O_k.append(Oks[:,it])
    O_k=np.array(O_k).transpose()
    for it in range(np.shape(Es)[0]):
        energies.append(Es[it])
    return O_k,energies,phases
class TestBand_Unfolding(unittest.TestCase):
    def test_Linear_Chain(self):
        print("Started Linear_Chain")
        Conversion=p.PhysConst.ConversionFactors
        V=-0.025*Conversion["eV->a.u."]
        epsilon=0.05*Conversion["eV->a.u."]
        print("Transferintegral [a.u.]: V=",V)
        print("On-Site Energies [a.u.]: epsilon=",epsilon)
        cellvectors=p.Read.read_cell_vectors()
        cv=[0.0,0.0,0.0,cellvectors[0][0],0.0,0.0,-1*cellvectors[0][0],0.0,0.0]
        for it in range(len(cv)):
            cv[it] *= Conversion["A->a.u."]
        Basis={}
        Basis["H"]=[['1', '1', 's', [1, 2.5264751109842596]]]
        symmetry_structure = p.Symmetry.determineSymmetry("/media/max/SSD1/PHD/Data/CP2K/CP2K_Python_Modules/parabola/Tests/BandUnfolding_Tests/Linear_Chain/")
        generators = symmetry_structure.TranslationSymmetry_Generators
        print(len(generators))
        Atoms=p.Read.read_atomic_coordinates("/media/max/SSD1/PHD/Data/CP2K/CP2K_Python_Modules/parabola/Tests/BandUnfolding_Tests/Linear_Chain/")
        HKS=np.diag(V*np.ones(len(Atoms)-1),1)+np.diag(V*np.ones(len(Atoms)-1),-1)+np.diag(epsilon*np.ones(len(Atoms)))
        HKS[-1,0]=V
        HKS[0,-1]=V
        # Determine total number of basis functions
        total_functions = sum(len(Basis[atom[1]]) for atom in Atoms)

        # Initialize Translation Operator
        T = np.zeros((total_functions, total_functions))

        # Fill in transformation matrix T based on translation generators
        row_start = 0
        for i, atom1 in enumerate(Atoms):
            basis_size1 = len(Basis[atom1[1]])
            col_start = 0
            for j, atom2 in enumerate(Atoms):
                basis_size2 = len(Basis[atom2[1]])
                
                # Assuming one generator (first in the list)
                if np.array(generators[0])[i][j] > 1e-10:
                    T[row_start:row_start + basis_size1, col_start:col_start + basis_size2] = np.eye(basis_size2)

                col_start += basis_size2
            row_start += basis_size1
        
        S=p.AtomicBasis.getTransformationmatrix(Atoms, Atoms, Basis, cell_vectors=cv)
        Sm12=p.Util.LoewdinTransformation(S)
        H_ort=np.linalg.multi_dot([Sm12,HKS,Sm12])
        es,U=np.linalg.eigh(H_ort)
        O_k,Ek,phases=determine_phases_from_Gamma_eigenstates(U,es,T,tolerance=10**(-5))
        print(len(Ek))
        print(np.array(Ek)*Conversion["a.u.->eV"])
        print(len(phases))
        print(phases)
        plt.scatter(np.array(phases),np.array(Ek)*27.211)
        plt.grid()
        plt.show()
if __name__ == '__main__':
    unittest.main()

