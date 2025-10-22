from . import Read
from . import AtomicBasis
from . import Symmetry
from . import Util
from . import Geometry
from .PhysConst import ConversionFactors
import numpy as np
from scipy.linalg import schur
from itertools import permutations
import matplotlib.pyplot as plt
import os
pathtocp2k=os.environ["cp2kpath"]
pathtobinaries=pathtocp2k+"/exe/local/"
def get_tensor_representation():
    """
    Constructs real-valued Cartesian tensor representations of the spherical harmonics
    for s, p, d, f, and g orbitals.

    Each orbital is expressed as a sum of monomials (x^i y^j z^k) with a known prefactor.
    These are then converted into symmetric tensors of rank:
        - 0 for s (scalar)
        - 1 for p (vector)
        - 2 for d (symmetric 3×3)
        - 3 for f (symmetric 3×3×3)
        - 4 for g (symmetric 3×3×3×3)

    The resulting tensors are **unnormalized**, consistent with the radial wavefunction
    later correcting normalization for full atomic orbitals.

    Returns:
        representation_matrices (dict): Maps each orbital label (e.g. 'd+1', 'f-2') to its
        corresponding tensor representation as a NumPy array.
    """
    # Dictionary mapping orbital labels to monomial expansions:
    # Each entry is a list of [exponent_vector, prefactor]
    cs={}
    # --- s orbital (scalar) ---
    cs['s']=[[[0,0,0],0.5/np.sqrt(np.pi)]]
    # --- p orbitals (vector) ---
    cs['py']=[[[0,1,0],np.sqrt(3./(4.0*np.pi))]]
    cs['pz']=[[[0,0,1],np.sqrt(3./(4.0*np.pi))]]
    cs['px']=[[[1,0,0],np.sqrt(3./(4.0*np.pi))]]
    # --- d orbitals (symmetric 3×3 tensors) ---
    cs['d-2']=[[[1,1,0],0.5*np.sqrt(15./np.pi)]]
    cs['d-1']=[[[0,1,1],0.5*np.sqrt(15./np.pi)]]
    cs['d0']=[[[2,0,0],-0.25*np.sqrt(5./np.pi)],[[0,2,0],-0.25*np.sqrt(5./np.pi)],[[0,0,2],0.5*np.sqrt(5./np.pi)]]
    cs['d+1']=[[[1,0,1],0.5*np.sqrt(15./np.pi)]]
    cs['d+2']=[[[2,0,0],0.25*np.sqrt(15./np.pi)],[[0,2,0],-0.25*np.sqrt(15./np.pi)]]
    # --- f orbitals (symmetric 3×3×3 tensors) ---
    cs['f-3']=[[[2,1,0],0.75*np.sqrt(35./2./np.pi)],[[0,3,0],-0.25*np.sqrt(35./2./np.pi)]]
    cs['f-2']=[[[1,1,1],0.5*np.sqrt(105./np.pi)]]
    cs['f-1']=[[[0,1,2],np.sqrt(21./2./np.pi)],[[2,1,0],-0.25*np.sqrt(21./2./np.pi)],[[0,3,0],-0.25*np.sqrt(21./2./np.pi)]]
    cs['f0']=[[[0,0,3],0.5*np.sqrt(7./np.pi)],[[2,0,1],-0.75*np.sqrt(7/np.pi)],[[0,2,1],-0.75*np.sqrt(7/np.pi)]]
    cs['f+1']=[[[1,0,2],np.sqrt(21./2./np.pi)],[[1,2,0],-0.25*np.sqrt(21./2./np.pi)],[[3,0,0],-0.25*np.sqrt(21./2./np.pi)]]
    cs['f+2']=[[[2,0,1],0.25*np.sqrt(105./np.pi)],[[0,2,1],-0.25*np.sqrt(105./np.pi)]]
    cs['f+3']=[[[3,0,0],0.25*np.sqrt(35./2./np.pi)],[[1,2,0],-0.75*np.sqrt(35./2./np.pi)]]
    # --- g orbitals (symmetric 3×3×3×3 tensors) ---
    cs['g-4']=[[[3,1,0],0.75*np.sqrt(35./np.pi)],[[1,3,0],-0.75*np.sqrt(35./np.pi)]] 
    cs['g-3']=[[[2,1,1],9.0*np.sqrt(35./(2*np.pi))/4.0],[[0,3,1],-0.75*np.sqrt(35./(2.*np.pi))]] 
    cs['g-2']=[[[1,1,2],18.0*np.sqrt(5./(np.pi))/4.0],[[3,1,0],-3.*np.sqrt(5./(np.pi))/4.0],[[1,3,0],-3.*np.sqrt(5./(np.pi))/4.0]] 
    cs['g-1']=[[[0,1,3],3.0*np.sqrt(5./(2*np.pi))],[[2,1,1],-9.0*np.sqrt(5./(2*np.pi))/4.0],[[0,3,1],-9.0*np.sqrt(5./(2*np.pi))/4.0]] 
    cs['g0']=[[[0,0,4],3.0*np.sqrt(1./(np.pi))/2.0],[[4,0,0],9.0*np.sqrt(1./(np.pi))/16.0],[[0,4,0],9.0*np.sqrt(1./(np.pi))/16.0],[[2,0,2],-9.0*np.sqrt(1./np.pi)/2.0],[[0,2,2],-9.0*np.sqrt(1./np.pi)/2.0],[[2,2,0],9.0*np.sqrt(1./np.pi)/8.0]]
    cs['g+1']=[[[1,0,3],3.0*np.sqrt(5./(2*np.pi))],[[1,2,1],-9.0*np.sqrt(5./(2*np.pi))/4.0],[[3,0,1],-9.0*np.sqrt(5./(2*np.pi))/4.0]]
    cs['g+2']=[[[2,0,2],18.0*np.sqrt(5./(np.pi))/8.0],[[0,2,2],-18.*np.sqrt(5./(np.pi))/8.0],[[0,4,0],3.*np.sqrt(5./(np.pi))/8.0],[[4,0,0],-3.*np.sqrt(5./(np.pi))/8.0]]
    cs['g+3']=[[[1,2,1],-9.0*np.sqrt(35./(2*np.pi))/4.0],[[3,0,1],0.75*np.sqrt(35./(2.*np.pi))]]
    cs['g+4']=[[[4,0,0],3.0*np.sqrt(35./np.pi)/16.0],[[2,2,0],-18.0*np.sqrt(35./np.pi)/16.0],[[0,4,0],3.0*np.sqrt(35./np.pi)/16.0]]
    # Map monomial powers to i,j tensor indices
    def add_monomial(M_lambda, monomial, type):
        """Adds a monomial to the tensor by symmetrizing over index permutations."""
        powers = monomial[0]
        factor = monomial[1]
        indices = [0]*powers[0] + [1]*powers[1] + [2]*powers[2]
        perms = set(permutations(indices))

        weight = factor / len(perms)
        for p in perms:
            if type == "d":
                M_lambda[p[0], p[1]] += weight
            elif type == "f":
                M_lambda[p[0], p[1], p[2]] += weight
            elif type == "g":
                M_lambda[p[0], p[1], p[2], p[3]] += weight
        return M_lambda
    # Assemble tensor representations
    representation_matrices = {}
    for lm in cs:
        if lm[0] == "s":
            representation_matrices[lm] = cs[lm][0][1]  # scalar
        elif lm[0] == "p":
            representation_matrices[lm] = cs[lm][0][1] * np.array(cs[lm][0][0]) # vector
        elif lm[0] == "d":
            mat = np.zeros((3, 3))
            for mono in cs[lm]:
                mat = add_monomial(mat, mono, type="d") 
            representation_matrices[lm] = mat # 3x3 matrix
        elif lm[0] == "f":
            ten3 = np.zeros((3, 3, 3))
            for mono in cs[lm]:
                ten3 = add_monomial(ten3, mono, type="f")
            representation_matrices[lm] = ten3 # 3x3x3 tensor
        elif lm[0] == "g":
            ten4 = np.zeros((3, 3, 3, 3))
            for mono in cs[lm]:
                ten4 = add_monomial(ten4, mono, type="g")
            representation_matrices[lm] = ten4 # 3x3x3x3 tensor

    return representation_matrices
def get_basis_transformation_matrices_l_block(O):
    def scalar_product(tensor1,tensor2):
        rank=len(np.shape(tensor1))
        if rank>0:
            ranklist=[it for it in range(rank)]
            return np.tensordot(tensor1,tensor2,axes=(ranklist,ranklist))
        else:
            return tensor1*tensor2
    def apply_O_matrix(O,tensor):
        rank=len(np.shape(tensor))
        if rank>0:
            ranklist=[it for it in range(rank)]
            for it in ranklist:
                tensor=np.tensordot(O,tensor,axes=(1,it))
            return tensor
        else:
            return tensor

    canonical_ordering={}
    canonical_ordering["s"]=["s"]
    canonical_ordering["p"]=["py","pz","px"]
    canonical_ordering["d"]=["d-2","d-1","d0","d+1","d+2"]
    canonical_ordering["f"]=["f-3","f-2","f-1","f0","f+1","f+2","f+3"]
    canonical_ordering["g"]=["g-4","g-3","g-2","g-1","g0","g+1","g+2","g+3","g+4"]


    representation_matrices=get_tensor_representation()
    transformation_matrices={}
    for l in ["s","p","d","f","g"]:
        transformation_matrix=np.zeros((len(canonical_ordering[l]),len(canonical_ordering[l])))
        for it1 in range(len(canonical_ordering[l])):
            for it2 in range(len(canonical_ordering[l])):
                B1=representation_matrices[canonical_ordering[l][it1]]
                B2=representation_matrices[canonical_ordering[l][it2]]
                O_B2=apply_O_matrix(O,B2)
                overlap=scalar_product(B1,O_B2)
                norm=scalar_product(B1,B1)
                transformation_matrix[it1][it2]=overlap/norm
        transformation_matrices[l]=transformation_matrix
    return transformation_matrices

def get_l_ordering(Basis):
    ordering={}
    for atom in Basis:
        ordering_atom=[]
        ordering_atom.append(Basis[atom][0][2][0])
        for it in range(1,len(Basis[atom])):
            if (Basis[atom][it][2][0]!=Basis[atom][it - 1][2][0] or Basis[atom][it][1]!=Basis[atom][it-1][1]) or Basis[atom][it][0]!=Basis[atom][it-1][0]:
                ordering_atom.append(Basis[atom][it][2][0])
        ordering[atom]=ordering_atom
    return ordering
def get_l_sizes(label):
    if label=="s":
        return 1
    elif label=="p":
        return 3
    elif label=="d":
        return 5
    elif label=="f":
        return 7
    elif label=="g":
        return 9
    else:
        ValueError("Higher labels not yet implemented")
def get_index_block_start(atom_index,atoms,l_ordering):
    start_index=0
    for idx,atom in enumerate(atoms):
        if idx<atom_index:
            for order in l_ordering[atom]:
                start_index+=get_l_sizes(order)
    return start_index
        
def get_basis_transformation(O,P,atoms,basis):
    basis_transformation_matrices_l_block=get_basis_transformation_matrices_l_block(O)
    l_ordering=get_l_ordering(basis)
    basis_size=0
    for atom in atoms:
        for order in l_ordering[atom]:
            basis_size+=get_l_sizes(order)
    basis_transformation=np.zeros((basis_size,basis_size))
    nonzero_indices = np.nonzero(P)
    for idx1, idx2 in zip(nonzero_indices[0], nonzero_indices[1]):
        start_index_1=get_index_block_start(idx1,atoms,l_ordering)
        start_index_2=get_index_block_start(idx2,atoms,l_ordering)
        for l in l_ordering[atoms[idx1]]:
            block=basis_transformation_matrices_l_block[l]
            basis_transformation[start_index_1:start_index_1+np.shape(block)[0],start_index_2:start_index_2+np.shape(block)[1]]=block
            start_index_1+=np.shape(block)[0]
            start_index_2+=np.shape(block)[1]
    return basis_transformation








        
class ElectronicStructure:
    def __init__(self, mol):
        print("    -> Initializing ElectronicStructure object...")
        # --- This is the Back-Reference ---
        # Store a reference to the parent MolecularStructure object. 
        if hasattr(mol, 'KS_Hamiltonian_alpha'):
            return
        self.mol_path=mol.path
        self.basis=AtomicBasis.getBasis(mol.electronic_structure_path)
        num_e,charge=Read.get_number_of_electrons(parentfolder=mol.electronic_structure_path)
        self.num_e=num_e
        self.charge=charge
        self.multiplicity=Read.read_multiplicity(path=mol.electronic_structure_path)
        KS_alpha,KS_beta,OLM=Read.read_ks_matrices(mol.electronic_structure_path)
        self.UKS=Read.check_uks(mol.electronic_structure_path)
        self.KS_Hamiltonian_alpha=KS_alpha
        self.KS_Hamiltonian_beta=KS_beta
        self.OLM=OLM
        cond_number=np.linalg.cond(OLM)
        if cond_number>10**(6):
            Warning("Large Condition number of the Overlap Matrix! Cond(OLM)="+str(cond_number))
        self.inverse_sqrt_OLM=Util.LoewdinTransformation(OLM,algorithm='Schur-Pade')
        self.real_eigenstates={}
        self.energies={}
        self.indexmap={}
        self.indexmap["alpha"]={}
        self.real_eigenstates["alpha"]={}
        self.energies["alpha"]={}
        if self.UKS:
            self.indexmap["beta"]={}
            self.real_eigenstates["beta"]={}
            self.energies["beta"]={}
        # Attach method dynamically
        atoms=mol.atoms
        self.electronic_symmetry = electronic_symmetry(mol.Molecular_Symmetry,atoms,self.basis)
        name=mol.name
        axes=mol.Geometric_UC_Principle_Axis
        self.get_real_electronic_eigenstates(name,atoms,axes)
        symmetry_labels=self.electronic_symmetry.Symmetry_Generators.keys()
        if "t1" in symmetry_labels or "t2" in symmetry_labels or "t3" in symmetry_labels:
            self.bloch_states={}

        
    def get_real_electronic_eigenstates(self,name,atoms,axes):
        def TransformHamiltonian(self,atoms,axes):
            U=get_basis_transformation(np.array(axes).T,np.eye(len(atoms)),atoms,self.basis)
            if self.UKS:
                KS_Hamiltonian_alpha=self.KS_Hamiltonian_alpha
                KS_Hamiltonian_beta=self.KS_Hamiltonian_beta
                KS_Hamiltonian_alpha=U.T@KS_Hamiltonian_alpha@U
                KS_Hamiltonian_beta=U.T@KS_Hamiltonian_beta@U
            else:
                KS_Hamiltonian_alpha=self.KS_Hamiltonian_alpha
                KS_Hamiltonian_alpha=U.T@KS_Hamiltonian_alpha@U
                KS_Hamiltonian_beta=KS_Hamiltonian_alpha
            return KS_Hamiltonian_alpha,KS_Hamiltonian_beta,U
        print(f"ℹ️ : Calculating electronic eigenstates for {name}")
        # --- Setup and Hessian Transformation ---
        KS_Hamiltonian_alpha,KS_Hamiltonian_beta,U=TransformHamiltonian(self,atoms,axes)
        sqrtSm1 = self.inverse_sqrt_OLM
        sqrtSm1=U.T@sqrtSm1@U
        KS_Hamiltonian_alpha_orth = np.real(sqrtSm1 @ KS_Hamiltonian_alpha @ sqrtSm1)
        if self.UKS:
            KS_Hamiltonian_beta_orth = sqrtSm1 @ KS_Hamiltonian_beta @ sqrtSm1
        
        
        # (Optional) Visualize the original mass-weighted Hessian
        plt.imshow(KS_Hamiltonian_alpha_orth, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.savefig("./Hamiltonian_alpha_orth_Original.png")
        plt.close()

        if self.UKS:
            # (Optional) Visualize the original mass-weighted Hessian
            plt.imshow(KS_Hamiltonian_beta_orth, cmap='viridis', interpolation='nearest')
            plt.colorbar()
            plt.savefig("./Hamiltonian_beta_orth_Original.png")
            plt.close()
        
        # --- CONSISTENT SYMMETRY ORDERING ---
        SymSectors = self.electronic_symmetry.SymSectors
        VIrr = self.electronic_symmetry.IrrepsProjector
        
        # 1. Sort symmetry labels alphabetically to ensure a consistent order.
        sorted_sym_labels = sorted(SymSectors.keys())
        
        # 2. Build the reordering array based on the sorted labels.
        # This groups basis functions by symmetry, in a fixed order.
        reordering = np.concatenate([SymSectors[key] for key in sorted_sym_labels])
        
        # 3. Reorder the projector matrix columns based on the sorted symmetry order.
        VIrr_reordered = VIrr[:, reordering]
        
        # 4. Create the block-diagonal Hessian. The blocks are now in a consistent order.
        KS_Hamiltonian_alpha_orth_Sectors = VIrr_reordered.T @ KS_Hamiltonian_alpha_orth @ VIrr_reordered
        if self.UKS:
            KS_Hamiltonian_beta_orth_Sectors = VIrr_reordered.T @ KS_Hamiltonian_beta_orth @ VIrr_reordered
        
        # (Optional) Visualize the block-diagonalized Hessian
        plt.imshow(KS_Hamiltonian_alpha_orth_Sectors, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.savefig("./Hamiltonian_alpha_orth_Sectors.png")
        plt.close()
        if self.UKS:
            plt.imshow(KS_Hamiltonian_alpha_orth_Sectors, cmap='viridis', interpolation='nearest')
            plt.colorbar()
            plt.savefig("./Hamiltonian_beta_orth_Sectors.png")
            plt.close()
        
        # --- BLOCK DIAGONALIZATION (IN CONSISTENT ORDER) ---
        print(f"ℹ️ : Symmetry Sectors and Electronic Energies")
        current_index = 0
        # 5. Iterate through the sorted labels to process each block.
        label_alpha=[]
        energy_alpha=[]
        if self.UKS:
            label_beta=[]
            energy_beta=[]
        # 5. Iterate through the sorted labels to process each block.
        for sym in sorted_sym_labels:
            block_size = len(SymSectors[sym])
            
            # Extract the symmetry block
            Block_Hamiltonian_alpha = KS_Hamiltonian_alpha_orth_Sectors[current_index : current_index + block_size,
                                                        current_index : current_index + block_size]
            if self.UKS:
                Block_Hamiltonian_beta = KS_Hamiltonian_beta_orth_Sectors[current_index : current_index + block_size,
                                                        current_index : current_index + block_size]
            
            # Diagonalize the block to get eigenvalues and eigenvectors for this symmetry
            eigenvalues_alpha, eigenvectors_alpha = np.linalg.eigh(Block_Hamiltonian_alpha)

            
            # energies in eV
            energies_alpha_eV = eigenvalues_alpha*27.211
            print(f"Symmetry Sector for Spin Species ↑: {sym}, Energies (eV): \n {energies_alpha_eV}")
            
            # Transform eigenvectors from the symmetry basis back to the mass-weighted basis
            V_mwh_alpha = U@VIrr_reordered[:, current_index : current_index + block_size] @ eigenvectors_alpha
            if self.UKS:
                 # Diagonalize the block to get eigenvalues and eigenvectors for this symmetry
                eigenvalues_beta, eigenvectors_beta = np.linalg.eigh(Block_Hamiltonian_beta)
                
                # energies (eV) 
                energies_beta_eV = eigenvalues_beta*27.211
                print(f"Symmetry Sector for Spin Species ↓: {sym}, Energies (eV): \n {energies_beta_eV}")
                
                # Transform eigenvectors from the symmetry basis back 
                V_mwh_beta = U@VIrr_reordered[:, current_index : current_index + block_size] @ eigenvectors_beta
            
            V_mwh_alpha=fix_phases_states(V_mwh_alpha, threshold=1e-10)
            self.real_eigenstates["alpha"][sym]=V_mwh_alpha
            self.energies["alpha"]=energies_alpha_eV
            if self.UKS:
                V_mwh_beta=fix_phases_states(V_mwh_beta, threshold=1e-10)
                self.real_eigenstates["beta"][sym].append(V_mwh_beta)
                self.energies["beta"]=energies_alpha_eV

            for i in range(block_size):
                label_alpha.append((sym,i))
                energy_alpha.append(eigenvalues_alpha[i])
                if self.UKS:
                    label_beta.append((sym,i))
                    energy_beta.append(eigenvalues_beta[i])
            # Advance the index to the start of the next block
            current_index += block_size
        if not self.UKS:
            sorted_indices=np.argsort(energy_alpha)
            label_alpha=np.array(label_alpha)[sorted_indices]
            Homoindex=self.num_e/2
            for id,element in enumerate(label_alpha):
                self.indexmap["alpha"][int(id+1-Homoindex)]=element

def test_phase_op(mol):
    # Initialize dictionary to track periodicity
    periodic = {'x': False, 'y': False, 'z': False}

    # Loop through unit cells and mark directions as periodic if any nonzero value found
    for uc in mol.unitcells.keys():
        periodic['x'] |= uc[0] != 0
        periodic['y'] |= uc[1] != 0
        periodic['z'] |= uc[2] != 0

    # Get list of periodic directions
    periodic_dirs = [axis for axis, is_periodic in periodic.items() if is_periodic]
    print(periodic_dirs)
    # Count how many directions are periodic
    num_periodic = len(periodic_dirs)
    
    # get reciprocal lattice vectors
    if num_periodic==1:
        q_points=get_q_points(mol.cellvectors,mol.periodicity)
        delta_q=np.array(q_points[1])-np.array(q_points[0])
        if periodic_dirs[0]=="x":
            m=2;n=0;l=0
        elif periodic_dirs[0]=="y":
            m=0;n=2;l=0
        else:
            m=0;n=0;l=2
        cellvectors=Geometry.getNeibouringCellVectors(cell=mol.cellvectors,m=m,n=n,l=l)
        atoms=Read.read_atomic_coordinates(mol.xyz_path)
        basis=mol.electronic_structure.basis
        phi_q=AtomicBasis.get_phase_operators(atoms,basis,q_vector=list(delta_q),cell_vectors=cellvectors)
        Sm12=mol.electronic_structure.inverse_sqrt_OLM
        phase_1 = np.linalg.multi_dot([Sm12, phi_q, Sm12])
        
        #get gamma point symmetry sectors
        if periodic_dirs[0]=="x":
            gamma_label="Id=1|t1=1|"
        elif periodic_dirs[1]=="y":
            gamma_label="Id=1|t2=1|"
        else:
            gamma_label="Id=1|t3=1|"
        gamma_sym_sectors={}
        other_sym_sectors={}
        #Closed Shell case
        if len(mol.electronic_structure.real_eigenstates.keys())==1:
            for sym_sector in mol.electronic_structure.real_eigenstates["alpha"].keys():
                if len(sym_sector.split(gamma_label))==2:
                    gamma_sym_sectors[sym_sector]=np.array([mol.electronic_structure.real_eigenstates["alpha"][sym_sector][it].A for it in range(len(mol.electronic_structure.real_eigenstates["alpha"][sym_sector]))]).T
                else:
                    other_sym_sectors[sym_sector]=np.array([mol.electronic_structure.real_eigenstates["alpha"][sym_sector][it].A for it in range(len(mol.electronic_structure.real_eigenstates["alpha"][sym_sector]))]).T
            #compute overlaps
            for sym_sector1 in other_sym_sectors.keys():
                for sym_sector2 in gamma_sym_sectors.keys():
                    OLP=np.dot(other_sym_sectors[sym_sector1].T,np.dot(phase_1,gamma_sym_sectors[sym_sector2]))
                    print(sym_sector1,sym_sector2,np.max(np.abs(OLP)))

    '''
    basis=mol.electronic_structure.basis
    xyz_filepath=Read.get_xyz_filename("./")
    atoms=Read.read_atomic_coordinates(xyz_filepath)
    
    print(q_points)
    
    s0=mol.electronic_structure.real_eigenstates["alpha"]["Id=1|t1=1|Sz=2"][0]
    Sm12=mol.electronic_structure.inverse_sqrt_OLM
      # Adjust this line as needed
    
    
    phase_1 = np.linalg.multi_dot([Sm12, phi_q, Sm12])
    '''
    """
    for q_point in q_points:
        phi_q=AtomicBasis.get_phase_operators(atoms,basis,q_vector=q_point,cell_vectors=cellvectors)
        phase_1 = np.linalg.multi_dot([Sm12, phi_q, Sm12])
        s0_transformed = phase_1 @ s0.A
        print(q_point)
        for sym_sector in mol.electronic_structure.real_eigenstates["alpha"]:
            for it,state in enumerate(mol.electronic_structure.real_eigenstates["alpha"][sym_sector]):
                value=np.dot(state.A,s0_transformed)
                if np.abs(value)>0.01:
                    print(sym_sector,it,value,np.abs(value)*np.sqrt(2))
    """
def band_index(mol,nh,nl,path='./',threshold=0.25):

    # Figuring out the translational symmetries in the structure:
    p_gamma_ind = []
    periodic = np.array(mol.periodicity) > 1
    periodic = periodic.astype(int)

    translate = []
    for dir in range(periodic.shape[0]):
        if periodic[dir] == 1:
            translate.append('t' + str(dir + 1) + '=1')

    # Isolating Symmetry sectors that correspond to the primitive gamma point
    symsecs = list(mol.electronic_structure.Electronic_Symmetry.SymSectors.keys())
    prim_gam_sym = symsecs.copy()
    for sym in symsecs:
        if 'Id=1' in sym:
            for t in translate:
                if not t in sym:
                    prim_gam_sym.remove(sym)
        elif 'Id=1' not in sym:
            prim_gam_sym.remove(str(sym))

    non_prim_gam_sym = list(set(symsecs) - set(prim_gam_sym))

    # Isolating the states that correspond to the primitive gamma point

    estate_dict = mol.electronic_structure.indexmap['alpha']
    finalind = next(iter(estate_dict))
    occ_ind = -1 * np.arange(0, -1 * finalind + 1)
    occ_prim_gam_states = []
    if nh > 0:
        for ind in occ_ind:
            if estate_dict[ind][0] in prim_gam_sym:
                occ_prim_gam_states.append(estate_dict[ind])
                if len(occ_prim_gam_states) == nh:
                    break

    unocc_prim_gam_states = []
    if nl > 0:
        for ind in range(0, int(len(estate_dict) - np.shape(occ_ind)[0])):
            if estate_dict[ind][0] in prim_gam_sym:
                unocc_prim_gam_states.append(estate_dict[ind])
                if len(unocc_prim_gam_states) == nl:
                    break

    prim_gam_states = unocc_prim_gam_states + occ_prim_gam_states

    basis=mol.electronic_structure.Basis
    xyz_filepath=Read.get_xyz_filename(path)
    atoms=Read.read_atomic_coordinates(xyz_filepath)
    q_points=get_q_points(mol.cellvectors,mol.periodicity)

    cellvectors = Geometry.getNeibouringCellVectors(cell=mol.cellvectors, m=periodic[0], n=periodic[1], l=periodic[2])
    # assuming the supercell is large enough so that just the first neighbouring cells are enough; otherwise a convergence check would be needed!

    all_connected_bands = []
    g_point = q_points.index([0.0,0.0,0.0])

    for state in prim_gam_states:
        print(state,mol.electronic_structure.ElectronicEigenstates["alpha"][state[0]][int(state[1])].energy)
        band = {}
        band[tuple(q_points[g_point])] = state
        s0 = mol.electronic_structure.ElectronicEigenstates["alpha"][state[0]][int(state[1])]
        Sm12 = mol.electronic_structure.inverse_sqrt_OLM
        for q_point in q_points:
            print(q_point)
            phi_q = AtomicBasis.get_phase_operators(atoms, basis, q_vector=q_point, cell_vectors=cellvectors)
            phase_1 = np.linalg.multi_dot([Sm12, phi_q, Sm12])
            s0_transformed = phase_1 @ s0.A
            q_state_list = []
            for sym_sector in non_prim_gam_sym:
                for it, state in enumerate(mol.electronic_structure.ElectronicEigenstates["alpha"][sym_sector]):
                    value = np.dot(state.A, s0_transformed)
                    if np.abs(value) > 0.25:
                        print(np.array([str(sym_sector), str(it)]), value, np.abs(value) * np.sqrt(2), state.energy-s0.energy)
                        q_state_list.append(np.array([str(sym_sector), str(it)]))

            band[tuple(q_point)] = q_state_list
        print(len(band))
        all_connected_bands.append(band)


    return all_connected_bands #states#band_indices
def get_reciprocal_lattice_vectors(a1, a2, a3):
        """
        Calculate reciprocal lattice vectors from direct lattice vectors.
        
        Parameters:
        -----------
        a1, a2, a3 : array-like, shape (3,)
            Direct lattice vectors in real space
            
        Returns:
        --------
        b1, b2, b3 : numpy.ndarray, shape (3,)
            Reciprocal lattice vectors
            
        Notes:
        ------
        The reciprocal lattice vectors are defined by:
        b1 = 2π * (a2 × a3) / (a1 · (a2 × a3))
        b2 = 2π * (a3 × a1) / (a1 · (a2 × a3))  
        b3 = 2π * (a1 × a2) / (a1 · (a2 × a3))
        
        The factor of 2π ensures that aᵢ · bⱼ = 2π δᵢⱼ
        """
        # Convert to numpy arrays
        a1 = np.array(a1, dtype=float)
        a2 = np.array(a2, dtype=float)
        a3 = np.array(a3, dtype=float)
        
        # Calculate the volume of the unit cell (scalar triple product)
        volume = np.dot(a1, np.cross(a2, a3))
        
        if np.abs(volume) < 1e-12:
            raise ValueError("Lattice vectors are coplanar (volume = 0). Cannot form 3D lattice.")
        
        # Calculate reciprocal lattice vectors
        b1 = 2 * np.pi * np.cross(a2, a3) / volume
        b2 = 2 * np.pi * np.cross(a3, a1) / volume
        b3 = 2 * np.pi * np.cross(a1, a2) / volume
        
        return b1, b2, b3
def incl_kpoints(N):
    klist = []
    for kx in range(-int(np.floor(N/2)),int(np.ceil(N/2))):
        k = (kx/N)
        klist.append(k)
    return klist
    
def get_q_points(cellvectors,periodicity,return_format="combined"):
    """
    Generate k-point grid in reciprocal space for periodic systems.
    
    Parameters:
    -----------
    mol : molecule object
        Must have attributes: cellvectors, periodicity
    return_format : str, optional
        - 'combined': returns all k-points as single list (default)
        - 'separate': returns separate lists for each direction
        
    Returns:
    --------
    Depends on return_format:
    - 'combined': list of k-point vectors [k1+k2+k3 combinations]
    - 'separate': tuple (qs1, qs2, qs3) of individual direction k-points
    """
    
    
    # Calculate primitive cell vectors
    primitive_cell_vectors = [cellvectors[i] / periodicity[i] for i in range(3)]
    
    # Get reciprocal lattice vectors
    b1, b2, b3 = get_reciprocal_lattice_vectors(*primitive_cell_vectors)
    
    # Generate symmetric points for each direction
    #k1s = symmetric_points(periodicity[0])
    #k2s = symmetric_points(periodicity[1])
    #k3s = symmetric_points(periodicity[2])
    #print('old',k1s,k2s,k3s)
    # Generate q_points in fractional coordinates (more general alternative)
    k1s = incl_kpoints(periodicity[0])
    k2s = incl_kpoints(periodicity[1])
    k3s = incl_kpoints(periodicity[2])
    # Convert to reciprocal space coordinates with proper units
    conversion_factor = ConversionFactors["a.u.->A"]
    
    
    # Individual direction k-points
    qs1 = [list(k1s[i] * b1 * conversion_factor) for i in range(len(k1s))]
    qs2 = [list(k2s[i] * b2 * conversion_factor) for i in range(len(k2s))]
    qs3 = [list(k3s[i] * b3 * conversion_factor) for i in range(len(k3s))]
    
    if return_format == 'separate':
        return qs1, qs2, qs3
    
    elif return_format == 'combined':
        # Return all combinations of k-points (original behavior + generalizations)
        q_points = []
        for k1 in k1s:
            for k2 in k2s:
                for k3 in k3s:
                    q_vector = (k1 * b1 + k2 * b2 + k3 * b3) * conversion_factor
                    q_points.append(list(q_vector))
        return q_points
    
    else:
        raise ValueError(f"Unknown return_format: {return_format}. "
                        "Choose from 'combined', 'separate', 'grid', or 'mesh'.")     
    

def fix_phases_states(V, threshold=1e-10):
    """
    A vectorized function to fix the phase of state vectors.

    Args:
        V (np.ndarray): A 2D array where each column is a state vector.
        threshold (float): A small value to check if the sum is close to zero.

    Returns:
        np.ndarray: The array V with phases of columns adjusted in-place.
    """
    # 1. Calculate the sum of each column all at once
    column_sums = np.sum(V, axis=0)

    # 2. Condition 1: Find columns where sum is strongly negative
    mask1 = column_sums < -threshold

    # 3. Condition 2: Find columns where sum is close to zero
    mask_zero_sum = np.abs(column_sums) <= threshold
    
    # For columns with a near-zero sum, check the sign of the largest component
    if np.any(mask_zero_sum):
        # Get row indices of the max absolute value for each column
        max_abs_indices = np.argmax(np.abs(V), axis=0)
        
        # Get the actual values at those indices
        # V[max_abs_indices, np.arange(V.shape[1])] creates a 1D array of these values
        dominant_values = V[max_abs_indices, np.arange(V.shape[1])]
        
        # Create a mask for columns where the dominant value is negative
        # and the sum is near-zero.
        mask2 = mask_zero_sum & (dominant_values < 0)
    else:
        # Avoid unnecessary calculations if no columns have a near-zero sum
        mask2 = False

    # 4. Combine masks: flip a column if either condition is met
    final_mask_to_flip = mask1 | mask2

    # 5. Apply the flip to all selected columns in one operation
    V[:, final_mask_to_flip] *= -1.0
    
    return V
        
        


        

    
    



    
    
class electronic_symmetry(Symmetry.Symmetry):
    def __init__(self,molecular_symmetry,atoms,basis):
        super().__init__()  # Initialize parent class
        if "Id" not in molecular_symmetry.Symmetry_Generators.keys():
            generators={}
            for sym in molecular_symmetry.Symmetry_Generators:
                P=molecular_symmetry.Symmetry_Generators[sym]
                O=get_xyz_representation(symmetrylabel=sym)
                generator=get_basis_transformation(O,P,atoms,basis)
                generators[sym]=generator
            self.Symmetry_Generators=generators
            self._iscommutative()
            if self.commutative:
                self.IrrepsProjector=simultaneous_real_block_diagonalization(list(self.Symmetry_Generators.values()))
            else:
                self._determineCentralizer()
                self._determineIrrepsProjector()
        else:
            self.IrrepsProjector=np.kron(molecular_symmetry.Symmetry_Generators["Id"],np.eye(3))
        self._determineSymmetrySectors()
#### Define Symmetry Class ####
def detect_block_sizes(matrix, tol=1e-8):
        """
        Detects block sizes in a (approximately) block-diagonal square matrix.
        Args:
            matrix: (n x n) NumPy array (assumed square and block-diagonal).
            tol: threshold below which off-diagonal elements are considered zero.

        Returns:
            A list of (start_index, block_size) tuples.
        """
        n = matrix.shape[0]
        assert matrix.shape[0] == matrix.shape[1], "Matrix must be square."
        blocks = []
        i = 0

        while i < n:
            block_found = False
            for size in range(1, n - i + 1):
                # Extract candidate block
                block = matrix[i:i+size, i:i+size]

                # Check if it's isolated: off-block rows/cols should be zero
                off_block = matrix[i:i+size, i+size:]
                off_block_T = matrix[i+size:, i:i+size]

                if np.all(np.abs(off_block) < tol) and np.all(np.abs(off_block_T) < tol):
                    # Check if next row/col introduces new coupling
                    if i + size == n:
                        blocks.append((i, size))
                        i += size
                        block_found = True
                        break
                    next_col = matrix[i:i+size, i+size]
                    next_row = matrix[i+size, i:i+size]
                    if np.all(np.abs(next_col) < tol) and np.all(np.abs(next_row) < tol):
                        blocks.append((i, size))
                        i += size
                        block_found = True
                        break
            if not block_found:
                # Fallback: treat single diagonal element as a block
                blocks.append((i, 1))
                i += 1

        return blocks
def simultaneous_real_block_diagonalization(matrices):
    """
    Block-diagonalize a set of mutually commuting real matrices using a single real orthogonal basis.
    
    The matrices are assumed to commute and be real-valued. This function finds a single
    real orthogonal matrix Q that transforms every matrix A in the input list into a
    block-diagonal matrix Q.T @ A @ Q. The blocks are 1x1 for real eigenvalues
    and 2x2 for complex-conjugate eigenvalue pairs.

    Parameters:
    -----------
    matrices : list of np.ndarray
        A list of real-valued, square, mutually commuting matrices of the same shape (n x n).

    Returns:
    --------
    Q : np.ndarray
        The real orthogonal matrix (n x n) that simultaneously block-diagonalizes all matrices.

    transformed_matrices : list of np.ndarray
        The list of transformed (block-diagonal) matrices.
    """
    if not matrices:
        raise ValueError("Input list of matrices cannot be empty.")

    n = matrices[0].shape[0]
    for A in matrices:
        if A.shape != (n, n):
            raise ValueError("All matrices must be square and have the same shape.")
        # This check is good practice, though scipy.linalg.schur can handle complex inputs.
        if not np.allclose(A, A.real):
            raise ValueError("All matrices must be real-valued.")

    # 1. Create a random linear combination of the matrices.
    # Since all matrices commute, any linear combination of them also commutes with them.
    # A random combination is very likely to have distinct eigenvalues, which simplifies
    # the identification of the common eigenspaces.
    C = np.zeros((n, n))
    for A in matrices:
        C += np.random.randn() * A

    # 2. Compute the real Schur decomposition of the combined matrix C.
    # The schur function returns an orthogonal matrix Q and a block-upper-triangular
    # matrix T (the "real Schur form") such that C = Q @ T @ Q.T.
    # This matrix Q is the transformation we need.
    _, Q = schur(C, output='real')
    return Q
    
    
def get_xyz_representation(symmetrylabel):
    """
    Return the 3 × 3 Cartesian (XYZ) matrix representation of a basic
    point‑symmetry operation.

    The function currently recognises **inversion, proper rotations,
    mirror reflections** and a trivial identity operation, using the
    following label grammar:

    ──────────────────────────────────────────────────────────────────────
    Label       Meaning                               Returned matrix
    ──────────────────────────────────────────────────────────────────────
    "i"         Inversion through the origin          −I₃
    "C<axis>_<n>"
                n‑fold proper rotation about          R_axis(2π / n)
                the given axis (x, y or z)            (right‑hand rule)
    "S<axis>"   Mirror (σ) plane normal to <axis>     diag(±1, ±1, ±1)
    "t"         Identity (useful placeholder)         I₃
    ──────────────────────────────────────────────────────────────────────

    Parameters
    ----------
    symmetrylabel : str
        A string encoded as described above.
        Examples: "i", "Cx_2", "Cy_4", "Sz", "t".

    Returns
    -------
    numpy.ndarray
        A 3 × 3 `float64` NumPy array representing the operation in
        Cartesian coordinates.

    Raises
    ------
    ValueError
        If *symmetrylabel* does not conform to any of the supported
        patterns.

    Notes
    -----
    * **Right‑hand convention** – Positive rotation angles follow the
      right‑hand rule about the specified axis.
    * Rotations are constructed with ``theta = 2π / n`` (radians), so
      ``Cx_2`` is a 180 ° (π) rotation, ``Cz_4`` is a 90 ° (π/2) rotation,
      etc.
    * The mirror (“S”) labels here implement *simple* reflections, not
      roto‑reflections; extend as needed for improper rotations **Sₙ**.

    Examples
    --------
    >>> getXYZRepresentation("i")
    array([[-1.,  0.,  0.],
           [ 0., -1.,  0.],
           [ 0.,  0., -1.]])

    >>> getXYZRepresentation("Cy_4")        # 90° rotation about y
    array([[ 0. ,  0. ,  1. ],
           [ 0. ,  1. ,  0. ],
           [-1. ,  0. ,  0. ]])

    >>> getXYZRepresentation("Sz")          # mirror in xy‑plane
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0., -1.]])
    """
    if symmetrylabel=="i":
        return (-1.0)*np.eye(3)
    elif symmetrylabel[0]=="C":
        theta=2*np.pi/int(symmetrylabel.split("_")[1])
        if symmetrylabel[1]=="x":
            return np.array([[1, 0, 0],[0, np.cos(theta), -np.sin(theta)],[0, np.sin(theta),  np.cos(theta)]]).T
        elif symmetrylabel[1]=="y":
            return np.array([[np.cos(theta), 0, np.sin(theta)],[0, 1, 0],[-np.sin(theta), 0, np.cos(theta)]]).T
        elif symmetrylabel[1]=="z":
            return np.array([[np.cos(theta), -np.sin(theta), 0],[np.sin(theta),  np.cos(theta), 0],[0, 0, 1]]).T
    elif symmetrylabel[0]=="S":
        if symmetrylabel[1]=="x":
            return np.array([[-1, 0, 0],[0, 1, 0],[0, 0, 1]])
        elif symmetrylabel[1]=="y":
            return  np.array([[1, 0, 0],[0, -1, 0],[0, 0, 1]])
        elif symmetrylabel[1]=="z":
            return np.array([[1, 0, 0],[0, 1, 0],[0, 0, -1]])
    elif symmetrylabel[0]=="t":
        return np.eye(3)
def test_IrrepsProjector(name):
    el=ElectronicStructure(name)
    SymSectors = el.electronic_symmetry.SymSectors
    VIrr = el.electronic_symmetry.IrrepsProjector
    
    # 1. Sort symmetry labels alphabetically to ensure a consistent order.
    sorted_sym_labels = sorted(SymSectors.keys())
    
    # 2. Build the reordering array based on the sorted labels.
    # This groups basis functions by symmetry, in a fixed order.
    reordering = np.concatenate([SymSectors[key] for key in sorted_sym_labels])
    
    # 3. Reorder the projector matrix columns based on the sorted symmetry order.
    VIrr_reordered = VIrr[:, reordering]
    for sym in el.electronic_symmetry.Symmetry_Generators:
        symm=VIrr_reordered.T@el.electronic_symmetry.Symmetry_Generators[sym]@VIrr_reordered
        plt.imshow(symm, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.title(sym)
        # Save the figure
        plt.savefig("./{}.png".format(sym))
        plt.close()
