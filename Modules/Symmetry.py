import numpy as np
from scipy.linalg import schur
from scipy.linalg import null_space
class Symmetry:
    """
    Base class for symmetry analysis.

    Stores symmetry generators, irreducible representation projectors,
    and symmetry sectors (blocks).
    """

    def __init__(self):
        self.Symmetry_Generators = {}  # Dictionary to store symmetry operators (as matrices) 
        self.Centralizer=None
        self.IrrepsProjector=None             
        self.SymSectors = None         # Will store labeled symmetry sectors after projection
        self.commutative = None
    def _iscommutative(self):
        commutative=True
        generators=self.Symmetry_Generators
        for sym1 in generators:
            for sym2 in generators:
                comm=generators[sym1]@generators[sym2]-generators[sym2]@generators[sym1]
                if np.linalg.norm(comm)>10**(-9):
                    commutative=False
                    break
        self.commutative=commutative
    def _determineCentralizer(self):
        matrices = [self.Symmetry_Generators[key] for key in self.Symmetry_Generators]
        self.Centralizer=compute_common_centralizer(matrices)
    def _determineIrrepsProjector(self):
        self.IrrepsProjector = getIrrepsProjector(self.Centralizer)
    
    def _determineSymmetrySectors(self):
        symmetry_generators=self.Symmetry_Generators
        V=self.IrrepsProjector
        symmetry_generators = self.Symmetry_Generators
        dim = V.shape[0]
        Gs_in_V_basis = {sym: V.T @ G @ V for sym, G in symmetry_generators.items()}

        # Build a joint matrix to detect common invariant subspaces
        combined = np.zeros((dim, dim))
        for G in Gs_in_V_basis.values():
            combined += np.abs(G)**2

        # Optionally symmetrize numerically
        combined = (combined + combined.T) / 2

        # Use your block detection algorithm
        blocks = detect_block_sizes(combined)
        SymSectors={}
        for block in blocks:
            label=""
            for sym in Gs_in_V_basis:
                label+=sym+"="+str(int(np.round(np.trace(Gs_in_V_basis[sym][block[0]:block[0]+block[1],block[0]:block[0]+block[1]]),0)))
            if label in SymSectors:
                for it in range(block[0],block[0]+block[1]):
                    SymSectors[label].append(it)
            else:
                SymSectors[label]=[]
                for it in range(block[0],block[0]+block[1]):
                    SymSectors[label].append(it)
        self.SymSectors=SymSectors
       
#############################Helper Routines for Base Symmetry Class#############################
def compute_common_centralizer(matrices):
    """
    Given a list of d x d numpy arrays (matrices),
    compute the common centralizer: all X such that [X, A] = 0 for all A in matrices.
    Returns basis matrices of the centralizer.
    """
    d = matrices[0].shape[0]
    I = np.eye(d)
    A_blocks = []

    for M in matrices:
        A = np.kron(I, M) - np.kron(M.T, I)
        A_blocks.append(A)
    # Symmetric constraint: X = X^T -> X_ij - X_ji = 0 for i < j
    sym_constraints = []
    for i in range(d):
        for j in range(i+1, d):
            row = np.zeros((d, d))
            row[i, j] = 1
            row[j, i] = -1
            sym_constraints.append(row.flatten())

    # Combine all constraints
    A_total = np.vstack(A_blocks + sym_constraints)
    null = null_space(A_total)

    # Each column corresponds to vec(X), reshape to d x d matrix
    centralizer_basis = [vec.reshape((d, d)) for vec in null.T]
    # Enforce symmetry numerically
    symmetric_basis = [(X + X.T)/2 for X in centralizer_basis]
    return symmetric_basis


def getIrrepsProjector(centralizers):
    m = len(centralizers)
    np.random.seed(10)
    alpha = np.random.randn(m)
    P = sum(a * X for a, X in zip(alpha, centralizers))
    _,eigenvectors=np.linalg.eigh(P)
    return eigenvectors

def detect_block_sizes(matrix, tol=1e-10):
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
############################# END Helper Routines for Base Symmetry Class#############################







def getRotEigenvectors(pathtoEquilibriumxyz="./Equilibrium_Geometry/",rescale=True):
    """
    Computes the rotational eigenvectors according to "Vibrational Analysis in Gaussian,"
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
    - The function relies on the `readCoordinatesAndMasses`, `getInertiaTensor`, and `ComputeCenterOfMassCoordinates` functions.
    - The rotational eigenvectors are generated based on the principle axis obtained from the inertia tensor.
    - The translational eigenvectors are generated along each Cartesian axis.
    - The generated eigenvectors can be rescaled based on atom masses if the `rescale` flag is set to True.
    - All generated eigenvectors are normalized.
    """

    coordinates,masses,_=Read.readCoordinatesAndMasses(pathtoEquilibriumxyz)
    #Compute the Intertia Tensor
    I=Geometry.getInertiaTensor(coordinates,masses)
    centerofmasscoordinates,_=Geometry.ComputeCenterOfMassCoordinates(coordinates,masses)
    numofatoms=int(len(masses))
    sqrtM=np.zeros((3*numofatoms,3*numofatoms))
    for it in range(3*numofatoms):
        atomnum=int(np.floor((it/3)))
        sqrtM[it][it]=(np.sqrt(masses[atomnum]))
    #Get the principle Axis
    _,principleAxis=np.linalg.eigh(I)
    
    Roteigenvectors=[]
    factor=1.0
    for it in [0,1,2]:
        #Define the RotEigenvector
        RotEigenvector=np.zeros(3*numofatoms)
        #generate the vector X along the principle axis
        X=principleAxis[:,it]
        for s in range(numofatoms):
            rvector=np.cross(X,centerofmasscoordinates[s])
            factor=np.sqrt(masses[s])
            RotEigenvector[3*s]=rvector[0]*factor
            RotEigenvector[3*s+1]=rvector[1]*factor
            RotEigenvector[3*s+2]=rvector[2]*factor
        #Transform back 
        if not rescale:
            RotEigenvector=sqrtM@RotEigenvector
        #Normalize the generated Eigenvector
        RotEigenvector/=np.linalg.norm(RotEigenvector)
        Roteigenvectors.append(RotEigenvector)
    return Roteigenvectors
def ImposeGlobalRotationalSymmetry(Hessian,pathtoEquilibriumxyz="./Equilibrium_Geometry/"):
    # Imposes exact relation on the Hessian, that has to be fullfilled for the Translational D.O.F. to decouple from the rest
    #input: 
    #Hessian:    (numpy.array)  Hessian in carthesian coordinates 
    #Hessian:    (numpy.array)  Hessian in carthesian coordinates, which has been cleaned from contamination of the Rotations
    Roteigenvectors_unscaled=getRotEigenvectors(pathtoEquilibriumxyz=pathtoEquilibriumxyz,rescale=False)
    Transeigenvectors_unscaled=getTransEigenvectors(pathtoEquilibriumxyz=pathtoEquilibriumxyz,rescale=False)
    Orthogonalprojector_rot=np.identity(len(Roteigenvectors_unscaled[0]))
    for rotation in Roteigenvectors_unscaled:
        Orthogonalprojector_rot-=np.outer(rotation,rotation)
    Orthogonalprojector_trans=np.identity(len(Transeigenvectors_unscaled[0]))
    for trans in Transeigenvectors_unscaled:
        Orthogonalprojector_trans-=np.outer(trans,trans)
    print(Orthogonalprojector_rot@Orthogonalprojector_trans-Orthogonalprojector_trans@Orthogonalprojector_rot)
    Hessian=Orthogonalprojector_rot@Hessian@Orthogonalprojector_rot
    return Hessian

def Generate_Cn_Symmetry_Group(generator,group_order):
    group=[]
    group.append(np.eye(np.shape(generator)[0]))
    group.append(generator)
    generator_n=generator
    for it in range(2,group_order):
        generator_n=generator_n@generator
        group.append(generator_n)
    return group
def Generate_Z2_Symmetry_Group(generator):
    group=[]
    group.append(np.eye(np.shape(generator)[0]))
    group.append(generator)
    return group
def Combine_Symmetry_Group(group1,group2):
    #Check if group generators commute
    combined_group = []
    seen = set()
    # Initialize T and a set to track hashable representations of matrices in T
    for g1 in group1:
        for g2 in group2:
            # Compute the product
            product1 = g1 @ g2
            product2 = g2 @ g1
            # Convert the product to a hashable representation (e.g., tuple of tuples)
            product1_hashable = tuple(map(tuple, product1))
            product2_hashable = tuple(map(tuple, product2))
            
            # Add to T only if the product is not already seen
            if product1_hashable not in seen:
                combined_group.append(product1)
                seen.add(product1_hashable)
            # Add to T only if the product is not already seen
            if product2_hashable not in seen:
                combined_group.append(product2)
                seen.add(product2_hashable)
    return combined_group

    

def Enforce_Symmetry_On_Hessian(Hessian,parentfolder="./"):
    struct=determineSymmetry(parentfolder)
    #Translational Symmetry
    if struct.GlobalTranslationSymmetry:
        Hessian=ImposeTranslationalSymmetry(Hessian,pathtoEquilibriumxyz=parentfolder)
    #Rotational Symmetry
    #if struct.GlobalRotationSymmetry:
    #    Hessian=ImposeGlobalRotationalSymmetry(Hessian,pathtoEquilibriumxyz=parentfolder)
    group=[]
    if struct.TranslationSymmetry:
        Use_TranslationSymmetry=False
        str=input("Do You Want to use the Translation Symmetry?[Y/N]")
        if str=="Y" or len(str)==0:
            print("Using Translation Symmetry.")
            Use_TranslationSymmetry=True
        elif str=="N":
            Use_TranslationSymmetry=False
        else:
            print("Have not recognized Input.")
            print("Continuing With Default (Y).")
        #Translation Symmetry
        if Use_TranslationSymmetry:
            supercell=struct.supercell
            generators=np.array(struct.TranslationSymmetry_Generators)
            if supercell[0]!=1:
                group1=Generate_Cn_Symmetry_Group(generators[0,:,:],supercell[0])
                if supercell[1]!=1 and supercell[2]!=1:
                    group2=Generate_Cn_Symmetry_Group(generators[1,:,:],supercell[1])
                    group3=Generate_Cn_Symmetry_Group(generators[2,:,:],supercell[2])
                    group12=Combine_Symmetry_Group(group1,group2)
                    group=Combine_Symmetry_Group(group12,group3)
                elif supercell[1]==1 and supercell[2]!=1:
                    group2=Generate_Cn_Symmetry_Group(generators[2,:,:],supercell[2])
                    group=Combine_Symmetry_Group(group1,group2)
                else:
                    group=group1 
                
            else:
                if supercell[1]!=1:
                    group1=Generate_Cn_Symmetry_Group(generators[1,:,:],supercell[1])
                    if supercell[2]!=1:
                        group2=Generate_Cn_Symmetry_Group(generators[2,:,:],supercell[2])
                        group=Combine_Symmetry_Group(group1,group2)
                    else:
                        group=group1
                else:
                    group=Generate_Cn_Symmetry_Group(generators[2,:,:],supercell[2])
    if struct.InversionSymmetry:
        Use_InversionSymmetry=False
        str=input("Do You Want to use Inversion Symmetry?[Y/N]")
        if str=="Y" or len(str)==0:
            print("Using Inversion Symmetry.")
            Use_InversionSymmetry=True
        elif str=="N":
            Use_InversionSymmetry=False
        else:
            print("Have not recognized Input.")
            print("Continuing With Default (Y).")
        #Inversion Symmetry
        if Use_InversionSymmetry:
            generator_inversion=struct.InversionSymmetry_Generator[0]
            group_Inversion=Generate_Z2_Symmetry_Group(generator_inversion)
            if len(group)>0:
                group=Combine_Symmetry_Group(group,group_Inversion)
            else:
                group=group_Inversion

    if len(group)>0:
        symmetrized_Hessian=np.zeros(np.shape(Hessian))
        for element in group:
            representation=np.kron(element,np.eye(3))
            symmetrized_Hessian+=representation@Hessian@np.transpose(representation)
        symmetrized_Hessian/=len(group)
        return symmetrized_Hessian
    else:
        return Hessian



