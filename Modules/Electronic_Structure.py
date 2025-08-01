import Modules.Read as Read
import Modules.AtomicBasis as AtomicBasis
from Modules.Molecular_Structure import MolecularStructure
import Modules.Symmetry as Symmetry
import numpy as np
from scipy.linalg import schur
from itertools import permutations
import matplotlib.pyplot as plt
import os
pathtocp2k=os.environ["cp2kpath"]
pathtobinaries=pathtocp2k+"/exe/local/"
def get_representation_matrices():
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
def get_transformation_matrices(O):
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


    representation_matrices=get_representation_matrices()
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
        print(ordering_atom)
        for it in range(1,len(Basis[atom])):
            if (Basis[atom][it][2][0]!=ordering_atom[it-1] or Basis[atom][it][1]!=Basis[atom][it-1][1]) or Basis[atom][it][0]!=Basis[atom][it-1][0]:
                ordering_atom.append(Basis[atom][it][2][0])
        ordering[atom]=ordering_atom
    return ordering
        
class ElectronicStructure(MolecularStructure):
    def __init__(self,name,path="./"):
        # Initialize MolecularStructure
        super().__init__(name, path)
        # If the object was loaded from a pickle, it will already have its
        # vibrational attributes. If so, we can exit immediately.
        # We check for 'Hessian', an attribute unique to this child class.
        if hasattr(self, 'Hamiltonian'):
            return
        self.Basis=AtomicBasis.getBasis(path)
        print(self.Basis)
        KS_alpha,KS_beta,OLM=Read.readinMatrices(path)
        self.UKS=Read.checkforUKS(path)
        self.KS_Hamiltonian_alpha=KS_alpha
        self.KS_Hamiltonian_beta=KS_beta
        self.OLM=OLM
        self.Electronic_Symmetry = Electronic_Symmetry(self.Molecular_Symmetry)
        
    
    


class ElectronicState:
    def __init__(self,state, symmetry_label=None):
        self.a=state
        self.A=None
        self.symmetry_label=symmetry_label
        

    
    



    
    
class Electronic_Symmetry(Symmetry.Symmetry):
    def __init__(self,ElectronicStructure):
        super().__init__()  # Initialize parent class
        molecular_symmetry=ElectronicStructure.Molecular_Symmetry

        if "Id" not in molecular_symmetry.Symmetry_Generators.keys():
            generators={}
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
    
    
def getXYZRepresentation(symmetrylabel):
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

