import numpy as np
from scipy.linalg import schur
from scipy.linalg import null_space
from scipy.sparse import csgraph


class Symmetry:
    """
    Base class for symmetry analysis.

    Stores symmetry generators, irreducible representation projectors,
    and symmetry sectors (blocks).
    """

    def __init__(self):
        self.Symmetry_Generators = (
            {}
        )  # Dictionary to store symmetry operators (as matrices)
        self.Centralizer = None
        self.IrrepsProjector = None
        self.SymSectors = None  # Will store labeled symmetry sectors after projection
        self.commutative = None

    def _iscommutative(self):
        commutative = True
        generators = self.Symmetry_Generators
        for sym1 in generators:
            for sym2 in generators:
                comm = (
                    generators[sym1] @ generators[sym2]
                    - generators[sym2] @ generators[sym1]
                )
                if np.linalg.norm(comm) > 10 ** (-9):
                    commutative = False
                    break
        self.commutative = commutative

    def _determineCentralizer(self):
        matrices = [self.Symmetry_Generators[key] for key in self.Symmetry_Generators]
        self.Centralizer = compute_common_centralizer(matrices)

    def _determineIrrepsProjector(self):
        self.IrrepsProjector = getIrrepsProjector(self.Centralizer)

    def _determineSymmetrySectors(self):
        symmetry_generators = self.Symmetry_Generators
        V = self.IrrepsProjector
        dim = V.shape[0]
        
        # 1. Transform generators to the V basis
        # Note: Use V.conj().T if V is complex!
        Gs_in_V_basis = {sym: V.conj().T @ G @ V for sym, G in symmetry_generators.items()}

        # 2. Build the Adjacency Matrix
        # We sum absolute values. If element (i,j) is non-zero, states i and j mix.
        adjacency = np.zeros((dim, dim))
        for G in Gs_in_V_basis.values():
            adjacency += np.abs(G)

        # 3. Apply Tolerance to remove noise
        # This is crucial. Anything below 1e-10 is considered zero connectivity.
        tol = 1e-10
        adjacency[adjacency < tol] = 0
        
        # 4. Find Connected Components (Invariant Subspaces)
        # This works even if indices are [0, 5, 2] for one sector.
        n_components, labels = csgraph.connected_components(adjacency, directed=False)

        SymSectors = {}
        
        # Group indices by their component label
        # sector_indices is a list of lists, e.g. [[0,1], [2,3,4], ...]
        sector_indices = [[] for _ in range(n_components)]
        for idx, label in enumerate(labels):
            sector_indices[label].append(idx)

        # 5. Label the sectors based on Characters (Traces)
        for indices in sector_indices:
            # We need to extract the sub-block for these specific indices
            # Use np.ix_ to slice non-contiguous rows/cols
            ix_mesh = np.ix_(indices, indices)
            
            label_parts = []
            
            # Calculate trace for Identity (Dimension of irrep)
            dim_trace = len(indices) 
            label_parts.append(f"Id={dim_trace}")

            for sym, G in Gs_in_V_basis.items():
                # Extract submatrix for this sector
                sub_G = G[ix_mesh]
                
                # Calculate Trace (Character)
                # Do NOT force int(). Use formatted float/complex string.
                chi = np.trace(sub_G)
                
                # Format to avoid 1.0000000002 issues
                if abs(chi.imag) < 1e-10:
                    chi_str = f"{chi.real:.0f}" # Real character
                else:
                    chi_str = f"{chi:.0f}" # Complex character
                
                label_parts.append(f"|{sym}={chi_str}")

            full_label = "".join(label_parts)

            if full_label not in SymSectors:
                SymSectors[full_label] = []
            
            SymSectors[full_label].extend(indices)

        self.SymSectors = SymSectors

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
        for j in range(i + 1, d):
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
    symmetric_basis = [(X + X.T) / 2 for X in centralizer_basis]
    return symmetric_basis


def getIrrepsProjector(centralizers):
    m = len(centralizers)
    np.random.seed(10)
    alpha = np.random.randn(m)
    P = sum(a * X for a, X in zip(alpha, centralizers))
    _, eigenvectors = np.linalg.eigh(P)
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
            off_block = matrix[i : i + size, i + size :]
            off_block_T = matrix[i + size :, i : i + size]

            if np.all(np.abs(off_block) < tol) and np.all(np.abs(off_block_T) < tol):
                # Check if next row/col introduces new coupling
                if i + size == n:
                    blocks.append((i, size))
                    i += size
                    block_found = True
                    break
                next_col = matrix[i : i + size, i + size]
                next_row = matrix[i + size, i : i + size]
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
