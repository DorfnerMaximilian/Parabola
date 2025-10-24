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
        symmetry_generators = self.Symmetry_Generators
        dim = V.shape[0]
        Gs_in_V_basis = {sym: V.T @ G @ V for sym, G in symmetry_generators.items()}

        # Build a joint matrix to detect common invariant subspaces
        combined = np.zeros((dim, dim))
        for G in Gs_in_V_basis.values():
            combined += np.abs(G) ** 2

        # Optionally symmetrize numerically
        combined = (combined + combined.T) / 2

        # Use your block detection algorithm
        blocks = detect_block_sizes(combined)
        SymSectors = {}
        for block in blocks:
            label = "Id=" + str(
                int(
                    np.round(
                        np.trace(
                            np.eye(np.shape(combined)[0])[
                                block[0] : block[0] + block[1],
                                block[0] : block[0] + block[1],
                            ]
                        ),
                        0,
                    )
                )
            )
            for sym in Gs_in_V_basis:
                label += (
                    "|"
                    + sym
                    + "="
                    + str(
                        int(
                            np.round(
                                np.trace(
                                    Gs_in_V_basis[sym][
                                        block[0] : block[0] + block[1],
                                        block[0] : block[0] + block[1],
                                    ]
                                ),
                                0,
                            )
                        )
                    )
                )
            if label in SymSectors:
                for it in range(block[0], block[0] + block[1]):
                    SymSectors[label].append(it)
            else:
                SymSectors[label] = []
                for it in range(block[0], block[0] + block[1]):
                    SymSectors[label].append(it)
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
