import Modules.Util as util
import Modules.PhysConst as PhysConst
import Modules.Read as Read
import Modules.Write as Write
import Modules.Geometry as Geometry
import numpy as np
import pickle
import os
pathtocp2k=os.environ["cp2kpath"]
pathtobinaries=pathtocp2k+"/exe/local/"

class MolecularStructure():
    def __init__(self,name,path="./"):
        self.name=name
        self.path=os.getcwd()
        coordinates, masses, atomic_symbols = Read.readCoordinatesAndMasses(path)
        self.coordinates=coordinates
        self.masses=masses
        self.atoms=atomic_symbols
        if Read.readinPeriodicity(path):
            self.periodicity=(1,1,1)
        else:
            self.periodicity=(0,0,0)
        self.primitive_indices=np.arange(len(self.coordinates))
        self.cellvectors=Read.readinCellSize(path)
        self.symmetry={}
    def determine_symmetry(self):
        self._test_translation(tol_translation=5*10**(-4))
        self._test_inversion(tol_inversion=5*10**(-2))
        self._test_rotation(tol_rotation=5*10**(-2))

    def _test_translation(self,tol_translation):
        if self.periodicity==(1,1,1):
            coords=self.coordinates
            cellvectors=self.cellvectors
            atomic_symbols=self.atoms
            supercell,primitive_indices, scaled_lattice=getPrimitiveUnitCell(cellvectors, coords, atomic_symbols,tolerance=tol_translation)
            self.periodicity=supercell
            self.primitive_indices=primitive_indices
            relative_cell_coordinates, _=getCellCoordinates(scaled_lattice,coords,primitive_indices,supercell,tolerance=tol_translation)
            Tx,Ty,Tz,xFlag,yFlag,zFlag=getTranslationOps(relative_cell_coordinates,supercell)
            if xFlag and np.sum(np.abs(Tx-np.eye(np.shape(Tx)[0])))>10**(-10):
                self.symmetry["t1"]=Tx
            if yFlag and np.sum(np.abs(Ty-np.eye(np.shape(Ty)[0])))>10**(-10):
                self.symmetry["t2"]=Ty
            if zFlag and  np.sum(np.abs(Tz-np.eye(np.shape(Tz)[0])))>10**(-10):
                self.symmetry["t3"]=Tz
    def _test_inversion(self, tol_inversion):
        coords = self.coordinates
        atomic_symbols=self.atoms
        primitive_indices=self.primitive_indices
        geometry_centered_coordinates, _ = Geometry.ComputeCenterOfGeometryCoordinates(np.array(coords)[primitive_indices])
        has_symmetry, inversion_pairs = detect_inversion_symmetry(geometry_centered_coordinates, np.array(atomic_symbols)[primitive_indices], tol_inversion)
        if has_symmetry:
            #Generate the Original Pairs
            pairs={}
            for idx, inv_idx in inversion_pairs.items():
                pairs[primitive_indices[idx]]=primitive_indices[inv_idx]
            #Add the remaining pairs on the diagonal
            Ultimate_Pairs={}
            nAtoms=len(atomic_symbols)
            for it in range(nAtoms):
                if it in pairs:
                    Ultimate_Pairs[it]=pairs[it]
                else:
                    Ultimate_Pairs[it]=it
            PrimitiveInversion=get_Inversion_Symmetry_Generator(Ultimate_Pairs,nAtoms)
            self.symmetry["i"]=PrimitiveInversion
    def _test_rotation(self,tol_rotation,nmax=10):
        coordinates=self.coordinates
        atomic_symbols=self.atoms
        masses=self.masses
        primitive_indices=self.primitive_indices
        principleaxiscoordinates,masses,atomic_symbols=Geometry.getPrincipleAxisCoordinates(np.array(coordinates)[primitive_indices],np.array(masses)[primitive_indices],np.array(atomic_symbols)[primitive_indices])
        for n in range(nmax,1,-1):
            has_symmetry, rotation_pairs=detect_rotational_symmetry(principleaxiscoordinates, atomic_symbols, axis='x', n=n, tolerance=tol_rotation)
            if has_symmetry and n!=1:
                break
        if has_symmetry and n!=1:
            #Generate the Original Pairs
            pairs={}
            for idx, rot_idx in rotation_pairs.items():
                pairs[primitive_indices[idx]]=primitive_indices[rot_idx]
            #Add the remaining pairs on the diagonal
            Ultimate_Pairs={}
            nAtoms=len(atomic_symbols)
            for it in range(nAtoms):
                if it in pairs:
                    Ultimate_Pairs[it]=pairs[it]
                else:
                    Ultimate_Pairs[it]=it
            PrimitiveRotation=get_Inversion_Symmetry_Generator(Ultimate_Pairs,nAtoms)
            self.symmetry["Cx"+"_"+str(n)]=PrimitiveRotation
        for n in range(nmax,1,-1):
            has_symmetry, rotation_pairs=detect_rotational_symmetry(principleaxiscoordinates, atomic_symbols, axis='y', n=n, tolerance=tol_rotation)
            if has_symmetry and n!=1:
                break
        if has_symmetry and n!=1:
            #Generate the Original Pairs
            pairs={}
            for idx, rot_idx in rotation_pairs.items():
                pairs[primitive_indices[idx]]=primitive_indices[rot_idx]
            #Add the remaining pairs on the diagonal
            Ultimate_Pairs={}
            nAtoms=len(atomic_symbols)
            for it in range(nAtoms):
                if it in pairs:
                    Ultimate_Pairs[it]=pairs[it]
                else:
                    Ultimate_Pairs[it]=it
            PrimitiveRotation=get_Inversion_Symmetry_Generator(Ultimate_Pairs,nAtoms)
            self.symmetry["Cy"+"_"+str(n)]=PrimitiveRotation
        #check rotations 
        for n in range(nmax,1,-1):
            has_symmetry, rotation_pairs=detect_rotational_symmetry(principleaxiscoordinates, atomic_symbols, axis='z', n=n, tolerance=tol_rotation)
            if has_symmetry and n!=1:
                break
        if has_symmetry and n!=1:
            #Generate the Original Pairs
            pairs={}
            for idx, rot_idx in rotation_pairs.items():
                pairs[primitive_indices[idx]]=primitive_indices[rot_idx]
            #Add the remaining pairs on the diagonal
            Ultimate_Pairs={}
            nAtoms=len(atomic_symbols)
            for it in range(nAtoms):
                if it in pairs:
                    Ultimate_Pairs[it]=pairs[it]
                else:
                    Ultimate_Pairs[it]=it
            PrimitiveRotation=get_Inversion_Symmetry_Generator(Ultimate_Pairs,nAtoms)
            self.symmetry["Cz"+"_"+str(n)]=PrimitiveRotation
    def save(self, filename):
        """Save the current object to a pickle file."""
        with open(filename, "wb") as f:
            pickle.dump(self, f)
    @classmethod
    def load(cls, filename):
        """Load a MyData object from a pickle file."""
        with open(filename, "rb") as f:
            return pickle.load(f)
    def info(self):
        print("===============================")
        print("Molecular Structure Information")
        print("===============================")
        print(f"Name: {self.name}")
        print(f"Relative Path: {self.path}")
        print(f"Number of atoms: {len(self.atoms)}")
        print("Atomic Symbols:")
        print(", ".join(self.atoms))
        print("\n Carthesian Coordinates [in Angstroms]:")
        for i, coord in enumerate(self.coordinates):
            print(f"  Atom {i+1} [{self.atoms[i]}]: {coord}")
        print("\nCell Vectors (in Angstroms):")
        for i, vec in enumerate(self.cellvectors):
            print(f"  Vector {i+1}: {vec}")
        print("\nPeriodicity:")
        print(f"  {self.periodicity}")
        print("\nSymmetry Information:")
        if self.symmetry:
            for sym_type, op in self.symmetry.items():
                print(f"\nSymmetry Type: {sym_type}")
                print(f"  Generator matrix:\n{op}")
        else:
            print("  No symmetry information available.")

#### Translation Symmetry Helper Functions ####
def to_fractional(lattice, positions):
    """
    Converts Cartesian coordinates to fractional coordinates based on the given lattice.

    Parameters:
    - lattice (array-like, shape (3, 3)): 
        The lattice vectors defining the unit cell, where each row represents a lattice 
        vector in Cartesian coordinates.
    - positions (array-like, shape (n, 3)): 
        Cartesian coordinates of atoms or points to be converted into fractional coordinates.

    Returns:
    - fractional_positions (numpy array, shape (n, 3)): 
        Fractional coordinates of the input positions, where each coordinate is expressed 
        relative to the unit cell defined by the lattice vectors.

    Methodology:
    1. Compute the inverse of the lattice matrix using `np.linalg.inv`.
    2. For each position in `positions`, compute its fractional coordinates by multiplying 
    the position vector with the inverse lattice matrix (`inv_lattice @ pos`).
    3. Return the resulting fractional coordinates as a NumPy array.

    Example:
    ```python
    lattice = np.array([[1.0, 0.0, 0.0], 
                        [0.0, 1.0, 0.0], 
                        [0.0, 0.0, 1.0]])
    positions = np.array([[0.5, 0.5, 0.5], 
                        [1.0, 1.0, 1.0]])

    fractional_positions = to_fractional(lattice, positions)

    print("Fractional Coordinates:")
    print(fractional_positions)
    # Output:
    # [[0.5 0.5 0.5]
    #  [1.0 1.0 1.0]]
    ```

    Notes:
    - This function assumes the lattice is invertible.
    """
    inv_lattice = np.linalg.inv(lattice)
    fractional_positions = [inv_lattice @ pos for pos in positions]
    return np.array(fractional_positions)
def is_legitimate_scaled_cell(v1, v2, v3, coordinates, atomicsymbols, N1,N2,N3,tolerance=1e-5):
    """
    Checks if a scaled unit cell is legitimate by verifying that all atoms 
    in the scaled lattice can be mapped into the primitive unit cell using 
    periodic boundary conditions.

    Parameters:
    - v1, v2, v3 (array-like): 
        The three lattice vectors defining the original crystal lattice.
    - coordinates (array-like, shape (n, 3)): 
        Cartesian coordinates of the atomic positions in the original lattice.
    - atomicsymbols (list or array-like, shape (n,)): 
        List of atomic symbols corresponding to the atoms in `coordinates`.
    - N1, N2, N3 (int): 
        Scaling factors along the directions of lattice vectors `v1`, `v2`, and `v3`, respectively.
    - tolerance (float, optional, default=1e-5): 
        Numerical tolerance used to compare atomic positions and determine equivalence.

    Returns:
    - is_legitimate (bool): 
        True if the scaled unit cell is legitimate, False otherwise.
    - primitive_indices (array-like): 
        Indices of the atoms that belong to the primitive unit cell.
    - scaled_lattice (array-like, shape (3, 3)): 
        The scaled lattice vectors of the reduced unit cell.

    The function works as follows:
    1. Scales the lattice vectors based on the provided scaling factors.
    2. Converts Cartesian atomic coordinates to fractional coordinates in the scaled lattice.
    3. Identifies atoms within the primitive unit cell by their fractional coordinates.
    4. Generates translation vectors for periodic boundary conditions.
    5. Checks if all atoms in the scaled lattice can be mapped to the primitive unit cell, 
    comparing positions and atomic symbols within the specified tolerance.
    """
    # Scale the lattice vectors
    scaled_lattice = np.column_stack((v1 / N1, v2/N2, v3/N3 ))
    # Convert atom positions to fractional coordinates in the original lattice
    fractional_positions = to_fractional(scaled_lattice, coordinates)
    # Scale and wrap fractional positions into [0,1) range for periodicity
    primitive_indices=np.where((fractional_positions[:, 0] < 1-tolerance) & (fractional_positions[:, 1] < 1-tolerance) & (fractional_positions[:, 2] < 1-tolerance))[0]
    # Extract the candidate unit cell positions and symbols
    primitive_positions = fractional_positions[primitive_indices]
    primitive_symbols = np.array(atomicsymbols)[primitive_indices]
    # Generate all possible translation vectors within the periodic limits
    translations = [
        np.array([i , j , k ])
        for i in range(N1) for j in range(N2) for k in range(N3)
    ]
    # Check each translated position to confirm it maps into the unit cell
    for pos, symbol in zip(fractional_positions, atomicsymbols):
        # Check if the translated positions of `pos` map into the unit cell
        found_match = False
        for translation in translations:
            translated_pos = pos - translation  # Wrap within [0,1)
            # Check if the translated position matches any point in the primitive cell
            if any(
                np.allclose(translated_pos, primitive_pos, atol=tolerance) and primitive_symbol == symbol
                for primitive_pos, primitive_symbol in zip(primitive_positions, primitive_symbols)
            ):
                found_match = True
                break  # Found a match, no need to check other translations

        # If no match found for this atom, the cell is not legitimate
        if not found_match:
            return False, primitive_indices, scaled_lattice

    return True, primitive_indices, scaled_lattice
def getPrimitiveUnitCell(cellvectors, coordinates, atomicsymbols,tolerance=1e-8):
    """
    Identifies the primitive unit cell of a crystal lattice by determining the 
    smallest valid scaling factors along each lattice vector direction.

    Parameters:
    - cellvectors (array-like, shape (3, 3)): 
        The three lattice vectors defining the crystal lattice.
    - coordinates (array-like, shape (n, 3)): 
        Cartesian coordinates of the atomic positions in the lattice.
    - atomicsymbols (list or array-like, shape (n,)): 
        List of atomic symbols corresponding to the atoms in `coordinates`.
    - tolerance (float, optional, default=1e-5): 
        Numerical tolerance used to compare atomic positions and determine equivalence.
    - Nx, Ny, Nz (int, optional, default=5): 
        Maximum multipliers for refining the divisors of the scaling factors 
        along the x, y, and z lattice directions, respectively.

    Returns:
    - scaling_factors (tuple of 3 ints): 
        The scaling factors `(Nx, Ny, Nz)` that define the primitive unit cell 
        along the x, y, and z lattice directions.
    - primitive_indices (array-like): 
        Indices of the atoms that belong to the primitive unit cell.
    - scaled_lattice (array-like, shape (3, 3)): 
        The lattice vectors of the identified primitive unit cell.

    The function works as follows:
    1. Extracts the lattice vectors (`v1`, `v2`, `v3`) from the `cellvectors`.
    2. Iteratively tests divisors of the lattice vector lengths to find valid 
    scaling factors using the `is_legitimate_scaled_cell` function:
    - Initial divisors are selected from small prime numbers (e.g., 2, 3, 5).
    - The scaling factors are adjusted to maximize the number of divisions 
        that still produce a legitimate unit cell.
    3. Combines the refined scaling factors along the x, y, and z directions 
    to identify the primitive unit cell.
    4. Returns the scaling factors, indices of atoms in the primitive cell, 
    and the lattice vectors of the identified cell.
    """
    v1=cellvectors[0]
    v2=cellvectors[1]
    v3=cellvectors[2]
    #primes=[2,3,5,6,7,8,9,10,11,13,17,1]
    primes=range(30,0,-1)
    #x1 divisor
    for itx in primes:
        iscell,_,_=is_legitimate_scaled_cell(v1, v2, v3, coordinates, atomicsymbols, itx,1,1,tolerance=tolerance)
        if iscell:
            break
    #x2 divisor
    for ity in primes:
        iscell,_,_=is_legitimate_scaled_cell(v1, v2, v3, coordinates, atomicsymbols, 1,ity,1,tolerance=tolerance)
        if iscell:
            break
    #x3 divisor
    for itz in primes:
        iscell,_,_=is_legitimate_scaled_cell(v1, v2, v3, coordinates, atomicsymbols, 1,1,itz,tolerance=tolerance)
        if iscell:
            break
    iscell,primitive_indices, scaled_lattice=is_legitimate_scaled_cell(v1, v2, v3, coordinates, atomicsymbols,itx,ity,itz,tolerance=tolerance)
    return (itx,ity,itz),primitive_indices, scaled_lattice

def getCellCoordinates(lattice,coordinates,primitive_indices,supercell,tolerance):
    """
    Computes the relative cell coordinates and fractional coordinates of atoms within a unit cell.

    Parameters:
    - lattice (array-like, shape (3, 3)): 
        The lattice vectors defining the crystal structure.
    - coordinates (array-like, shape (n, 3)): 
        Cartesian coordinates of all atoms in the lattice.
    - primitive_indices (list of ints): 
        Indices of atoms belonging to the primitive unit cell.

    Returns:
    - relative_cell_coordinates (list of arrays): 
        List of relative cell coordinates [x_shift, y_shift, z_shift, primitive_index] for each atom.
    - in_cell_coordinates (array-like, shape (len(primitive_indices), 3)): 
        Fractional coordinates of atoms in the primitive unit cell wrapped to [0, 1).
    """
    # Convert atom positions to fractional coordinates in the original lattice
    fractional_positions = to_fractional(lattice, coordinates)
    primitive_positions = fractional_positions[primitive_indices]
    relative_cell_coordinates=[]
    translations = [
        np.array([i , j , k ])
        for i in range(supercell[0]) for j in range(supercell[1]) for k in range(supercell[2])
    ]

    for frac_coord in fractional_positions:
        for translation in translations:
            translated_pos = frac_coord - translation  # Wrap within [0,1)
            # Check if the translated position matches any point in the primitive cell
            isclosearray=[np.allclose(translated_pos, primitive_position, atol=tolerance) for primitive_position in primitive_positions]
            isclose=any(isclosearray)
            whereisclose=np.where(isclosearray)
            if isclose:
                relative_cell_coordinates.append(np.array([translation[0],translation[1],translation[2],whereisclose[0][0]]))
                break  # Found a match, no need to check other translations
    return np.array(relative_cell_coordinates),np.array(primitive_positions)

def getTranslationOps(relative_cell_coordinates,supercell):
    """
    Computes the translation operators \( T_x \), \( T_y \), and \( T_z \) for a given set 
    of relative cell coordinates in a supercell. These operators represent translations 
    by one unit along the x, y, and z directions in the supercell.

    Parameters:
    - relative_cell_coordinates (list of arrays): 
        A list where each element is an array of the form `[x, y, z, index]`, representing 
        the relative coordinates `(x, y, z)` of an atom in the supercell and its 
        corresponding index in the primitive cell.
    - supercell (array-like, shape (3,)): 
        The dimensions of the supercell `[Nx, Ny, Nz]`, where \( Nx, Ny, Nz \) are the 
        number of units along the x, y, and z directions, respectively.

    Returns:
    - Tx, Ty, Tz (numpy arrays, shape (n, n)): 
        Translation matrices along the x, y, and z directions, respectively. Each matrix 
        has shape `(n, n)` where \( n \) is the number of elements in `relative_cell_coordinates`. 
        The element \( T_x[i, j] = 1 \) indicates that applying a translation along x to 
        the atom at index \( j \) results in the atom at index \( i \). Similar logic applies 
        to \( T_y \) and \( T_z \).

    Raises:
    - ValueError: 
        If a translated position does not match any of the elements in `relative_cell_coordinates`.

    Methodology:
    1. Initialize zero matrices `Tx`, `Ty`, and `Tz` of size `(n, n)`.
    2. For each relative cell coordinate in `relative_cell_coordinates`:
       - Compute the translated coordinates by adding 1 to the x, y, and z components, 
         wrapping around using `np.mod` for periodic boundary conditions.
       - Find the matching relative cell coordinate in the list for each translation 
         (x, y, z) and update the corresponding entry in `Tx`, `Ty`, or `Tz`.
    3. If a match is not found for any translation, raise a `ValueError`.

    Example:
    ```python
    relative_cell_coordinates = [
        np.array([0, 0, 0, 0]),
        np.array([1, 0, 0, 0]),
        np.array([0, 1, 0, 0]),
        np.array([0, 0, 1, 0]),
    ]
    supercell = [2, 2, 2]

    Tx, Ty, Tz = getTranslationOps(relative_cell_coordinates, supercell)

    print("Tx:")
    print(Tx)
    print("Ty:")
    print(Ty)
    print("Tz:")
    print(Tz)
    ```
    """
    Tx=np.zeros((len(relative_cell_coordinates),len(relative_cell_coordinates)))
    Ty=np.zeros((len(relative_cell_coordinates),len(relative_cell_coordinates)))
    Tz=np.zeros((len(relative_cell_coordinates),len(relative_cell_coordinates)))
    xFlag=False
    yFlag=False
    zFlag=False
    for it1,rel_cell_coo1 in enumerate(relative_cell_coordinates):
        shiftedx=np.array([np.mod(rel_cell_coo1[0]+1,supercell[0]),rel_cell_coo1[1],rel_cell_coo1[2],rel_cell_coo1[3]])
        shiftedy=np.array([rel_cell_coo1[0],np.mod(rel_cell_coo1[1]+1,supercell[1]),rel_cell_coo1[2],rel_cell_coo1[3]])
        shiftedz=np.array([rel_cell_coo1[0],rel_cell_coo1[1],np.mod(rel_cell_coo1[2]+1,supercell[2]),rel_cell_coo1[3]])
        Txflag=False
        Tyflag=False
        Tzflag=False
        for it2,rel_cell_coo2 in enumerate(relative_cell_coordinates):
            if (shiftedx==rel_cell_coo2).all() and not Txflag:
                Tx[it2,it1]=1.0
                Txflag=True
                xFlag=True
            elif (shiftedy==rel_cell_coo2).all() and not Tyflag:
                Ty[it2,it1]=1.0
                Tyflag=True
                yFlag=True
            elif (shiftedz==rel_cell_coo2).all() and not Tzflag:
                Tz[it2,it1]=1.0
                Tzflag=True
                zFlag=True
    return Tx,Ty,Tz,xFlag,yFlag,zFlag

#### Inversion Symmetry Helper Functions ####
def find_negated_vector(vectors, target_vector, tolerance=1e-5):
    """
    Determines if the negation of a target vector exists within a list of vectors.
    
    Args:
        vectors (list): List of numpy arrays representing 3D coordinates.
        target_vector (numpy array): Vector to check for its negation in vectors.
        tolerance (float): Tolerance for floating-point comparison.

    Returns:
        tuple:
            - bool: True if negation exists within tolerance, False otherwise.
            - int or None: Index of the negated vector if found, None otherwise.
    """
    for idx, vec in enumerate(vectors):
        if np.allclose(-target_vector, vec, atol=tolerance):
            return True, idx
    return False, None
def detect_inversion_symmetry(centered_coords, atomic_symbols, tolerance=1e-5):
    """
    Checks if a set of atomic coordinates has inversion symmetry by verifying if each coordinate's 
    negation exists in the set with the same atomic symbol.
    
    Args:
        centered_coords (list): List of numpy arrays, each representing an atom's 3D coordinates.
        atomic_symbols (list): List of atomic symbols corresponding to each coordinate in centered_coords.
        tolerance (float): Tolerance for floating-point precision in symmetry detection.

    Returns:
        bool: True if inversion symmetry is detected, False otherwise.
        dict: Dictionary mapping each index to its inversion pair index if symmetry exists, empty if not.
    """
    inversion_pairs = {}

    for idx, coord in enumerate(centered_coords):
        # Check if the negated coordinates are in the list
        negation_found, neg_idx = find_negated_vector(centered_coords, coord, tolerance=tolerance)
        
        # Verify the atomic symbols match for symmetry
        if negation_found and atomic_symbols[idx] == atomic_symbols[neg_idx]:
            inversion_pairs[idx] = neg_idx
        else:
            return False, {}  # No symmetry if any atom lacks a valid inversion pair

    return True, inversion_pairs
def get_Inversion_Symmetry_Generator(pairs_pairs, n_atoms):
    """
    Generates a permutation matrix based on detected inversion pairs.
    
    Args:
        inversion_pairs (dict): Dictionary mapping each atom index to its inversion pair index.
        n_atoms (int): Total number of atoms in the system.

    Returns:
        numpy array: Permutation matrix implementing inversion symmetry.
    """
    perm_matrix = np.zeros((n_atoms, n_atoms), dtype=float)

    for idx, inv_idx in pairs_pairs.items():
        perm_matrix[idx, inv_idx] = 1.0
    return perm_matrix

def rotate_coords(coords, axis, angle_rad):
    """
    Rotates coordinates around a given axis by a given angle (in radians).
    """
    c, s = np.cos(angle_rad), np.sin(angle_rad)

    if axis == 'x':
        R = np.array([[1, 0, 0],
                      [0, c, -s],
                      [0, s, c]])
    elif axis == 'y':
        R = np.array([[c, 0, s],
                      [0, 1, 0],
                      [-s, 0, c]])
    elif axis == 'z':
        R = np.array([[c, -s, 0],
                      [s, c, 0],
                      [0, 0, 1]])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")

    return [R @ coord for coord in coords]


def find_rotated_match(rotated_coords, original_coords, atomic_symbols, tolerance=1e-5):
    """
    Attempts to match rotated coordinates back to original ones by atom type and position.
    Returns True and a mapping if a complete match is found.
    """
    used_indices = set()
    rotation_pairs = {}

    for i, (rot_coord, symbol) in enumerate(zip(rotated_coords, atomic_symbols)):
        matched = False
        for j, (orig_coord, orig_symbol) in enumerate(zip(original_coords, atomic_symbols)):
            if j in used_indices:
                continue
            if orig_symbol != symbol:
                continue
            if np.linalg.norm(rot_coord - orig_coord) < tolerance:
                rotation_pairs[i] = j
                used_indices.add(j)
                matched = True
                break
        if not matched:
            return False, {}

    return True, rotation_pairs


def detect_rotational_symmetry(centered_coords, atomic_symbols, axis='z', n=2, tolerance=1e-5):
    """
    Checks for Cn rotational symmetry about a specified axis.

    Args:
        centered_coords (list): List of numpy arrays, each representing an atom's 3D coordinates.
        atomic_symbols (list): List of atomic symbols corresponding to each coordinate.
        axis (str): Axis of rotation: 'x', 'y', or 'z'.
        n (int): Order of rotation (C_n means 360/n degrees).
        tolerance (float): Tolerance for matching atom positions.

    Returns:
        bool: True if Cn rotational symmetry is detected.
        dict: Mapping of original atom indices to rotated counterparts if symmetry exists.
    """
    angle_rad = 2 * np.pi / n
    rotated_coords = rotate_coords(centered_coords, axis, angle_rad)
    return find_rotated_match(rotated_coords, centered_coords, atomic_symbols, tolerance=tolerance)

class VibrationalStructure():
    def __init__(self,name,path="./"):
        MS=MolecularStructure(name,path)
        MS.determine_symmetry()
        self.MolecularStructure=MS
        self.Hessian=Read.readinHessian(path)
        self.MassMatrix=np.kron(np.diag(MS.masses),np.eye(3))
        self.Vibrational_Frequencies=[]
        self.Normalized_Carthesian_Displacements=[]
    def determine_symmetry(self):
        self.symmetry.determineSymmetry(self)
    def info(self):
        print("=================================")
        print("Vibrational Structure Information")
        print("=================================")
        print(f"Relative Path: {self.path}")
        print(f"Number of atoms: {len(self.atoms)}")
        print("Atomic Symbols:")
        print(", ".join(self.atoms))
        print("\n Carthesian Coordinates [in Angstroms]:")
        for i, coord in enumerate(self.coordinates):
            print(f"  Atom {i+1} [{self.atoms[i]}]: {coord}")
        print("\nCell Vectors (in Angstroms):")
        for i, vec in enumerate(self.cellvectors):
            print(f"  Vector {i+1}: {vec}")
        print("\nPeriodicity:")
        print(f"  {self.periodicity}")
        print("\nSymmetry Information:")
        if self.symmetry and isinstance(self.symmetry.SymmetryOperations, dict) and self.symmetry.SymmetryOperations:
            for sym_type, op in self.symmetry.SymmetryOperations.items():
                print(f"\nSymmetry Type: {sym_type}")
                print(f"  Generator matrix:\n{op.generator}")
        else:
            print("  No symmetry information available.")

        
