import Modules.Geometry as Geometry
import numpy as np
import Modules.PhysConst as PhysConst
import Modules.Read as Read
import Modules.Write as Write
import Modules.Util as Util
import os
pathtocp2k=os.environ["cp2kpath"]
pathtobinaries=pathtocp2k+"/exe/local/"


class Structure():
    def __init__(self,path="./", tol=1e-4):
        self.path=path
        coordinates, _, atomic_symbols = Geometry.getCoordinatesAndMasses(path)
        self.coordinates=coordinates
        self.atoms=atomic_symbols
        self.periodicity=Read.readinPeriodicity(path)
        self.cellvectors=Read.readinCellSize(path)
        self.supercell=(1,1,1)
        self.primitiveIndices=np.arange(len(self.coordinates))
        self.GlobalTranslationSymmetry=True
        self.GlobalRotationSymmetry=True
        self.TranslationSymmetry=[]
        self.tol_translation = tol
        self.InversionSymmetry=[]
        self.tol_Inversion = 10**(-2)
        self.MirrorSymmetry=[]

    def detect_GlobalTranslationalSymmetry(self):
        return  
    def detect_GlobalRotationalSymmetry(self):
        has_rotational_symmetry=not(bool(Read.readinPeriodicity(self.path)))
        self.GlobalRotationSymmetry=has_rotational_symmetry
        return 
    def detect_TranslationSymmetry(self):
        if self.periodicity:
            supercell,primitive_indices, scaled_lattice=getPrimitiveUnitCell(self.cellvectors, self.coordinates, self.atoms,tolerance=self.tol_translation,Nx=10,Ny=10,Nz=10)
            self.supercell=supercell
            if not(supercell==(1,1,1)):
               self.primitiveIndices=primitive_indices
               relative_cell_coordinates, _=getCellCoordinates(scaled_lattice,self.coordinates,primitive_indices)
               Tx,Ty,Tz=getTranslationOps(relative_cell_coordinates,supercell)
               self.TranslationSymmetry.append(Tx)
               self.TranslationSymmetry.append(Ty)
               self.TranslationSymmetry.append(Tz)
            else:
                print("Primitive Cell is given Cell!")
            return True
        else:
            print("No Periodicity Detected!")
            return False

    def detect_InversionSymmetry(self):
        print(self.supercell)
        geometry_centered_coordinates, _ = Geometry.ComputeCenterOfGeometryCoordinates(np.array(self.coordinates)[self.primitiveIndices])
        has_symmetry, inversion_pairs = detect_inversion_symmetry(geometry_centered_coordinates, np.array(self.atoms)[self.primitiveIndices], self.tol_Inversion)
        #Generate the Original Pairs
        pairs={}
        for idx, inv_idx in inversion_pairs.items():
            pairs[self.primitiveIndices[idx]]=self.primitiveIndices[inv_idx]
        #Add the remaining pairs on the diagonal
        Ultimate_Pairs={}
        nAtoms=len(self.atoms)
        for it in range(nAtoms):
            if it in pairs:
                Ultimate_Pairs[it]=pairs[it]
            else:
                Ultimate_Pairs[it]=it
        PrimitiveInversion=get_I_sym_permutation_matrix(Ultimate_Pairs,nAtoms)
        Tx=self.TranslationSymmetry[0]
        Ty=self.TranslationSymmetry[1]
        Tz=self.TranslationSymmetry[2]
        for n in range(self.supercell[0]):
            for m in range(self.supercell[1]):
                for p in range(self.supercell[2]):
                    print(n,m,p)
                    if n==0:
                        Tnx=np.eye(np.shape(Tx)[0])
                    else:
                        Tnx=np.linalg.matrix_power(Tx,n)
                    if m==0:
                        Tmy=np.eye(np.shape(Ty)[0])
                    else:
                        Tmy=np.linalg.matrix_power(Ty,m)
                    if p==0:
                        Tpz=np.eye(np.shape(Tz)[0])
                    else:
                        Tpz=np.linalg.matrix_power(Tz,p)
        self.InversionSymmetry.append(Tpz@Tmy@Tnx@PrimitiveInversion@np.transpose(Tnx)@np.transpose(Tmy)@np.transpose(Tpz))
        return has_symmetry
'''
        return has_symmetry
class xyMirrorSymmetry(Structure):
    """Class to detect inversion symmetry."""
    def detect_symmetry(self):
        geometry_centered_coordinates, _ = Geometry.ComputeCenterOfGeometryCoordinates(self.coordinates[self.primitiveIndices])
        has_symmetry, xyreflection_pairs = detect_reflection_symmetry(geometry_centered_coordinates, self.atoms[self.primitiveIndices],"xy", self.tol)
        return has_symmetry, xyreflection_pairs
class xzMirrorSymmetry(Structure):
    """Class to detect inversion symmetry."""
    def detect_symmetry(self):
        geometry_centered_coordinates, _ = Geometry.ComputeCenterOfGeometryCoordinates(self.coordinates)
        has_symmetry, xzreflection_pairs = detect_reflection_symmetry(geometry_centered_coordinates, self.atoms,"xz", self.tol)
        return has_symmetry, xzreflection_pairs
class yzMirrorSymmetry(Structure):
    """Class to detect inversion symmetry."""
    def detect_symmetry(self):
        geometry_centered_coordinates, _ = Geometry.ComputeCenterOfGeometryCoordinates(self.coordinates)
        has_symmetry, xyreflection_pairs = detect_reflection_symmetry(geometry_centered_coordinates, self.atoms,"yz", self.tol)
        return has_symmetry, xyreflection_pairs
'''
def getTransEigenvectors(pathtoEquilibriumxyz="./Equilibrium_Geometry/",rescale=True):
    """
    Computes the translational according to "Vibrational Analysis in Gaussian,"
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
    - The function relies on the `getCoordinatesAndMasses`, `getInertiaTensor`, and `ComputeCenterOfMassCoordinates` functions.
    - The rotational eigenvectors are generated based on the principle axis obtained from the inertia tensor.
    - The translational eigenvectors are generated along each Cartesian axis.
    - The generated eigenvectors can be rescaled based on atom masses if the `rescale` flag is set to True.
    - All generated eigenvectors are normalized.
    """

    _,masses,_=Geometry.getCoordinatesAndMasses(pathtoEquilibriumxyz)
    numofatoms=int(len(masses))
    Transeigenvectors=[]
    for it in [0,1,2]:
        #Define the TransEigenvector
        TransEigenvector=np.zeros(3*numofatoms)
        for s in range(numofatoms):
            if rescale:
                factor=np.sqrt(masses[s])
            else:
                factor=1.0
            TransEigenvector[3*s+it]=factor
        #Normalize the generated Eigenvector
        TransEigenvector/=np.linalg.norm(TransEigenvector)
        Transeigenvectors.append(TransEigenvector)
    return Transeigenvectors
def ImposeTranslationalSymmetry(Hessian,pathtoEquilibriumxyz="./Equilibrium_Geometry/"):
    # Imposes exact relation on the Hessian, that has to be fullfilled for the Translational D.O.F. to decouple from the rest
    #input: 
    #Hessian:    (numpy.array)  Hessian in carthesian coordinates 
    #Hessian:    (numpy.array)  Hessian in carthesian coordinates, which has been cleaned from contamination of the Translations
    Transeigenvectors_unscaled=getTransEigenvectors(pathtoEquilibriumxyz=pathtoEquilibriumxyz,rescale=False)
    Orthogonalprojector=np.identity(len(Transeigenvectors_unscaled[0]))
    for translation in Transeigenvectors_unscaled:
        Orthogonalprojector-=np.outer(translation,translation)
    Hessian=Orthogonalprojector@Hessian
    Hessian=0.5*(Hessian+np.transpose(Hessian))
    SaW=PhysConst.StandardAtomicWeights()
    #Get the Atomic coordinates 
    AtomicCoordinates=Read.readinAtomicCoordinates(pathtoEquilibriumxyz)
    #get the center of mass 
    MassofMolecule=0.0
    masses=[]
    for coords in AtomicCoordinates:
        mass=SaW[coords[1]]
        masses.append(mass)
        MassofMolecule+=mass
    J=getJ(masses)
    Hessiantilde=np.transpose(np.linalg.inv(J))@Hessian@np.linalg.inv(J)
    for it1 in range(3*len(masses)):
        for it2 in range(3*len(masses)):
            if it1>=3*len(masses)-3 or it2>=3*len(masses)-3:
                Hessiantilde[it1][it2]=0.0
    Hessian=np.transpose(J)@Hessiantilde@J
    Hessian=0.5*(Hessian+np.transpose(Hessian))
    return Hessian
def detect_reflection_symmetry(coordinates,atoms, plane,tol=1e-5):
    """
    Reflect a coordinates across a specified mirror plane.
    
    Parameters:
    - coordinates (np.array): An array of shape (N, 3) representing coordinates of N atoms.
    - plane (str): The plane to reflect across. Options are 'xy', 'xz', 'yz'.
    
    Returns:
    - np.array: The reflected coordinates.
    """
    reflected_structure = [coord.copy() for coord in coordinates]  # Create a copy of the coordinates
    
    # Reflect the coordinates based on the specified plane
    for coord in reflected_structure:
        if plane == 'xy':
            coord[2] *= -1  # Reflect z-coordinate
        elif plane == 'xz':
            coord[1] *= -1  # Reflect y-coordinate
        elif plane == 'yz':
            coord[0] *= -1  # Reflect x-coordinate
        else:
            raise ValueError("Invalid plane. Choose from 'xy', 'xz', 'yz'.")
    
    # Create a dictionary to store equivalent pairs
    equivalent_pairs = {}
    for i, original in enumerate(coordinates):
        # Find all atoms in the reflected structure that are equivalent to the original atom
        equivalent_indices = []
        for j, reflected in enumerate(reflected_structure):
            if np.allclose(original, reflected, atol=tol) and atoms[i] == atoms[j]:  # Check for equivalence
                equivalent_indices.append(j)
        
        equivalent_pairs[i] = equivalent_indices
    if not np.allclose(reflected_structure, coordinates, atol=tol):
        equivalent_pairs={}
    return np.allclose(reflected_structure, coordinates, atol=tol), equivalent_pairs



def getJ(masses):
    # Computes the J matrix, the linear, invertible transformation, that transforms into the Jacobi coordinates, in which the center of mass
    # motion explicitly decouples, x_rel=J*x
    #input: 
    #Hessian:    (numpy.array)  Hessian in carthesian coordinates 
    #Hessian:    (numpy.array)  Hessian in carthesian coordinates, which has been cleaned from contamination of the Translations
    Rcm=np.zeros(len(masses))
    Rcm[0]=1.0
    Mj=masses[0]
    Jupdown=np.zeros(((len(masses),len(masses))))
    for j in range(1,len(masses)):
        Rj=np.zeros(len(masses))
        Rj[j]=1.0
        Deltaj=Rj-Rcm
        Jupdown[j-1][:]=Deltaj
        Rcm=(Mj*Rcm+masses[j]*Rj)/(Mj+masses[j])
        Mj+=masses[j]
    Jupdown[-1][:]=Rcm
    J=np.zeros((3*len(masses),3*len(masses)))
    for it1 in range(len(masses)):
        for it2 in range(len(masses)):
            J[3*it1][3*it2]=Jupdown[it1][it2]
            J[3*it1+1][3*it2+1]=Jupdown[it1][it2]
            J[3*it1+2][3*it2+2]=Jupdown[it1][it2]
    return J
def getJMJTranspose(masses):
    # Transforms the Mass Matrix into the Jacobi coordinates = metric of the kinetic energy in these coordinates 
    #input: 
    #masses:         (N numpy.array)        masses of the atoms as a numpy array, requires same ordering as the coordinates
    #g:              (3Nx3N numpy.array)    metric of the kinetic energy in the Jacobi coordinates
    #get the mass matrix
    M=np.zeros((3*len(masses),3*len(masses)))
    for it in range(3*len(masses)):
        atomnum=int(np.floor((it/3)))
        M[it][it]=1./masses[atomnum]
    #get the mass matrix
    J=getJ(masses)
    return J@M@np.transpose(J)
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
    - The function relies on the `getCoordinatesAndMasses`, `getInertiaTensor`, and `ComputeCenterOfMassCoordinates` functions.
    - The rotational eigenvectors are generated based on the principle axis obtained from the inertia tensor.
    - The translational eigenvectors are generated along each Cartesian axis.
    - The generated eigenvectors can be rescaled based on atom masses if the `rescale` flag is set to True.
    - All generated eigenvectors are normalized.
    """

    coordinates,masses,_=Geometry.getCoordinatesAndMasses(pathtoEquilibriumxyz)
    #Compute the Intertia Tensor
    I=Geometry.getInertiaTensor(coordinates,masses)
    centerofmasscoordinates,_=Geometry.ComputeCenterOfMassCoordinates(coordinates,masses)

    #Get the principle Axis
    _,principleAxis=np.linalg.eigh(I)
    numofatoms=int(len(masses))
    Roteigenvectors=[]
    factor=1.0
    for it in [0,1,2]:
        #Define the RotEigenvector
        RotEigenvector=np.zeros(3*numofatoms)
        #generate the vector X along the principle axis
        X=principleAxis[:,it]
        for s in range(numofatoms):
            rvector=np.cross(X,centerofmasscoordinates[s])
            if rescale:
                factor=np.sqrt(masses[s])
            else:
                factor=1.0
            RotEigenvector[3*s]=rvector[0]*factor
            RotEigenvector[3*s+1]=rvector[1]*factor
            RotEigenvector[3*s+2]=rvector[2]*factor
        #Normalize the generated Eigenvector
        RotEigenvector/=np.linalg.norm(RotEigenvector)
        Roteigenvectors.append(RotEigenvector)
    return Roteigenvectors
def ImposeGlobalRotationalSymmetry(Hessian,pathtoEquilibriumxyz="./Equilibrium_Geometry/"):
    # Imposes exact relation on the Hessian, that has to be fullfilled for the Translational D.O.F. to decouple from the rest
    #input: 
    #Hessian:    (numpy.array)  Hessian in carthesian coordinates 
    #Hessian:    (numpy.array)  Hessian in carthesian coordinates, which has been cleaned from contamination of the Translations
    Roteigenvectors_unscaled=getRotEigenvectors(pathtoEquilibriumxyz=pathtoEquilibriumxyz,rescale=False)
    Orthogonalprojector=np.identity(len(Roteigenvectors_unscaled[0]))
    for rotation in Roteigenvectors_unscaled:
        Orthogonalprojector-=np.outer(rotation,rotation)
    Hessian=Orthogonalprojector@Hessian
    Hessian=0.5*(Hessian+np.transpose(Hessian))
    return Hessian
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
def get_I_sym_permutation_matrix(pairs_pairs, n_atoms):
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

def Impose_Inversion_Symmetry(Hessian,pairs):
    # Imposes exact relation on the Hessian
    #input: 
    #Hessian:    (numpy.array)  Hessian in carthesian coordinates 
    #Hessian:    (numpy.array)  Hessian in carthesian coordinates, which has been cleaned from contamination of the Translations
    n_atoms=int(np.shape(Hessian)[0]/3)
    permutation_Matrix=get_I_sym_permutation_matrix(pairs, n_atoms)
    transformed_Hessian=permutation_Matrix@Hessian@permutation_Matrix
    Hessian=0.5*(Hessian+transformed_Hessian)
    return Hessian
def get_reflection_permutation_matrix(equivalent_pairs, n_atoms):
    """
    Generates a permutation matrix based on detected equivalent pairs for reflection symmetry.
    
    Args:
        equivalent_pairs (dict): Dictionary mapping each atom index to its equivalent pair indices after reflection.
        n_atoms (int): Total number of atoms in the system.
        plane (str): The plane of reflection ('xy', 'xz', 'yz').

    Returns:
        numpy array: Permutation matrix implementing reflection symmetry.
    """
    perm_matrix = np.zeros((3 * n_atoms, 3 * n_atoms), dtype=int)

    for idx, equivalent_indices in equivalent_pairs.items():
        for inv_idx in equivalent_indices:
            # Fill the permutation matrix for the reflection symmetry
            perm_matrix[3 * idx:3 * idx + 3, 3 * inv_idx:3 * inv_idx + 3] = np.eye(3, dtype=float)

    return perm_matrix
def Impose_Refection_Symmetry(Hessian,pairs):
    # Imposes exact relation on the Hessian
    #input: 
    #Hessian:    (numpy.array)  Hessian in carthesian coordinates 
    #Hessian:    (numpy.array)  Hessian in carthesian coordinates, which has been cleaned from contamination of the Translations
    n_atoms=int(np.shape(Hessian)[0]/3)
    permutation_Matrix=get_reflection_permutation_matrix(pairs, n_atoms)
    transformed_Hessian=permutation_Matrix@Hessian@permutation_Matrix
    Hessian=0.5*(Hessian+transformed_Hessian)
    return Hessian
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
    primitive_indices=np.where((fractional_positions[:, 0] < 1) & (fractional_positions[:, 1] < 1) & (fractional_positions[:, 2] < 1))[0]
    # Extract the candidate unit cell positions and symbols
    primitive_positions = fractional_positions[primitive_indices]
    primitive_symbols = np.array(atomicsymbols)[primitive_indices]
    # Generate all possible translation vectors within the periodic limits
    translations = [
        np.array([i / N1, j / N2, k / N3])
        for i in range(N1) for j in range(N2) for k in range(N3)
    ]
    # Check each translated position to confirm it maps into the unit cell
    for pos, symbol in zip(fractional_positions, atomicsymbols):
        # Check if the translated positions of `pos` map into the unit cell
        found_match = False
        for translation in translations:
            translated_pos = np.mod(pos + translation, 1.0)  # Wrap within [0,1)
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
def getPrimitiveUnitCell(cellvectors, coordinates, atomicsymbols,tolerance=1e-5,Nx=5,Ny=5,Nz=5):
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
    primes=[2,3,5,7,11,13,1]
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
    #Check maximum divisor:
    for multx in range(1,Nx):
        iscell,_,_=is_legitimate_scaled_cell(v1, v2, v3, coordinates, atomicsymbols, multx*itx,1,1,tolerance=tolerance)
        if not iscell:
            break
    multx-=1
    #Check maximum divisor:
    for multy in range(1,Ny):
        iscell,_,_=is_legitimate_scaled_cell(v1, v2, v3, coordinates, atomicsymbols,1,multy*ity,1,tolerance=tolerance)
        if not iscell:
            break
    multy-=1
    #Check maximum divisor:
    for multz in range(1,Nz):
        iscell,_,_=is_legitimate_scaled_cell(v1, v2, v3, coordinates, atomicsymbols,1,1,multz*itz,tolerance=tolerance)
        if not iscell:
            break
    multz-=1
    iscell,primitive_indices, scaled_lattice=is_legitimate_scaled_cell(v1, v2, v3, coordinates, atomicsymbols,multx*itx,multy*ity,multz*itz,tolerance=tolerance)
    return (multx*itx,multy*ity,multz*itz),primitive_indices, scaled_lattice


def getCellCoordinates(lattice,coordinates,primitive_indices):
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
    #determine cell
    in_cell_coordinates=[]
    for it in primitive_indices:
        in_cell_coordinates.append(np.array([fractional_positions[it][0]-np.floor(fractional_positions[it][0]),fractional_positions[it][1]-np.floor(fractional_positions[it][1]),fractional_positions[it][2]-np.floor(fractional_positions[it][2])]))
    relative_cell_coordinates=[]
    for frac_coord in fractional_positions:
        cell=(np.floor(frac_coord[0]),np.floor(frac_coord[1]),np.floor(frac_coord[2]))
        arr=np.array([frac_coord[0]-np.floor(frac_coord[0]),frac_coord[1]-np.floor(frac_coord[1]),frac_coord[2]-np.floor(frac_coord[2])])
        for it,coor in enumerate(in_cell_coordinates):
            if np.allclose(coor,arr , rtol=1e-3, atol=1e-5, equal_nan=False):
                relative_cell_coordinates.append(np.array([cell[0],cell[1],cell[2],it]))
    return np.array(relative_cell_coordinates),np.array(in_cell_coordinates)
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
            elif (shiftedy==rel_cell_coo2).all() and not Tyflag:
                Ty[it2,it1]=1.0
                Tyflag=True
            elif (shiftedz==rel_cell_coo2).all() and not Tzflag:
                Tz[it2,it1]=1.0
                Tzflag=True
    return Tx,Ty,Tz

def determineSymmetry(Hessian,parentfolder="./", tol=1e-5):
    """
    Detects various symmetries in a molecular geometry and prints results.
    
    Args:
        parentfolder (str): Path to the folder containing molecular data.
        tol (float): Tolerance for symmetry detection.
    """    
    # List of symmetry operations to check
    symmetry_operations = [
        GlobalTranslationalSymmetry(path=parentfolder, tol=tol),
        GlobalRotationalSymmetry(path=parentfolder, tol=tol),
        TranslationSymmetry(path=parentfolder,tol=tol)
        #InversionSymmetry(path=parentfolder, tol=tol),
        #xyMirrorSymmetry(path=parentfolder,tol=tol),
        #xzMirrorSymmetry(path=parentfolder,tol=tol),
        #yzMirrorSymmetry(path=parentfolder,tol=tol)
        # RotationalSymmetry(tol),  # Uncomment as implemented
        # MirrorSymmetry(tol)       # Uncomment as implemented
    ]
    
    # Detect symmetries
    for it,symmetry in enumerate(symmetry_operations):
        has_symmetry = symmetry.detect_symmetry()
        if has_symmetry:
            print(f"Found {symmetry.__class__.__name__}")
            if it==0:
                strin=input("Impose GlobalTranslationalSymmetry [Y/N]?")
                if strin=="N":
                    print(f"Not imposing {symmetry.__class__.__name__}!")
                else:
                    print(f"Imposing {symmetry.__class__.__name__}!")
                    Hessian=ImposeTranslationalSymmetry(Hessian,pathtoEquilibriumxyz=parentfolder+"/Equilibrium_Geometry/")
            if it==1:
                strin=input("Impose GlobalRotationalSymmetry [Y/N]?")
                if strin=="N":
                    print(f"Not imposing {symmetry.__class__.__name__}!")
                else:
                    print(f"Have not recognized input!")
                    print(f"Imposing {symmetry.__class__.__name__}!")
                    Hessian=ImposeGlobalRotationalSymmetry(Hessian,pathtoEquilibriumxyz=parentfolder+"/Equilibrium_Geometry/")
            if it==2:
                strin=input("Impose InversionSymmetry [Y/N]?")
                if strin=="N":
                    print(f"Not imposing {symmetry.__class__.__name__}!")
                else:
                    print(f"Have not recognized input!")
                    print(f"Imposing {symmetry.__class__.__name__}!")
                    Hessian=Impose_Inversion_Symmetry(Hessian,symmetry_pairs)
    return 
