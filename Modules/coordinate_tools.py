import numpy as np
from typing import List, Tuple, Dict
import itertools
import collections
def compute_center_of_mass_coordinates(coordinates,masses):
    """
    Computes the center of mass of a molecule with respect to the basis of the coordinate file.

    Parameters:
    - coordinates (Nx3 numpy.array): Coordinates of the atoms with respect to some basis (arbitrary origin).
    - masses (N numpy.array): Masses of the atoms as a numpy array, requires the same ordering as the coordinates.

    Returns:
    - centerofmasscoordinates (Nx3 numpy.array): Coordinates of the atoms with respect to some basis, origin at the center of mass.
    - centerofmass (3x1 numpy.array): Coordinates of the center of mass with respect to the old frame.

    Notes:
    - The coordinates provided have an arbitrary origin.
    - The returned centerofmasscoordinates have their origin at the center of mass.
    - The returned centerofmass is the coordinates of the center of mass with respect to the old frame.
    - The center of mass is computed based on the provided masses and coordinates.
    """
    if len(coordinates[0])==3:
        MassofMolecule=0.0
        centerofmass=np.array([0.,0.,0.])
        #Compute the center of mass: 
        for it,coords in enumerate(coordinates):
            mass=masses[it]
            MassofMolecule+=mass
            centerofmass+=mass*np.array([coords[0],coords[1],coords[2]])
        centerofmass/=MassofMolecule
        
        centerofmasscoordinates=[]
        #get the center of mass coordinates
        for coords in coordinates:
            centerofmasscoordinates.append(coords-centerofmass)
    elif len(coordinates[0])==2:
        MassofMolecule=0.0
        centerofmass=np.array([0.,0.])
        #Compute the center of mass: 
        for it,coords in enumerate(coordinates):
            mass=masses[it]
            MassofMolecule+=mass
            centerofmass+=mass*np.array([coords[0],coords[1]])
        centerofmass/=MassofMolecule
        
        centerofmasscoordinates=[]
        #get the center of mass coordinates
        for coords in coordinates:
            centerofmasscoordinates.append(coords-centerofmass)

    return centerofmasscoordinates,centerofmass
def compute_center_of_geometry_coordinates(coordinates):
    """
    Computes the center of geometry of a molecule with respect to the basis of the coordinate file.

    Parameters:
    - coordinates (Nx3 numpy.array): Coordinates of the atoms with respect to some basis (arbitrary origin).
    - masses (N numpy.array): Masses of the atoms as a numpy array, requires the same ordering as the coordinates.

    Returns:
    - centerofmasscoordinates (Nx3 numpy.array): Coordinates of the atoms with respect to some basis, origin at the center of mass.
    - centerofmass (3x1 numpy.array): Coordinates of the center of mass with respect to the old frame.

    Notes:
    - The coordinates provided have an arbitrary origin.
    - The returned centerofmasscoordinates have their origin at the center of mass.
    - The returned centerofmass is the coordinates of the center of mass with respect to the old frame.
    - The center of mass is computed based on the provided masses and coordinates.
    """
    if len(coordinates[0])==3:
        center=np.array([0.,0.,0.])
        #Compute the center of mass: 
        for _,coords in enumerate(coordinates):
            center+=np.array([coords[0],coords[1],coords[2]])
        center/=len(coordinates)
        
        centercoordinates=[]
        #get the center of mass coordinates
        for coords in coordinates:
            centercoordinates.append(coords-center)
        return centercoordinates,center
    elif len(coordinates[0])==2:
        center=np.array([0.,0.])
        #Compute the center of mass: 
        for _,coords in enumerate(coordinates):
            center+=np.array([coords[0],coords[1]])
        center/=len(coordinates)
        centercoordinates=[]
        #get the center of mass coordinates
        for coords in coordinates:
            centercoordinates.append(coords-center)
        return centercoordinates,center
def get_geometric_covariance_tensor(coordinates):
    """
    Computes the covariance matrix of the molecular geometry based on coordinates 
    centered at the center of geometry.

    Parameters:
    - coordinates (Nx3 numpy.array): Coordinates of the atoms with respect to some basis (arbitrary origin).

    Returns:
    - cov (3x3 numpy.array): Covariance matrix of the centered coordinates,
      representing the spread of atoms around the center of geometry.

    Notes:
    - The coordinates provided have an arbitrary origin.
    - The covariance is computed after translating the coordinates so that their origin
      is at the center of geometry.
    - The center of geometry is the arithmetic mean of the atomic coordinates (equal weighting).
    """
    centercoordinates,_=compute_center_of_geometry_coordinates(coordinates)
    cov = np.cov(np.array(centercoordinates).T)
    return cov
def get_inertia_tensor(coordinates,masses):
    """
    Computes the inertia tensor of a molecule with respect to the basis of the coordinate file.
    Note that the inertia tensor is only a physically meaningful object if the coordinates have
    the center of mass as an origin.

    Parameters:
    - coordinates (Nx3 numpy.array): Coordinates of the atoms with respect to some basis.
    - masses (N numpy.array): Masses of the atoms as a numpy array, requires the same ordering as the coordinates.

    Returns:
    - I (3x3 numpy.array): Moment of inertia tensor of the molecule.

    Notes:
    - This function relies on the `ComputeCenterOfMassCoordinates` function.
    - The coordinates provided should have the center of mass as the origin.
    - The returned inertia tensor is a 3x3 numpy array.
    - The moments of inertia are computed along the principal axes x, y, and z.
    """
    if len(coordinates[0])==3:
        #Compute the coordinates in the frame where the origin is the center of mass
        centerofmasscoordinates,_=compute_center_of_mass_coordinates(coordinates,masses)
        #get the moments of inertia tensor
        Ixx=0.0
        Iyy=0.0
        Izz=0.0
        Ixy=0.0
        Ixz=0.0
        Iyz=0.0
        for it,coords in enumerate(centerofmasscoordinates):
            Ixx+=masses[it]*(coords[1]**2+coords[2]**2)
            Iyy+=masses[it]*(coords[0]**2+coords[2]**2)
            Izz+=masses[it]*(coords[0]**2+coords[1]**2)
            Ixy+=(-1)*masses[it]*coords[0]*coords[1]
            Ixz+=(-1)*masses[it]*coords[0]*coords[2]
            Iyz+=(-1)*masses[it]*coords[1]*coords[2]
        #The moment of inertia tensor
        I=np.array([[Ixx,Ixy,Ixz],[Ixy,Iyy,Iyz],[Ixz,Iyz,Izz]])
        return I
    elif len(coordinates[0])==2:
        #Compute the coordinates in the frame where the origin is the center of mass
        centerofmasscoordinates,_=compute_center_of_mass_coordinates(coordinates,masses)
        #get the moments of inertia tensor
        Ixx=0.0
        Iyy=0.0
        Ixy=0.0
        for it,coords in enumerate(centerofmasscoordinates):
            Ixx+=masses[it]*(coords[1]**2+coords[2]**2)
            Iyy+=masses[it]*(coords[0]**2+coords[2]**2)
            Ixy+=(-1)*masses[it]*coords[0]*coords[1]
        #The moment of inertia tensor
        I=np.array([[Ixx,Ixy],[Ixy,Iyy]])
        return I
def get_principle_axis(centeredCoordinates, masses=None, tol=1e-3):
    """
    Computes and orients the principal axes of a system of particles.

    Parameters
    ----------
    centerofmasscoordinates : np.ndarray
        Coordinates (N, 3) relative to the center of mass.
    masses : np.ndarray
        Masses of the N particles.

    Returns
    -------
    tuple of np.ndarray
        Orthonormal principal axes (v1, v2, v3), forming a right-handed system.
    """
    def is_degenerate(evals, threshold=tol):
            pairs=[]
            # Check for near-equal eigenvalues
            for i in range(len(evals)):
                for j in range(i+1, len(evals)):
                    if np.abs(evals[i] - evals[j]) / np.mean([evals[i], evals[j]]) < threshold:
                        pairs.append((i,j))
            if not pairs:
                return False, [()]
            else:
                return True, pairs
    if len(centeredCoordinates[0])==3:
        if masses is not None:
            # Compute the inertia tensor
            I = get_inertia_tensor(centeredCoordinates, masses)
        else:
            I = get_geometric_covariance_tensor(np.array(centeredCoordinates))
        # Diagonalize
        evals, evecs = np.linalg.eigh(I)

        # Check degeneracy
        degenerate, pairs = is_degenerate(evals)
        if degenerate:
            if len(pairs) == 3:
                # Fully degenerate case: all eigenvalues nearly equal
                # Just align the first axis with x as much as possible
                v1 = evecs[:, 0]
                if v1[0] < 0:
                    v1 *= -1
                # Gram-Schmidt to get orthonormal basis
                v2 = evecs[:, 1]
                v2 = v2 - np.dot(v1, v2) * v1
                v2 /= np.linalg.norm(v2)
                v3 = np.cross(v1, v2)
                v3 /= np.linalg.norm(v3)
            else:
                # Partially degenerate: reorient degenerate subspace
                i, j = pairs[0]
                subspace = evecs[:, [i, j]]
                x_proj = subspace.T @ np.array([1, 0, 0])
                best_index = np.argmax(np.abs(x_proj))
                v1 = subspace[:, best_index]
                v1 *= np.sign(v1[0])  # Ensure positive x
                v2 = subspace[:, 1 - best_index]
                v2 = v2 - np.dot(v1, v2) * v1
                v2 /= np.linalg.norm(v2)
                v3 = np.cross(v1, v2)
                v3 /= np.linalg.norm(v3)
        else:
            v1 = evecs[:, 0]
            v2 = evecs[:, 1]
            v3 = evecs[:, 2]
            # Ensure v1 and v2 have positive x-component for consistency
            if v1[0] < 0: v1 *= -1
            if v2[0] < 0: v2 *= -1
            v3 = np.cross(v1, v2)
            v3 /= np.linalg.norm(v3)
        

        axes = [v1, v2, v3]
        overlap_matrix = np.abs(np.array(axes))

        max_overlap = -1.0
        best_assignment = None

        # Iterate through all 3! = 6 permutations of the axis assignments
        for p in [[0, 1, 2],[1, 0, 2],[0, 2, 1],[2, 1, 0],[1, 2, 0],[2, 0, 1]]:
            # 'p' represents a potential assignment. For example, p=(2,0,1) means:
            #   - new x-axis is old axis 2 (v3)
            #   - new y-axis is old axis 0 (v1)
            #   - new z-axis is old axis 1 (v2)
            # We calculate the total overlap score for this assignment.
            current_overlap = (overlap_matrix[p[0], 0] +  # old axis p[0] with new x
                            overlap_matrix[p[1], 1] +  # old axis p[1] with new y
                            overlap_matrix[p[2], 2])   # old axis p[2] with new z

            if current_overlap > max_overlap:
                max_overlap = current_overlap
                best_assignment = p

        # The best_assignment tells us which old axis to use for each new axis.
        # For example, if best_assignment is (2, 0, 1), the ordered list is [axes[2], axes[0], axes[1]].
        ordered_axes = [axes[i] for i in best_assignment]
        v1_final, v2_final, _ = ordered_axes

        # --- END: NumPy-based ordering logic ---
        
        # Ensure a consistent, right-handed final orientation.
        if np.dot(v1_final, [1, 0, 0]) < 0:
            v1_final = -v1_final
            
        if np.dot(v2_final, [0, 1, 0]) < 0:
            v2_final = -v2_final
            
        # Set v3 from the cross product to guarantee a right-handed system.
        v3_final = np.cross(v1_final, v2_final)

        return v1_final, v2_final, v3_final
    elif len(centeredCoordinates[0])==2:
        if masses is not None:
            # Compute the inertia tensor
            I = get_inertia_tensor(centeredCoordinates, masses)
        else:
            I = get_geometric_covariance_tensor(np.array(centeredCoordinates))
        # Diagonalize
        evals, evecs = np.linalg.eigh(I)
        # Check degeneracy
        degenerate, pairs = is_degenerate(evals)

        if degenerate:
            # In 2D, if degenerate, both eigenvalues are nearly equal
            # Choose the eigenvector that best aligns with the x-axis
            v1 = evecs[:, 0]
            v2 = evecs[:, 1]
            
            # Select the eigenvector with larger x-component
            if abs(v1[0]) >= abs(v2[0]):
                v1_final = v1 if v1[0] >= 0 else -v1
                v2_final = v2 if np.dot(v2, [0, 1]) >= 0 else -v2
            else:
                v1_final = v2 if v2[0] >= 0 else -v2
                v2_final = v1 if np.dot(v1, [0, 1]) >= 0 else -v1
            
            # Ensure orthogonality (Gram-Schmidt)
            v2_final = v2_final - np.dot(v1_final, v2_final) * v1_final
            v2_final /= np.linalg.norm(v2_final)
        else:
            # Non-degenerate case: use eigenvalues to determine ordering
            # Typically, we want the axis with larger moment of inertia as v1
            if evals[0] >= evals[1]:
                v1_final = evecs[:, 0]
                v2_final = evecs[:, 1]
            else:
                v1_final = evecs[:, 1] 
                v2_final = evecs[:, 0]
            
            # Ensure positive x-component for v1 and positive y-component for v2
            if v1_final[0] < 0:
                v1_final = -v1_final
            if v2_final[1] < 0:
                v2_final = -v2_final
        return v1_final,v2_final
def get_principle_axis_coordinates(xyzcoordinates,masses=None,axis=None):
    """
    Centers the molecule in the unit cell, aligns the principal axes, and 
    transforms the coordinates to the principal axis frame.

    Parameters
    ----------
    path : str, optional
        Path to the folder containing the calculation data (default is the current directory "./").

    Returns
    -------
    tuple
        A tuple containing the following elements:
        - principleaxiscoordinates : list of np.ndarray
            List of transformed coordinates (in the principal axis frame) for each particle.
        - masses : np.ndarray
            1D array containing the masses of the atoms.
        - atomicsymbols : list of str
            List of atomic symbols corresponding to the atoms in the system.

    Notes
    -----
    - This function first computes the **center of mass coordinates** to recenter the molecule 
      within the unit cell.
    - It calculates the **principal axes** from the center of mass coordinates and aligns the 
      molecule along the principal axes.
    - The function returns the transformed coordinates in the new frame, along with the masses 
      and atomic symbols.
    """
    if len(xyzcoordinates[0])==3:
        if axis is None:
            if masses is not None:
                centered_coordinates,center=compute_center_of_mass_coordinates(xyzcoordinates,masses)
                v1,v2,v3=get_principle_axis(centered_coordinates,masses)
            else:
                centered_coordinates,center=compute_center_of_geometry_coordinates(xyzcoordinates)
                v1,v2,v3=get_principle_axis(np.array(centered_coordinates))
        else:
            v1=axis[0]
            v2=axis[1]
            v3=axis[2]
        principleaxiscoordinates=[]
        for coordinate in centered_coordinates:
            v1coordinate=np.dot(coordinate,v1)
            v2coordinate=np.dot(coordinate,v2)
            v3coordinate=np.dot(coordinate,v3)
            principleaxiscoordinates.append(np.array([v1coordinate,v2coordinate,v3coordinate]))
    elif len(xyzcoordinates[0])==2:
        if axis is None:
            if masses is not None:
                centered_coordinates,center=compute_center_of_mass_coordinates(xyzcoordinates,masses)
                v1,v2=get_principle_axis(centered_coordinates,masses)
            else:
                centered_coordinates,center=compute_center_of_geometry_coordinates(xyzcoordinates)
                v1,v2=get_principle_axis(np.array(centered_coordinates))
        else:
            v1=axis[0]
            v2=axis[1]
        principleaxiscoordinates=[]
        for coordinate in centered_coordinates:
            v1coordinate=np.dot(coordinate,v1)
            v2coordinate=np.dot(coordinate,v2)
            principleaxiscoordinates.append(np.array([v1coordinate,v2coordinate]))
    return principleaxiscoordinates,center

def getTransAndRotEigenvectors(coordinates,masses):
    """
    Computes the translational and rotational eigenvectors according to "Vibrational Analysis in Gaussian,"
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
    - The function relies on the 'readCoordinatesAndMasses`, `getInertiaTensor`, and `ComputeCenterOfMassCoordinates` functions.
    - The rotational eigenvectors are generated based on the principle axis obtained from the inertia tensor.
    - The translational eigenvectors are generated along each Cartesian axis.
    - The generated eigenvectors can be rescaled based on atom masses if the `rescale` flag is set to True.
    - All generated eigenvectors are normalized.
    """

    #Compute the Intertia Tensor
    I=get_inertia_tensor(coordinates,masses)
    centerofmasscoordinates,_=compute_center_of_mass_coordinates(coordinates,masses)

    #Get the principle Axis
    _,principleAxis=np.linalg.eigh(I)
    numofatoms=int(len(masses))
    Roteigenvectors=[]
    
    for it in [0,1,2]:
        #Define the RotEigenvector
        RotEigenvector=np.zeros(3*numofatoms)
        #generate the vector X along the principle axis
        X=principleAxis[:,it]
        for s in range(numofatoms):
            rvector=np.cross(X,centerofmasscoordinates[s])
            RotEigenvector[3*s]=rvector[0]
            RotEigenvector[3*s+1]=rvector[1]
            RotEigenvector[3*s+2]=rvector[2]
        #Normalize the generated Eigenvector
        RotEigenvector/=np.linalg.norm(RotEigenvector)
        Roteigenvectors.append(RotEigenvector)
    Transeigenvectors=[]
    for it in [0,1,2]:
        #Define the TransEigenvector
        TransEigenvector=np.zeros(3*numofatoms)
        for s in range(numofatoms):
            TransEigenvector[3*s+it]=1
        #Normalize the generated Eigenvector
        TransEigenvector/=np.linalg.norm(TransEigenvector)
        Transeigenvectors.append(TransEigenvector)
    return Transeigenvectors,Roteigenvectors

def generate_internal_representation(
    coordinates_bohr: np.ndarray,
    cell_bohr: np.ndarray,
    atomic_symbols: List[str],
    tol: float = 0.4
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int, int]], List[Tuple[int, int, int, int]]]:
    """
    Generates redundant internal coordinates (bonds, angles, dihedrals).

    This improved version uses itertools for more concise and efficient generation
    of angles and dihedrals.

    Args:
        coordinates_bohr: A NumPy array of shape (n, 3) with atomic coordinates.
        cell_bohr: Unit cell parameters for periodic systems.
        atomic_symbols: A list of atomic symbols.
        tol: The tolerance for bond detection passed to get_bonds().

    Returns:
        A tuple containing three sorted lists for reproducibility:
        - bonds: List of (i, j) tuples with i < j.
        - angles: List of (i, j, k) tuples where j is the central atom and i < k.
        - dihedrals: List of (i, j, k, l) tuples, canonicalized to ensure
                    (i, j, k, l) is chosen over (l, k, j, i) based on
                    lexicographical order.
    """
    n_atoms = len(atomic_symbols)

    # 1. Get the list of bonds (assuming get_bonds returns sorted bonds i < j)
    # This step remains the same.
    bonds = get_bonds(
        coordinates_bohr=coordinates_bohr,
        atomic_symbols=atomic_symbols,
        cell_bohr=cell_bohr,
        tol=tol
    )

    # 2. Build an adjacency list for efficient neighbor lookup
    # This is already an optimal approach.
    adj = {i: [] for i in range(n_atoms)}
    for i, j in bonds:
        adj[i].append(j)
        adj[j].append(i)

    # 3. Generate angles using itertools.combinations
    angles = []
    for j, neighbors in adj.items():
        if len(neighbors) >= 2:
            # itertools.combinations is perfect for picking 2 unique neighbors
            for i, k in itertools.combinations(neighbors, 2):
                # Ensure canonical ordering (i, j, k) where i < k
                angles.append((min(i, k), j, max(i, k)))

    # 4. Generate proper dihedrals using itertools.product
    dihedral_set = set()
    for j, k in bonds:
        i_neighbors = [i for i in adj[j] if i != k]
        l_neighbors = [l for l in adj[k] if l != j]

        if not i_neighbors or not l_neighbors:
            continue

        # itertools.product gives the Cartesian product of the neighbor lists
        for i, l in itertools.product(i_neighbors, l_neighbors):
            # Avoid 3-membered rings (e.g., in cyclopropane) being dihedrals
            if i == l:
                continue
            
            # Canonicalize: a dihedral i-j-k-l is the same as l-k-j-i.
            # We choose the one that comes first lexicographically.
            dihedral = (i, j, k, l)
            reverse_dihedral = (l, k, j, i)
            if dihedral < reverse_dihedral:
                dihedral_set.add(dihedral)
            else:
                dihedral_set.add(reverse_dihedral)

    # Sort all lists for consistent, reproducible output
    bonds.sort()
    angles.sort()
    dihedrals = sorted(list(dihedral_set))
    impropers=get_impropers(n_atoms=n_atoms,adj=adj)
    return bonds, angles, dihedrals,impropers

def get_bonds(
    coordinates_bohr: np.ndarray, 
    cell_bohr: np.ndarray,
    atomic_symbols: List[str],
    tol: float = 0.8
) -> List[Tuple[int, int]]:
    """
    Determines chemical bonds for 3D periodic crystals using covalent radii and minimum image convention.

    Args:
        coordinates: Array of shape (N,3) with atomic coordinates in Angstroms.
        atomic_symbols: List of N atomic symbols.
        cellvectors: 3x3 lattice vectors. If None, non-periodic system assumed.
        tol: tolerance in Angstroms to add to covalent radii sum.

    Returns:
        List of tuples (i,j) with i<j representing bonded atoms.
    """

    # Original covalent radii in Å
    covalent_radii_angstrom = {
        'H': 0.32, 'He': 0.46, 'Li': 1.33, 'Be': 1.02, 'B': 0.85, 'C': 0.75,
        'N': 0.71, 'O': 0.63, 'F': 0.64, 'Ne': 0.67, 'Na': 1.55, 'Mg': 1.39,
        'Al': 1.26, 'Si': 1.16, 'P': 1.11, 'S': 1.03, 'Cl': 0.99, 'Ar': 0.96,
        'K': 1.96, 'Ca': 1.71, 'Sc': 1.48, 'Ti': 1.36, 'V': 1.34, 'Cr': 1.22,
        'Mn': 1.19, 'Fe': 1.16, 'Co': 1.11, 'Ni': 1.10, 'Cu': 1.12, 'Zn': 1.18,
        'Ga': 1.24, 'Ge': 1.21, 'As': 1.21, 'Se': 1.16, 'Br': 1.14, 'Kr': 1.17,
        'I': 1.33, 'X': 0.7
    }

    # Conversion factor Å → Bohr
    ANGSTROM_TO_BOHR = 1.8897259886

    # Convert dictionary
    covalent_radii_bohr = {atom: r * ANGSTROM_TO_BOHR for atom, r in covalent_radii_angstrom.items()}

    n_atoms = len(atomic_symbols)
    if n_atoms < 2:
        return []

    bonds = set()

    # Build inverse lattice for fractional coords
    frac = np.linalg.solve(np.array(cell_bohr).T, np.array(coordinates_bohr).T).T  # N×3
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            try:
                r_sum = covalent_radii_bohr[atomic_symbols[i]] + covalent_radii_bohr[atomic_symbols[j]] + tol
            except KeyError as e:
                raise KeyError(f"Missing covalent radius for {e}") from e

            # minimum image fractional delta
            de = frac[j] - frac[i]
            de -= np.round(de)  # wrap into [-0.5,0.5)
            d_cart = de @ cell_bohr  # Cartesian vector
            dist = np.linalg.norm(d_cart)

            if dist <= r_sum:
                bonds.add((i, j))
    return sorted(bonds)
# Define the typical total bond order (valence) for common elements.
# This dictionary can be expanded as needed.
TYPICAL_VALENCE = {
    # Hydrogen and halogens
    'H': 1, 'F': 1, 'Cl': 1, 'Br': 1, 'I': 1,
    # Chalcogens
    'O': 2, 'S': 2, 'Se': 2, 'Te': 2,
    # Group 15 (pnictogens)
    'N': 3, 'P': 3, 'As': 3, 'Sb': 3, 'Bi': 3,
    # Group 14 (tetrels)
    'C': 4, 'Si': 4, 'Ge': 4, 'Sn': 4, 'Pb': 4,
    # Group 13 (triels)
    'B': 3, 'Al': 3, 'Ga': 3, 'In': 3, 'Tl': 3,
    # Group 12 (zinc group)
    'Zn': 2, 'Cd': 2, 'Hg': 2,
    # Group 11 (coinage metals)
    'Cu': 2, 'Ag': 1, 'Au': 3,
    # Group 10 (nickel group)
    'Ni': 2, 'Pd': 2, 'Pt': 2,
    # Group 9 (cobalt group)
    'Co': 3, 'Rh': 3, 'Ir': 3,
    # Group 8 (iron group)
    'Fe': 3, 'Ru': 3, 'Os': 3,
    # Group 7 (manganese group)
    'Mn': 2, 'Tc': 2, 'Re': 2,
    # Group 6 (chromium group)
    'Cr': 3, 'Mo': 3, 'W': 3,
    # Group 5 (vanadium group)
    'V': 3, 'Nb': 3, 'Ta': 3,
    # Group 4 (titanium group)
    'Ti': 4, 'Zr': 4, 'Hf': 4,
    # Group 3 (scandium group)
    'Sc': 3, 'Y': 3, 'La': 3,
    # Lanthanides (common oxidation state)
    'Ce': 3, 'Pr': 3, 'Nd': 3, 'Sm': 3, 'Eu': 3,
    'Gd': 3, 'Tb': 3, 'Dy': 3, 'Ho': 3, 'Er': 3,
    'Tm': 3, 'Yb': 3, 'Lu': 3,
    # Actinides (common +3 oxidation state)
    'Th': 4, 'U': 4, 'Np': 3, 'Pu': 3, 'Am': 3,
}
def assign_bond_orders(atoms, bonds):
    """
    Determines the bond order for each bond in a molecule based on typical valences.

    Args:
        atoms (list[str]): A list of atom symbols (e.g., ['C', 'O', 'H', ...]).
        bonds (list[tuple[int, int]]): A list of tuples, where each tuple represents a
                                       bond between two atoms by their indices.

    Returns:
        list[float]: A list of bond orders corresponding to the input bonds list.
                     Returns fractional orders for resonant structures.
    """
    
    # 1. Initialize data structures for atoms and bonds
    num_atoms = len(atoms)
    atom_data = [{
        'symbol': atoms[i],
        'neighbors': [],
        'total_valence': TYPICAL_VALENCE.get(atoms[i], 0) # Default to 0 if unknown
    } for i in range(num_atoms)]

    # Use a dictionary for bonds with a canonical key (min_idx, max_idx) for easy lookup
    bond_order_map = collections.OrderedDict()
    for u, v in bonds:
        # Populate neighbor lists for each atom
        atom_data[u]['neighbors'].append(v)
        atom_data[v]['neighbors'].append(u)
        # Initialize all bond orders to None
        bond_order_map[tuple(sorted((u, v)))] = None

    # 2. Iteratively determine bond orders until no more can be uniquely assigned
    # This loop repeats the process, allowing information from one pass (e.g., resolving
    # a C-H bond) to help resolve more complex centers in the next pass.
    progress = True
    while progress:
        progress = False
        # Iterate through atoms sorted by the number of unassigned bonds.
        # This prioritizes atoms that are closer to being fully determined.
        sorted_indices = sorted(range(num_atoms), key=lambda i: sum(
            1 for n in atom_data[i]['neighbors'] if bond_order_map[tuple(sorted((i, n)))] is None
        ))
        
        for idx in sorted_indices:
            atom = atom_data[idx]
            
            unknown_bonds_to = []
            known_order_sum = 0
            
            for neighbor_idx in atom['neighbors']:
                key = tuple(sorted((idx, neighbor_idx)))
                if bond_order_map[key] is None:
                    unknown_bonds_to.append(neighbor_idx)
                else:
                    known_order_sum += bond_order_map[key]

            if not unknown_bonds_to:
                continue

            # Calculate the remaining valence that must be distributed among unknown bonds
            valence_to_distribute = atom['total_valence'] - known_order_sum
            
            # If only one bond is unknown, it must take all the remaining valence
            if len(unknown_bonds_to) == 1:
                neighbor_idx = unknown_bonds_to[0]
                key = tuple(sorted((idx, neighbor_idx)))
                if bond_order_map[key] is None:
                    bond_order_map[key] = valence_to_distribute
                    progress = True # Mark that we made a change
            # Heuristic: If multiple bonds are unknown, distribute remaining valence evenly
            # This handles symmetric cases like the two C-O bonds in a carboxylate group
            # or the two N-O bonds in a nitro group.
            elif len(unknown_bonds_to) > 1 and valence_to_distribute > 0:
                # Check if all neighbors in the unknown set are of the same element type.
                # This is a common pattern for resonance (e.g., -COO-, -NO2).
                first_neighbor_symbol = atom_data[unknown_bonds_to[0]]['symbol']
                is_symmetric = all(atom_data[n]['symbol'] == first_neighbor_symbol for n in unknown_bonds_to)

                if is_symmetric:
                    order = valence_to_distribute / len(unknown_bonds_to)
                    for neighbor_idx in unknown_bonds_to:
                        key = tuple(sorted((idx, neighbor_idx)))
                        if bond_order_map[key] is None:
                            bond_order_map[key] = order
                    progress = True

    # 3. Finalization: Default any remaining undetermined bonds to 1.0
    # This is a safe assumption for complex rings or structures where heuristics fail.
    for key in bond_order_map:
        if bond_order_map[key] is None:
            bond_order_map[key] = 1.0

    # 4. Format the output list to match the original order of the input `bonds`
    #result = [bond_order_map[tuple(sorted(bond))] for bond in bonds]
    return bond_order_map


def get_impropers(
    n_atoms: int, 
    adj: Dict[int, List[int]]
) -> List[Tuple[int, int, int, int]]:
    """
    Identifies improper torsions based on connectivity.

    An improper is defined for any atom bonded to exactly three other atoms.
    The central atom is listed first.
    
    Args:
        n_atoms: The total number of atoms.
        adj: An adjacency list where adj[i] is a list of atoms bonded to atom i.

    Returns:
        A list of improper torsions, where each is a tuple (i, j, k, l)
        with j being the central atom. The peripheral atoms (i, k, l) are sorted.
    """
    impropers = []
    # Iterate through each atom to see if it can be a central atom 'j'
    for j in range(n_atoms):
        neighbors = adj.get(j, [])
        
        # The key condition: the atom must have exactly 3 bonded partners
        if len(neighbors) == 3:
            # Sort the peripheral atoms for a canonical definition
            p1, p2, p3 = sorted(neighbors)
            # The central atom 'j' is the first atom in the tuple
            impropers.append((j,p1, p2, p3))
            
    return impropers

def get_w_vector(u, v):
    # Check if u and v are nearly parallel (cross product would be small)
    cross_prod = np.cross(u, v)
    cross_norm = np.linalg.norm(cross_prod)
    
    if cross_norm > 1e-10:  # Vectors are NOT parallel
        return cross_prod / cross_norm
    
    # Vectors are parallel - find perpendicular vector
    # Try standard basis vectors
    for test_vec in [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]:
        cross_prod = np.cross(u, test_vec)
        cross_norm = np.linalg.norm(cross_prod)
        if cross_norm > 1e-10:
            return cross_prod / cross_norm
    
    # Fallback (shouldn't reach here)
    return np.array([1,0,0])

def cartesian_to_internal_coordinates(coordinates,bonds,angles,dihedrals,impropers):
    q=np.zeros(len(bonds)+len(angles)+len(dihedrals)+len(impropers))
    Bq=np.zeros((len(bonds)+len(angles)+len(dihedrals)+len(impropers),3*len(coordinates)))
    #Bq=np.zeros((len(bonds)+len(angles)+len(dihedrals),3*len(coordinates)))
    for it,bond in enumerate(bonds):
        up=coordinates[bond[1]]-coordinates[bond[0]]
        up_norm=np.linalg.norm(up)
        u=up/up_norm
        q[it]=up_norm
        for xyz_it in [0,1,2]:
            Bq[it,3*bond[1]+xyz_it]=u[xyz_it]
            Bq[it,3*bond[0]+xyz_it]=-u[xyz_it]
        
    for it, angle in enumerate(angles):
        # Atom indices: m-o-n (o is the central atom)
        m_idx, o_idx, n_idx = angle[0], angle[1], angle[2]
        
        # Position vectors
        m = coordinates[m_idx]
        o = coordinates[o_idx]
        n = coordinates[n_idx]

        # Bond vectors pointing away from the central atom
        u_vec = m - o  # Vector from o to m
        v_vec = n - o  # Vector from o to n

        r_u = np.linalg.norm(u_vec)
        r_v = np.linalg.norm(v_vec)
        scale_angle=1
        # Unit vectors
        u = u_vec / r_u
        v = v_vec / r_v

        
        cos_theta = np.dot(u, v)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        
        # Store the calculated angle value
        q[it + len(bonds)] =theta*scale_angle

        # Avoid division by zero for linear angles
        sin_theta = np.sin(theta)
        if sin_theta < 1e-8:
            continue # Gradient is undefined for linear angle, so we skip it


        B_m = (cos_theta * u - v) / (r_u * sin_theta)
        B_n = (cos_theta * v - u) / (r_v * sin_theta)
        B_o = -(B_m + B_n)

        # Populate the B-matrix
        row_idx = it + len(bonds)
        Bq[row_idx, 3*m_idx : 3*m_idx+3] = B_m*scale_angle
        Bq[row_idx, 3*n_idx : 3*n_idx+3] = B_n*scale_angle
        Bq[row_idx, 3*o_idx : 3*o_idx+3] = B_o*scale_angle
    for it, dihedral in enumerate(dihedrals):
        # Atom indices: m-o-p-n
        m_idx, o_idx, p_idx, n_idx = dihedral[0], dihedral[1], dihedral[2], dihedral[3]

        # Position vectors
        m = coordinates[m_idx]
        o = coordinates[o_idx]
        p = coordinates[p_idx]
        n = coordinates[n_idx]
        
        # Bond vectors
        u_vec = m - o  # bond 1-2
        w_vec = p - o  # bond 2-3 (central bond)
        v_vec = n - p  # bond 3-4

        r_u = np.linalg.norm(u_vec)
        r_w = np.linalg.norm(w_vec)
        r_v = np.linalg.norm(v_vec)
        
        #add scale
        scale_dihedrals=1#0.5*r_u+0.5*r_w
        u = u_vec / r_u
        w = w_vec / r_w
        v = v_vec / r_v

        # Normals to the m-o-p and o-p-n planes
        normal_mop = np.cross(u, w)
        normal_opn = np.cross(v, -w) # Use -w to point away from central bond
        
        norm_mop_mag = np.linalg.norm(normal_mop)
        norm_opn_mag = np.linalg.norm(normal_opn)

        # Check for collinear atoms (which make planes undefined)
        if norm_mop_mag < 1e-8 or norm_opn_mag < 1e-8:
            continue # Skip if atoms are linear

        # Calculate the dihedral angle using a stable atan2 method
        # The axis of rotation is the central bond vector w
        y = np.dot(np.cross(normal_mop, normal_opn), w)
        x = np.dot(normal_mop, normal_opn)
        phi = np.arctan2(y, x)
        
        q[it + len(bonds) + len(angles)] = phi*scale_dihedrals
        
        # Pre-calculate angles needed for B-matrix
        cos_theta_u = np.dot(-u, w) # Angle m-o-p
        cos_theta_v = np.dot(v, w)  # Angle o-p-n
        
        # Use magnitudes of cross products for sines to avoid sqrt
        sin_sq_theta_u = norm_mop_mag**2
        sin_sq_theta_v = norm_opn_mag**2

        # B-matrix calculation using standard formulas
        B_m = normal_mop / (r_u * sin_sq_theta_u)
        B_n = -normal_opn / (r_v * sin_sq_theta_v)
        
        B_o = ((r_u * cos_theta_u - r_w) / (r_u * r_w * sin_sq_theta_u)) * normal_mop + \
            (cos_theta_v / (r_w * sin_sq_theta_v)) * normal_opn
            
        B_p = ((r_v * cos_theta_v - r_w) / (r_v * r_w * sin_sq_theta_v)) * normal_opn + \
            (cos_theta_u / (r_w * sin_sq_theta_u)) * normal_mop

        # Populate the B-matrix
        row_idx = it + len(bonds) + len(angles)
        Bq[row_idx, 3*m_idx : 3*m_idx+3] = B_m*scale_dihedrals
        Bq[row_idx, 3*n_idx : 3*n_idx+3] = B_n*scale_dihedrals
        Bq[row_idx, 3*o_idx : 3*o_idx+3] = -(B_m + B_p + B_n)*scale_dihedrals # B_o from formula above
        Bq[row_idx, 3*p_idx : 3*p_idx+3] = B_p*scale_dihedrals
    
    for it,improper in enumerate(impropers):
        COLINEARITY_TOLERANCE = 1e-6
        r1=coordinates[improper[0]]
        r2=coordinates[improper[1]]
        r3=coordinates[improper[2]]
        r4=coordinates[improper[3]]
        a=r3-r2
        b=r4-r2
        c=r1-r2
        n0=np.cross(a,b)
        d=np.linalg.norm(n0)
        if d < COLINEARITY_TOLERANCE:
            print(f"Warning: Collinear atoms found in improper {improper}. Skipping.")
            continue
        n0n=n0/d
        t=np.dot(c,n0n)
        q[it+len(bonds)+len(angles)+len(dihedrals)]=t
        u=c/d
        v=t*n0/d**2
        grad_r1_q=n0n
        grad_a_q=np.cross(u,b)-np.cross(v,b)
        grad_b_q=np.cross(a,u)-np.cross(a,v)
        grad_r2_q=-(grad_r1_q+grad_a_q+grad_b_q)
        grad_r3_q=grad_a_q
        grad_r4_q=grad_b_q
        for xyz_it in [0,1,2]:
            i_index=3*improper[0]+xyz_it
            j_index=3*improper[1]+xyz_it
            k_index=3*improper[2]+xyz_it
            l_index=3*improper[3]+xyz_it
            Bq[it+len(bonds)+len(angles)+len(dihedrals),i_index]=grad_r1_q[xyz_it]
            Bq[it+len(bonds)+len(angles)+len(dihedrals),j_index]=grad_r2_q[xyz_it]
            Bq[it+len(bonds)+len(angles)+len(dihedrals),k_index]=grad_r3_q[xyz_it]
            Bq[it+len(bonds)+len(angles)+len(dihedrals),l_index]=grad_r4_q[xyz_it]

    return q,Bq

def get_B_non_redundant_internal(B_prim):
    """
    Reduce B_prim so that the output has (n - n_remove) rows by
    dropping the lowest-|singular value| directions of B_prim.

    Parameters
    ----------
    B_prim : ndarray, shape (m, n)
        Input matrix.
    n_remove : int
        Number of components to remove relative to the column count n.
        Target kept dimension is (n - n_remove).

    Returns
    -------
    B : ndarray, shape (k, n)
        Projected matrix with k = min(m, n - n_remove) rows.
    U : ndarray, shape (m, k)
        Orthonormal basis (singular vectors of B_prim) used for the projection.
    kept_svals : ndarray, shape (k,)
        Singular values corresponding to the kept directions,
        ordered descending.
    """

    # SVD of B_prim
    U, svals, _ = np.linalg.svd(B_prim, full_matrices=False)
    # Select the k largest singular values
    indices=svals>1e-8
    U_keep = U[:, indices]               # (m, k)
    B = U_keep.T @ B_prim             # (k, n)

    return B, U_keep


def _wrap_to_pi(a):
    """Wrap angles (radians) into (-pi, pi]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi

def _wrap_primitive_dq(dq_prim, n_bonds, n_angles, n_dihedrals, radians=True):
    """
    Wrap angular parts of a primitive internal-coordinate difference vector.
    Order is assumed: [bonds | angles | dihedrals ].
    """
    dq = dq_prim.copy()
    start = n_bonds
    # angles
    if n_angles > 0:
        sl = slice(start, start + n_angles)
        dq[sl] = _wrap_to_pi(dq[sl]) if radians else ((dq[sl] + 180.0) % 360.0) - 180.0
        start += n_angles
    # dihedrals
    if n_dihedrals > 0:
        sl = slice(start, start + n_dihedrals)
        dq[sl] = _wrap_to_pi(dq[sl]) if radians else ((dq[sl] + 180.0) % 360.0) - 180.0
        start += n_dihedrals
    return dq

def _wrap_to_pi(a):
    """Wrap angles (radians) into (-pi, pi]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi

def _wrap_to_180(a):
    """Wrap angles (degrees) into (-180, 180]."""
    return (a + 180.0) % 360.0 - 180.0

def _wrap_primitive_dq(dq_prim, n_bonds, n_angles, n_dihedrals, radians=True):
    """
    Wrap angular parts of a primitive internal-coordinate difference vector.
    Order assumed: [bonds | angles | dihedrals ].
    """
    dq = dq_prim.copy()
    start = 0
    # bonds: (no wrapping)
    start += n_bonds
    # angles
    if n_angles > 0:
        sl = slice(start, start + n_angles)
        dq[sl] = _wrap_to_pi(dq[sl]) if radians else _wrap_to_180(dq[sl])
        start += n_angles
    # dihedrals
    if n_dihedrals > 0:
        sl = slice(start, start + n_dihedrals)
        dq[sl] = _wrap_to_pi(dq[sl]) if radians else _wrap_to_180(dq[sl])
        start += n_dihedrals
    return dq

def backtransform_iterative_refinement(
    x0, s_q, bonds, angles, dihedrals,
    tolerance=1e-10, max_iterations=100, verbose=False,
    max_step=0.3,
    angles_in_radians=True
):
    """
    Iteratively refine Cartesian coordinates to match a target step in
    non-redundant internal coordinates using a FIXED non-redundant basis U.

    Returns the best Cartesian coordinates encountered (flattened).
    """

    x0 = np.asarray(x0).reshape(-1)  # flattened vector
    # reshape helper
    natoms = x0.size // 3

    # --- initial primitives and Jacobian ---
    coords0 = x0.reshape((natoms, 3))
    q0_prim, B0_prim = cartesian_to_internal_coordinates(
        coords0, bonds, angles, dihedrals
    )
    # counts for wrapping
    n_bonds = len(bonds)
    n_angles = len(angles)
    n_dihedrals = len(dihedrals)


    # fixed nonredundant projector U (primitive -> nonredundant)
    B0_nr, U = get_B_non_redundant_internal(B0_prim)
    B0_plus = np.linalg.pinv(B0_nr,rcond=1e-5)
    # check dimension match
    q0_nr = U.T @ q0_prim
    s_q = np.asarray(s_q).reshape(-1)
    if s_q.shape[0] != q0_nr.shape[0]:
        raise ValueError(f"s_q dim {s_q.shape[0]} != nonredundant dim {q0_nr.shape[0]}")

    # --- First estimate x1 = x0 + B^+ s_q (Eq. 13) ---
    dx1 = B0_plus @ s_q
    nrm_dx1 = np.linalg.norm(dx1)
    if nrm_dx1 > max_step:
        dx1 = dx1 * (max_step / nrm_dx1)
    x1 = x0 + dx1
    x = x1.copy()

    # Compute dq1 (primitive) and its nonredundant projection for revert/best tracking
    q1_prim, _ = cartesian_to_internal_coordinates(x1.reshape((natoms, 3)), bonds, angles, dihedrals)
    dq1_prim = q1_prim - q0_prim
    dq1_prim = _wrap_primitive_dq(dq1_prim, n_bonds, n_angles, n_dihedrals, radians=angles_in_radians)
    dq1_nr = U.T @ dq1_prim
    norm_dq1 = np.linalg.norm(dq1_nr)

    # initialize best seen geometry (use x1 as initial best)
    best_x = x1.copy()
    best_norm = norm_dq1

    # convergence thresholds per the referenced procedure
    rms_thresh = tolerance if (tolerance is not None and tolerance > 0) else 1e-6
    rms_change_thresh = 1e-12
    max_iter = min(max_iterations, 25)  # per paper cap

    if verbose:
        print(f"Back-transform refinement: initial ||Δq1|| = {norm_dq1:.3e}, max_iter = {max_iter}")

    prev_rms = None

    for k in range(1, max_iter + 1):
        # compute current primitive internals at x
        qk_prim, _ = cartesian_to_internal_coordinates(x.reshape((natoms, 3)), bonds, angles, dihedrals,)

        # achieved change in primitive internals (wrapped)
        achieved_prim = qk_prim - q0_prim
        achieved_prim = _wrap_primitive_dq(achieved_prim, n_bonds, n_angles, n_dihedrals, radians=angles_in_radians)

        # project to nonredundant and compute delta_q = s_q - achieved_change_nr
        achieved_nr = U.T @ achieved_prim
        delta_q = s_q - achieved_nr
        norm_delta_q = np.linalg.norm(delta_q)

        # Update best seen geometry if improved
        if norm_delta_q < best_norm:
            best_norm = norm_delta_q
            best_x = x.copy()
            if verbose:
                print(f"[iter {k}] New best ||Δq|| = {best_norm:.3e}")

        # Cartesian correction
        dx = B0_plus @ delta_q
        nrm_dx = np.linalg.norm(dx)
        if nrm_dx > max_step:
            dx = dx * (max_step / nrm_dx)
            nrm_dx = np.linalg.norm(dx)

        # RMS of Cartesian correction (per paper)
        rms = np.sqrt(np.mean(dx**2))

        if verbose:
            print(f"[iter {k}] ||Δq||={norm_delta_q:.3e}, RMS(|B^+Δq|)={rms:.3e}")

        # Convergence checks
        if rms <= rms_thresh:
            if verbose:
                print(f"Converged in {k} iterations (RMS {rms:.3e} ≤ {rms_thresh:.3e}).")
            return (x + dx).copy().reshape(-1)
        if prev_rms is not None and abs(prev_rms - rms) <= rms_change_thresh:
            if verbose:
                print(f"Stalled at iter {k} (RMS change ≤ {rms_change_thresh:.1e}). Returning best geometry.")
            return best_x.copy().reshape(-1)

        # Revert safeguard from paper: if delta_q grows past delta_q1, we do NOT immediately revert;
        # instead we keep best_x (which could be x1 if that was best). This matches your request to
        # "take the best stepsize".
        if norm_delta_q > norm_dq1 and verbose:
            print(f"[iter {k}] ||Δq_k|| ({norm_delta_q:.3e}) > ||Δq_1|| ({norm_dq1:.3e}) - continuing but best may be x1")

        # update geometry
        x = x + dx
        prev_rms = rms

    if verbose:
        print(f"Max iterations reached ({max_iter}). Returning best geometry with ||Δq|| = {best_norm:.3e}.")
    return best_x.copy().reshape(-1)

