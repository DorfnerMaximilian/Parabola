import numpy as np
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

def calculate_distance(p1, p2):
    """Calculates the Euclidean distance between two points."""
    return np.linalg.norm(p1 - p2)

def calculate_angle(p1, p2, p3):
    """Calculates the bond angle (in degrees) for a P1-P2-P3 triplet."""
    vec_p2p1 = p1 - p2
    vec_p2p3 = p3 - p2
    dot_product = np.dot(vec_p2p1, vec_p2p3)
    norm_product = np.linalg.norm(vec_p2p1) * np.linalg.norm(vec_p2p3)
    if norm_product == 0:
        return 0.0
    cos_theta = np.clip(dot_product / norm_product, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

def calculate_dihedral(p1, p2, p3, p4):
    """
    Calculates the dihedral angle (in degrees) for a P1-P2-P3-P4 quartet.
    The sign of the angle is determined by the cross product.
    """
    vec_p2p1 = p1 - p2
    vec_p3p2 = p2 - p3
    vec_p4p3 = p3 - p4

    # Plane normal vectors
    n1 = np.cross(vec_p2p1, vec_p3p2)
    n2 = np.cross(vec_p3p2, vec_p4p3)

    if np.linalg.norm(n1) == 0 or np.linalg.norm(n2) == 0:
        return 0.0

    n1 /= np.linalg.norm(n1)
    n2 /= np.linalg.norm(n2)
    
    # Cosine of the angle between the two planes
    cos_phi = np.clip(np.dot(n1, n2), -1.0, 1.0)
    
    # Determine the sign
    sign = np.sign(np.dot(np.cross(n1, n2), vec_p3p2))
    
    return np.degrees(np.arccos(cos_phi)) * sign

def xyz_to_zmatrix(atoms, coords):
    """
    Converts a list of atoms and their Cartesian coordinates to a Z-matrix.
    The algorithm assumes atoms are placed one by one, using previously placed
    atoms as reference points.
    """
    z_matrix = []
    num_atoms = len(atoms)

    # Atom 1: No internal coordinates
    z_matrix.append([atoms[0]])

    if num_atoms > 1:
        # Atom 2: Defined by a bond length to atom 1
        r12 = calculate_distance(coords[0], coords[1])
        z_matrix.append([atoms[1], 1, r12])

    if num_atoms > 2:
        # Atom 3: Defined by bond length to atom 2, and angle to atom 1
        r23 = calculate_distance(coords[1], coords[2])
        theta123 = calculate_angle(coords[0], coords[1], coords[2])
        z_matrix.append([atoms[2], 2, r23, 1, theta123])

    if num_atoms > 3:
        # Atoms 4 to N: Defined by bond length, bond angle, and dihedral angle
        for i in range(3, num_atoms):
            # For simplicity, we use the three most recently placed atoms as reference
            # for the current atom.
            ref_i, ref_j, ref_k = i, i-1, i-2
            
            # Find the best reference atoms based on proximity to ensure
            # they are bonded. This is a simple heuristic.
            dist = [calculate_distance(coords[i], coords[j]) for j in range(i)]
            j_idx = np.argmin(dist)
            
            dist_j = [calculate_distance(coords[j_idx], coords[k]) for k in range(j_idx)]
            k_idx = np.argmin(dist_j)
            
            # Find a third reference atom for dihedral
            l_idx = -1
            for l in range(k_idx):
                if l != j_idx and l != k_idx:
                    l_idx = l
                    break
            if l_idx == -1: # Fallback
                l_idx = k_idx -1

            r_ij = calculate_distance(coords[i], coords[j_idx])
            theta_ijk = calculate_angle(coords[i], coords[j_idx], coords[k_idx])
            phi_ijkl = calculate_dihedral(coords[i], coords[j_idx], coords[k_idx], coords[l_idx])

            z_matrix.append([atoms[i], j_idx + 1, r_ij, k_idx + 1, theta_ijk, l_idx + 1, phi_ijkl])

    return z_matrix