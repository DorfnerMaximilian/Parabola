from .PhysConst import ConversionFactors
from . import Read
from . import Write
import numpy as np
import os
def ComputeCenterOfMassCoordinates(coordinates,masses):
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
    return centerofmasscoordinates,centerofmass
def ComputeCenterOfGeometryCoordinates(coordinates):
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
    center=np.array([0.,0.,0.])
    #Compute the center of mass: 
    for it,coords in enumerate(coordinates):
        center+=np.array([coords[0],coords[1],coords[2]])
    center/=len(coordinates)
    
    centercoordinates=[]
    #get the center of mass coordinates
    for coords in coordinates:
        centercoordinates.append(coords-center)
    return centercoordinates,center
def getGeometryCovarianceTensor(coordinates):
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
    centercoordinates,_=ComputeCenterOfGeometryCoordinates(coordinates)
    cov = np.cov(np.array(centercoordinates).T)
    return cov
def getInertiaTensor(coordinates,masses):
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

    #Compute the coordinates in the frame where the origin is the center of mass
    centerofmasscoordinates,_=ComputeCenterOfMassCoordinates(coordinates,masses)
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
def getPrincipleAxis(centeredCoordinates, masses=None, tol=1e-3):
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
        for i in range(3):
            for j in range(i+1, 3):
                if np.abs(evals[i] - evals[j]) / np.mean([evals[i], evals[j]]) < threshold:
                    pairs.append((i,j))
        if not pairs:
            return False, [()]
        else:
            return True, pairs
    if masses is not None:
        # Compute the inertia tensor
        I = getInertiaTensor(centeredCoordinates, masses)
    else:
        I = getGeometryCovarianceTensor(np.array(centeredCoordinates))
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
def getPrincipleAxisCoordinates(xyzcoordinates,masses=None,axis=None):
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
    centered_coordinates=xyzcoordinates
    if axis is None:
        if masses is not None:
            centered_coordinates,_=ComputeCenterOfMassCoordinates(xyzcoordinates,masses)
            v1,v2,v3=getPrincipleAxis(centered_coordinates,masses)
        else:
            centered_coordinates,_=ComputeCenterOfGeometryCoordinates(xyzcoordinates)
            v1,v2,v3=getPrincipleAxis(np.array(centered_coordinates))
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
    
    return principleaxiscoordinates
def get_Geometric_Principle_Axis_Coordinates(path="./"):
    xyzfilename=Read.get_xyz_filename(path,verbose=False)
    coordinates,_,atomic_symbols=Read.read_coordinates_and_masses(path)
    principleaxiscoordinates=getPrincipleAxisCoordinates(coordinates)
    Write.write_xyz_file(atomic_symbols=atomic_symbols, coordinates=principleaxiscoordinates, filename=xyzfilename, path=path, cell_coordinates=None, overwrite=True)
    centerMolecule(path)
def getNeibouringCellVectors(cell,m=1,n=1,l=1):
    """
    Computes the vectors connecting the central cell to its neighboring cells 
    in a periodic system.

    Parameters
    ----------
    path : str
        Path to the folder containing the calculation data, specifically the 
        unit cell dimensions.

    m : int, optional
        Range of neighboring cells to consider in each direction. For example, 
        if `m=1`, it includes the central cell and all adjacent cells in a 
        cubic grid of size (2m + 1)^3 (default is 1).

    Returns
    -------
    list
        A flattened list containing the components of the neighboring cell 
        vectors in atomic units (a.u.). Each triplet in the list corresponds to 
        a neighboring cell vector in the form `[vx, vy, vz]`.

    Notes
    -----
    - **Periodic Boundary Conditions:** This function generates a list of vectors 
      connecting the central cell to its neighbors based on the periodic boundary 
      conditions of the unit cell.
    - **Unit Conversion:** The cell vectors are converted from **Angstroms** to 
      **atomic units (a.u.)** using predefined conversion factors.
    - The total number of vectors generated is `(2m + 1)*(2n + 1)*(2l + 1)`. For example, if `m=n=l=1`, 
      it generates 27 vectors (including the central cell itself).

    Example
    -------
    >>> path = "./calculation_data"
    >>> cell_vectors = getNeibouringCellVectors(path, m=1)
    >>> print(cell_vectors[:9])  # Display the first 9 components
    [0.0, 0.0, 0.0, 5.29, 0.0, 0.0, 0.0, 5.29, 0.0]

    """
    cellvectors=[]
    for itx in range(-m,m+1,1):
        for ity in range(-n,n+1,1):
            for itz in range(-l,l+1,1):
                vectortoappend=(itx*cell[0]+ity*cell[1]+itz*cell[2])*ConversionFactors["A->a.u."]
                cellvectors.append(vectortoappend[0])
                cellvectors.append(vectortoappend[1])
                cellvectors.append(vectortoappend[2])
    return cellvectors
def centerMolecule(path=None):
    """
    Centers a molecule or collection of atoms within the unit cell.

    Parameters
    ----------
    path : str, optional
        Path to the folder containing the calculation data (default is "./").
    overwrite : bool, optional
        Whether to overwrite the existing output file (default is False).

    Returns
    -------
    None
        This function modifies the molecular coordinates and writes the centered molecule to an XYZ file 
        within the specified path.

    Raises
    ------
    ValueError
        If the unit cell is not orthorhombic (i.e., non-orthogonal vectors are detected).
    ValueError
        If the molecule is too large to fit within the current cell size.

    Notes
    -----
    - The function assumes an **orthorhombic unit cell** (orthogonal basis vectors).
    - The function checks whether the molecule fits within the cell; if it doesn’t, it raises an error 
      and suggests increasing the unit cell size.
    """
    if path is None:
        path=os.getcwd()
    xyz_file_path=Read.get_xyz_filename(path=path,verbose=True)
    with open(xyz_file_path,"r") as f:
        lines=f.readlines()
        comment=lines[1][:-1]
    xyzcoordinates, _, atomicsym = Read.read_coordinates_and_masses(xyz_file_path)


    try:
        cellcoordinates = Read.read_cell_vectors(path)
    except:
        centerofgeometry, center = ComputeCenterOfGeometryCoordinates(xyzcoordinates)
        xvalue=2*np.max(np.abs(np.array(xyzcoordinates)[:, 0] - center[0])) + 15
        yvalue=2*np.max(np.abs(np.array(xyzcoordinates)[:, 1] - center[1])) + 15
        zvalue=2*np.max(np.abs(np.array(xyzcoordinates)[:, 2] - center[2])) + 15
        cellcoordinates=[
            np.array([xvalue, 0, 0]),
            np.array([0, yvalue , 0]),
            np.array([0, 0, zvalue])
        ]
        comment+=f";cell=(({xvalue},0,0)(0,{yvalue},0)(0,0,{zvalue}))"

    

    # Check if the cell is orthorhombic, otherwise throw error:
    for cellvector1 in cellcoordinates:
        for cellvector2 in cellcoordinates:
            absscalarproduct = np.abs(np.dot(cellvector1, cellvector2))
            abscrossproduct = np.linalg.norm(np.cross(cellvector1, cellvector2))
            if absscalarproduct > 1e-10 and abscrossproduct > 1e-10:
                raise ValueError("Centering of molecule requires orthorhombic unit cells. Use orthogonal vectors.")

    # Compute center of cell (assuming orthogonal basis vectors)
    cellcenter = 0.5 * cellcoordinates[0] + 0.5 * cellcoordinates[1] + 0.5 * cellcoordinates[2]
    centerofcellCoordinates = []

    for coordinate in centerofgeometry:
        centerofcellCoordinates.append(coordinate + cellcenter)

    # Compute maximum distance of atom from coordinate center
    maxdistx = max(np.abs(coord[0]) for coord in xyzcoordinates)
    maxdisty = max(np.abs(coord[1]) for coord in xyzcoordinates)
    maxdistz = max(np.abs(coord[2]) for coord in xyzcoordinates)

    cutoff = 2
    mincellsizex = cutoff + maxdistx
    mincellsizey = cutoff + maxdisty
    mincellsizez = cutoff + maxdistz

    if mincellsizex <= 0.5 * maxdistx:
        raise ValueError(f"Increase cell size in x direction to at least {2 * mincellsizex}")
    if mincellsizey <= 0.5 * maxdisty:
        raise ValueError(f"Increase cell size in y direction to at least {2 * mincellsizey}")
    if mincellsizez <= 0.5 * maxdistz:
        raise ValueError(f"Increase cell size in z direction to at least {2 * mincellsizez}")
    Write.write_xyz_file(atomicsym, centerofcellCoordinates,xyz_file_path, comment=comment, overwrite=True,success_comment="Successfully centered coordinates in path")

def changeConfiguration(folderlabel,vector,delta,sign,path_xyz='./',path_to="./"):
    """
    Modifies the atomic configuration of an equilibrium XYZ file by shifting the 
    atomic positions along a specified unit vector.

    Parameters
    ----------
    folderlabel : str
        The label used to name the folder where the modified configuration will 
        be saved.

    vector : np.array
        A normalized 1D array of length 3 * (number of atoms) specifying the 
        direction along which the atoms will be shifted.

    delta : float
        The magnitude of the displacement applied to the atomic configuration.

    sign : int
        Determines the direction of the shift:
        - `0`: Positive shift (+delta * vector)
        - `1`: Negative shift (-delta * vector)

    path_xyz : str, optional
        Path to the folder containing the XYZ file of the equilibrium configuration 
        (default is './').

    path_to : str, optional
        Path to the folder where the new configuration folder will be created 
        (default is './').

    Returns
    -------
    None
        The function creates a new folder with the modified XYZ file and saves 
        the updated configuration.

    Raises
    ------
    ValueError
        If there is not exactly one XYZ file in the specified path.

    Notes
    -----
    - The function reads an XYZ file from the specified path, modifies the atomic 
      positions, and saves the new configuration in a newly created folder.
    - The displacement direction is determined by the input `sign` parameter.
    - This function assumes that the input vector is normalized.

    Example
    -------
    >>> folderlabel = "modified_config_"
    >>> vector = np.array([0.0, 1.0, 0.0] * num_atoms)  # Shift along y-axis
    >>> delta = 0.05  # Displacement in the same units as the XYZ file
    >>> changeConfiguration(folderlabel, vector, delta, sign=0)

    """

    #Get the .xyz file
    xyz_files = [f for f in os.listdir(path_xyz) if f.endswith('.xyz')]
    if len(xyz_files) != 1:
        raise ValueError('InputError: There should be only one xyz file in the current directory')
    xyzfilename = xyz_files[0]
    xyzcoordinates=[]
    atomtypes=[]
    with open(path_xyz+"/"+xyzfilename) as g:
        lines = g.readlines()
        for l in lines[2:]:
            if len(l.split())>=4:
                xyzcoordinates.append(float(l.split()[1]))
                xyzcoordinates.append(float(l.split()[2]))
                xyzcoordinates.append(float(l.split()[3]))
                atomtypes.append(l.split()[0])
    xyzcoordinates=np.array(xyzcoordinates)
    xyzcoordinates=xyzcoordinates+(-1)**(sign)*delta*vector
    if sign==0:
        symbolsign='+'
    if sign==1:
        symbolsign='-'
    foldername=path_to+folderlabel+"sign="+symbolsign
    os.mkdir(foldername)
    f=open(foldername+"/"+xyzfilename,"w")
    f.write(lines[0])
    f.write(lines[1])
    for iter in range(int(len(xyzcoordinates)/3)):
        atom=atomtypes[iter]
        xcoord=xyzcoordinates[3*iter]
        ycoord=xyzcoordinates[3*iter+1]
        zcoord=xyzcoordinates[3*iter+2]
        f.write(atom+' '+str(xcoord)+' '+str(ycoord)+' '+str(zcoord))
        f.write("\n")
    f.close()
    return
def changeConfiguration2(folderlabel,vector1,delta1,sign1,vector2,delta2,sign2,path_xyz='./',path_to="./"):
    """
    Modifies the atomic configuration of an equilibrium XYZ file by shifting 
    the atomic positions in the direction of a combined displacement:
    delta1 * vector1 + delta2 * vector2.

    Parameters
    ----------
    folderlabel : str
        The label used to name the folder where the modified configuration will 
        be saved.

    vector1 : np.array
        A normalized vector of length 3 * (number of atoms) specifying the first 
        direction of displacement.

    delta1 : float
        Magnitude of the displacement along `vector1`.

    sign1 : int
        Direction of the displacement along `vector1`:
        - `0`: Positive shift (+delta1 * vector1).
        - `1`: Negative shift (-delta1 * vector1).

    vector2 : np.array
        A normalized vector of length 3 * (number of atoms) specifying the second 
        direction of displacement.

    delta2 : float
        Magnitude of the displacement along `vector2`.

    sign2 : int
        Direction of the displacement along `vector2`:
        - `0`: Positive shift (+delta2 * vector2).
        - `1`: Negative shift (-delta2 * vector2).

    path_xyz : str, optional
        Path to the folder containing the equilibrium XYZ file (default is './').

    path_to : str, optional
        Path to the folder where the new configuration folder will be created 
        (default is './').

    Returns
    -------
    None
        Creates a new folder containing the modified XYZ file with the updated configuration.

    Raises
    ------
    ValueError
        If there is not exactly one XYZ file in the specified path.

    Notes
    -----
    - This function reads an XYZ file, modifies the atomic positions using two 
      displacements along `vector1` and `vector2`, and saves the modified 
      configuration to a new folder.
    - The displacement is calculated as:
      `(-1)^sign1 * delta1 * vector1 + (-1)^sign2 * delta2 * vector2`.
    - The input vectors should be normalized to ensure correct displacement scaling.

    Example
    -------
    >>> folderlabel = "double_shift_"
    >>> vector1 = np.array([1.0, 0.0, 0.0] * num_atoms)  # x-axis
    >>> vector2 = np.array([0.0, 1.0, 0.0] * num_atoms)  # y-axis
    >>> delta1 = 0.1  # Shift along x-axis
    >>> delta2 = 0.2  # Shift along y-axis
    >>> changeConfiguration2(folderlabel, vector1, delta1, 0, vector2, delta2, 1)

    """
    #Get the .xyz file
    xyz_files = [f for f in os.listdir(path_xyz) if f.endswith('.xyz')]
    if len(xyz_files) != 1:
        raise ValueError('InputError: There should be only one xyz file in the current directory')
    xyzfilename = xyz_files[0]
    xyzcoordinates=[]
    atomtypes=[]
    with open(path_xyz+"/"+xyzfilename) as g:
        lines = g.readlines()
        for l in lines[2:]:
            if len(l.split())>=4:
                xyzcoordinates.append(float(l.split()[1]))
                xyzcoordinates.append(float(l.split()[2]))
                xyzcoordinates.append(float(l.split()[3]))
                atomtypes.append(l.split()[0])
    xyzcoordinates=np.array(xyzcoordinates)
    xyzcoordinates=xyzcoordinates+(-1)**(sign1)*delta1*vector1+(-1)**(sign2)*delta2*vector2
    symbolsign1="+"
    symbolsign2="+"
    if sign1==0:
        symbolsign1='+'
    if sign1==1:
        symbolsign1='-'
    if sign2==0:
        symbolsign2='+'
    if sign2==1:
        symbolsign2='-'
    foldername=path_to+folderlabel+"sign1="+symbolsign1+"sign2="+symbolsign2
    os.mkdir(foldername)
    f=open(foldername+"/"+xyzfilename,"w")
    f.write(lines[0])
    f.write(lines[1])
    for iter in range(int(len(xyzcoordinates)/3)):
        atom=atomtypes[iter]
        xcoord=xyzcoordinates[3*iter]
        ycoord=xyzcoordinates[3*iter+1]
        zcoord=xyzcoordinates[3*iter+2]
        f.write(atom+' '+str(xcoord)+' '+str(ycoord)+' '+str(zcoord))
        f.write("\n")
    f.close()
    return
def getNewXYZ(path='.'):
    """
    Extracts the atomic coordinates from the last iteration of a geometry 
    optimization (`GeoOpt`) file and creates a new XYZ file named 
    `<original_xyz_filename>_opt.xyz` for use in vibrational analysis.

    Parameters
    ----------
    path : str, optional
        Path to the folder containing the geometry optimization results 
        (default is the current directory './').

    Returns
    -------
    None
        A new optimized XYZ file is generated in the same directory, with the 
        name `<original_xyz_filename>_opt.xyz`.

    Raises
    ------
    ValueError
        If there is not exactly one `.inp` file in the specified directory.
    ValueError
        If there is not exactly one old XYZ file in the directory (after excluding 
        the `-pos-1.xyz` file).

    Notes
    -----
    - This function reads the geometry optimization data from the last iteration 
      of the `-pos-1.xyz` file and generates a new XYZ file for further analysis.
    - The script assumes the project name is declared in the `.inp` file using 
      the `PROJECT` keyword.

    Example
    -------
    If your directory contains:
      - `myproject.inp` with a line `PROJECT myproject`
      - `myproject-pos-1.xyz` containing optimization iterations
      - `molecule.xyz` (initial XYZ file)

    Running `getNewXYZ()` will create `molecule_opt.xyz` containing the atomic 
    coordinates from the final iteration of `myproject-pos-1.xyz`.

    """

    #get the Projectname
    inp_files = [f for f in os.listdir(path) if f.endswith('.inp')]
    if len(inp_files) != 1:
        raise ValueError('InputError: There should be only one inp file in the current directory')
    filename = path+"/"+inp_files[0]
    f = open(filename, "r")
    projectname='NoName'
    for line in f.readlines():
        if len(line.split())>0:
            if line.split()[0]=="PROJECT":
                projectname=line.split()[1]
                print(projectname)
    f.close()
    xyz_files = [f for f in os.listdir(path) if f.endswith('.xyz')]
    #remove the projectname+'-pos-1.xyz' file from xyz files
    xyz_files.remove(projectname+'-pos-1.xyz')
    if len(xyz_files) != 1:
        raise ValueError('InputError: There should be only one old xyz file in the current directory')
    xyzfilename=path+"/"+xyz_files[0]
    iter2=0
    if projectname!='NoName':
        numiter=0
        buffer=[]
        getflag=False
        f=open(path+'/'+projectname+'-pos-1.xyz')
        for line in f.readlines():
            if len(line.split())>0:
                if line.split()[0]=='i':
                    numiter+=1
        f.close()
        f=open(path+'/'+projectname+'-pos-1.xyz')
        for line in f.readlines():
            if getflag:
                buffer.append(line.split())
            if len(line.split())>0:
                if line.split()[0]=='i':
                    iter2+=1
                if iter2==numiter:
                    getflag=True
            else:
                continue
        f.close()
        f=open(xyzfilename,'r')
        header=f.readlines()[0:2]
        f.close()
        g=open(xyzfilename[:-4]+'_opt.xyz','w')
        g.write(header[0])
        g.write(header[1])
        for line in buffer:
            for element in line:
                g.write(element)
                g.write(' ')
            g.write('\n')
        g.close()
