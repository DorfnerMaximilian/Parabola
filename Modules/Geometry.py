import Modules.Util as util
import Modules.PhysConst as PhysConst
import Modules.Read as Read
import Modules.Write as Write
import numpy as np
import os
pathtocp2k=os.environ["cp2kpath"]
pathtobinaries=pathtocp2k+"/exe/local/"
def getCoordinatesAndMasses(pathtoxyz="./Equilibrium_Geometry/"):
    """
    Helper routine to parse coordinates and masses from an xyz file.

    Parameters:
    - pathtoxyz (str, optional): Path to the corresponding xyz file where the data is saved.

    Returns:
    - coordinates (Nx3 numpy.array): Coordinates of the atoms with respect to some basis.
    - masses (N numpy.array): Masses of the atoms as a numpy array, in the same ordering as the coordinates.
    - atomicsymbols (list): Atomic symbols corresponding to each atom in the same ordering as the coordinates.

    Notes:
    - This function relies on the `getAtomicCoordinates` function and the `StandardAtomicWeights` functions defined in ReadFile & PhysConst.
    - The `getAtomicCoordinates` function is assumed to return a list of atomic coordinates in the xyz file format.
    - The `StandardAtomicWeights` class is assumed to provide standard atomic weights for element symbols.
    - The returned coordinates are in the format of a numpy array with shape (number of atoms, 3).
    - The returned masses are in the format of a numpy array with shape (number of atoms,).
    - The returned atomicsymbols is a list of atomic symbols in the same order as the coordinates.
    """
    SaW=PhysConst.StandardAtomicWeights()
    #Get the Atomic coordinates 
    AtomicCoordinates=Read.readinAtomicCoordinates(pathtoxyz)
    #Parse the Coordinates & the Masses
    coordinates=[]
    masses=[]
    atomicsymbols=[]
    for coords in AtomicCoordinates:
        mass=SaW[coords[1]]
        atomicsymbols.append(coords[1])
        masses.append(mass)
        coordinates.append(np.array([coords[2],coords[3],coords[4]]))
    return coordinates,masses,atomicsymbols

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
def getPrincipleAxis(centerofmasscoordinates,masses):
    """
    Computes the principal axes of a system of particles based on their center-of-mass coordinates and masses. 
    It ensures that the computed principal axes are orthogonal, normalized, and right-handed.

    Parameters
    ----------
    centerofmasscoordinates : np.ndarray
        A 2D array of shape (N, 3) representing the Cartesian coordinates of the particles relative to their center of mass. 
        Each row corresponds to the coordinates of a particle.

    masses : np.ndarray
        A 1D array of length N containing the masses of the particles.

    Returns
    -------
    tuple of np.ndarray
        A tuple containing three 1D arrays (v1, v2, v3) of length 3, which represent the normalized and orthogonal 
        principal axes of the system in the following order:
        - v1: First principal axis
        - v2: Second principal axis
        - v3: Third principal axis (adjusted to ensure a right-handed coordinate system)

    Raises
    ------
    ValueError
        If the principal axes are not orthogonal within a tolerance of 1e-12.
    ValueError
        If the principal axes are not normalized (within 1e-12 tolerance), which may lead to incorrect results.

    Notes
    -----
    - The principal axes are obtained by diagonalizing the inertia tensor of the system.
    - If degeneracy occurs (multiple identical eigenvalues), orthogonality of the eigenvectors might not be guaranteed.
      This implementation raises an error in such cases, suggesting that a more robust handling of degenerate cases is needed.
    - The function ensures that the returned principal axes form a right-handed coordinate system.
    """
    #Compute the Inertia Tensor
    I=getInertiaTensor(centerofmasscoordinates,masses)
    #Get the principle Axis
    _,principleAxis=np.linalg.eigh(I)
    #Check that the principle axis are orthorgonal (Could be not the case in case of degeneracy)
    for it1 in range(3):
        for it2 in range(3):
            if it1!=it2 and np.abs(np.dot(principleAxis[:,it1],principleAxis[:,it2]))>10**(-12):
                ValueError("Principle Axis of Molecules are not orthorgonal! Implement the Degenerate Case!")
            if it1==it2 and np.abs(np.dot(principleAxis[:,it1],principleAxis[:,it2])-1.0)>10**(-12):
                ValueError("Principle Axis are not normalized! This leads to Errorious results!")
    #Represent the centerofMassCoordinates in terms of the principle axis frame
    #Fix Orientation of Principle Axis 
    v1=principleAxis[:,0]
    v2=principleAxis[:,1]
    v3=principleAxis[:,2]
    if v1[0]<0.0:
        v1*=-1.0
    if v2[0]<0.0:
        v2*=-1.0
    #This makes Principle Axis righthanded
    B=np.zeros((3,3))
    B[:,0]=v1
    B[:,1]=v2
    B[:,2]=v3
    determinant=np.linalg.det(B)
    if determinant<0:
        v3*=-1.0
    return v1,v2,v3
def getPrincipleAxisCoordinates(path="./"):
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
    xyzcoordinates,masses,atomicsymbols=getCoordinatesAndMasses(path)
    centerofmasscoordinates,_=ComputeCenterOfMassCoordinates(xyzcoordinates,masses)
    v1,v2,v3=getPrincipleAxis(centerofmasscoordinates,masses)
    principleaxiscoordinates=[]
    for coordinate in centerofmasscoordinates:
        v1coordinate=np.dot(coordinate,v1)
        v2coordinate=np.dot(coordinate,v2)
        v3coordinate=np.dot(coordinate,v3)
        principleaxiscoordinates.append(np.array([v1coordinate,v2coordinate,v3coordinate]))
    
    return principleaxiscoordinates,masses,atomicsymbols

def getNeibouringCellVectors(path,m=1,n=1,l=1):
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
    ConFactors=PhysConst.ConversionFactors()
    cellvectors=[]
    cell=Read.readinCellSize(path)
    for itx in range(-m,m+1,1):
        for ity in range(-n,n+1,1):
            for itz in range(-l,l+1,1):
                vectortoappend=(itx*cell[0]+ity*cell[1]+itz*cell[2])*ConFactors["A->a.u."]
                cellvectors.append(vectortoappend[0])
                cellvectors.append(vectortoappend[1])
                cellvectors.append(vectortoappend[2])
    return cellvectors
def centerMolecule(path="./",Principle_Axis=False):
    """
    Centers a molecule or collection of atoms within the unit cell. It can optionally align 
    the molecule along its principal axes before centering.

    Parameters
    ----------
    path : str, optional
        Path to the folder containing the calculation data (default is "./").

    Principle_Axis : bool, optional
        If True, the function aligns the molecule along its principal axes before centering.
        If False, it uses the raw Cartesian coordinates without alignment (default is False).

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
        If the molecule is too large to fit within the current cell size, suggesting the need for 
        a larger unit cell.

    Notes
    -----
    - The function assumes an **orthorhombic unit cell** (orthogonal basis vectors). If non-orthogonal 
      vectors are detected, it raises an error.
    - If `Principle_Axis` is True, the molecule is first aligned along its principal axes before centering.
    - The function checks whether the molecule fits within the cell; if it doesn’t, it raises an error and suggests 
      increasing the unit cell size.
    """
    cellcoordinates=Read.readinCellSize(path)
    print(cellcoordinates)
    #Check if the cell is orthorhombic, otherwise throw error:
    for cellvector1 in cellcoordinates:
        for cellvector2 in cellcoordinates:
            absscalarproduct =np.abs(np.dot(cellvector1,cellvector2))
            abscrossproduct = np.linalg.norm((np.cross(cellvector1,cellvector2)))
            if absscalarproduct >10**(-10) and abscrossproduct>10**(-10):
                ValueError("Centering of Molecule in Unit Cell makes only sense for Non-Periodic Calculations! Use Orthorhombic Unit cells for this case!")

    #Compute center of Cell (assuming orthogonal basis vectors)
    cellcenter=0.5*cellcoordinates[0]+0.5*cellcoordinates[1]+0.5*cellcoordinates[2]
    centerofcellCoordinates=[]
    geometric_mean=np.array([0.0,0.0,0.0])
    if not Principle_Axis:
        xyzcoordinates,_,atomicsym=getCoordinatesAndMasses(path)
        for coordinate in xyzcoordinates:
            geometric_mean+=coordinate
        geometric_mean/=len(xyzcoordinates)
        for coordinate in xyzcoordinates:
            centerofcellCoordinates.append(coordinate+cellcenter-geometric_mean)
    else:
        xyzcoordinates,_,atomicsym=getPrincipleAxisCoordinates(path)
        for coordinate in xyzcoordinates:
            geometric_mean+=coordinate
        geometric_mean/=len(xyzcoordinates)
        for coordinate in xyzcoordinates:
            centerofcellCoordinates.append(coordinate+cellcenter-geometric_mean)
    #Compute maximum distance of atom from coordinatecenter
    maxdistx=0
    maxdisty=0
    maxdistz=0
    for coordinates in xyzcoordinates:
        distx=np.abs(coordinates[0])
        disty=np.abs(coordinates[1])
        distz=np.abs(coordinates[2])
        if distx > maxdistx:
            maxdistx=distx
        if disty > maxdisty:
            maxdisty=disty
        if distz > maxdistz:
            maxdistz=distz
    #Add cutoff to maximum distance
    cutoff=2
    mincellsizex=cutoff+maxdistx
    mincellsizey=cutoff+maxdisty
    mincellsizez=cutoff+maxdistz
    #Check if molecule fits into box otherwise shift coordinates to center of cell
    
    if mincellsizex<=0.5*distx:
        ValueError("Increase cell size in z direction to at least ",2*mincellsizez)
    if mincellsizey<=0.5*disty:
        ValueError("Increase cell size in y direction to at least ",2*mincellsizey)
    if mincellsizez<=0.5*distz:
        ValueError("Increase cell size in z direction to at least ",2*mincellsizez)
    
    Write.writexyzfile(atomicsym,centerofcellCoordinates,path)
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
def PES_inputs(deltas1,deltas2,vector1,vector2,linktobinary=True,binary="cp2k.popt",parentpath="./",binaryloc=pathtobinaries):
    ConFactors=PhysConst.ConversionFactors()
    #get the Projectname: 
    inp_files = [f for f in os.listdir(parentpath) if f.endswith('.inp')]
    if len(inp_files) != 1:
        raise ValueError('InputError: There should be only one .inp file in the current directory')
    inpfilename = inp_files[0]
    Projectname='emptyString'
    with open(inpfilename,'r') as f:
        for lines in f:
            if len(lines.split())>0:
                if lines.split()[0]=="PROJECT":
                    Projectname=lines.split()[1]
    if Projectname=='emptyString':
        raise ValueError('InputError: Projectname not found!')
    #get xyzfile
    xyz_files = [f for f in os.listdir(parentpath) if f.endswith('.xyz')]
    if len(xyz_files) != 1:
        raise ValueError('InputError: There should be only one .xyz file in the current directory')
    xyzfilename = xyz_files[0]
    Restart_files = [f for f in os.listdir(parentpath) if f.endswith('-RESTART.wfn')]
    if len(Restart_files) != 1:
        raise ValueError('InputError: There should be only one Restart file in the current directory')
    Restart_filename = Restart_files[0]
    if Restart_filename!=Projectname+'-RESTART.wfn':
        raise ValueError('InputError: Project- and Restartfilename differ! Reconsider your input.')
    os.mkdir(parentpath+"Equilibrium_Geometry")
    os.system("cp "+parentpath+inpfilename+" "+parentpath+"Equilibrium_Geometry")
    os.system("cp "+parentpath+xyzfilename+" "+parentpath+"Equilibrium_Geometry")
    os.system("cp "+parentpath+Restart_filename+" "+parentpath+"Equilibrium_Geometry")
    if linktobinary:
        os.system("ln -s "+binaryloc+"/"+binary+" "+parentpath+"Equilibrium_Geometry"+"/")
    for it1 in range(len(deltas1)):
        for it2 in range(len(deltas2)):
            folderlabel='delta1='+str(int(np.abs(deltas1[it1])*100))+'delta2='+str(int(np.abs(deltas2[it2])*100))
            sign1=0.5*(1-np.sign(deltas1[it1]))
            sign2=0.5*(1-np.sign(deltas2[it2]))
            if sign1==0:
                symbolsign1='+'
            if sign1==1:
                symbolsign1='-'
            if sign2==0:
                symbolsign2='+'
            if sign2==1:
                symbolsign2='-'
            work_dir=folderlabel+"sign1="+symbolsign1+"sign2="+symbolsign2
            changeConfiguration2(folderlabel,vector1,np.abs(deltas1[it1])*ConFactors['a.u.->A'],sign1,vector2,np.abs(deltas2[it2])*ConFactors['a.u.->A'],sign2)
            os.system("cp "+parentpath+inpfilename+" "+parentpath+work_dir)
            os.system("cp "+parentpath+Restart_filename+" "+parentpath+work_dir)
            if linktobinary:
                os.system("ln -s "+binaryloc+"/"+binary+" "+parentpath+work_dir+"/")
    np.save("vector1",vector1)
    np.save("vector2",vector2)
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
