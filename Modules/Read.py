import numpy as np
import os
from .PhysConst import StandardAtomicWeights,ConversionFactors
"####################################################################################"
"#########                            xyz-Read                           ############"
"####################################################################################"
def get_xyz_filename(path="./",verbose=True):
    """
    Finds a unique geometry file, prioritizing the optimized ('*_opt.xyz') version.

    The function first searches for an optimized file. If exactly one is found,
    it's used. Otherwise, it searches for a standard '*.xyz' file. A message
    is printed to clarify which file is selected.
    
    Args:
        path (str): The directory path to search in.
        
    Returns:
        str: The full path to the selected .xyz file.
        
    Raises:
        ValueError: If multiple files of the same type are found.
        FileNotFoundError: If no suitable geometry file is found at all.
    """
    # 1. Prioritize the optimized '_opt.xyz' file
    opt_files = [f for f in os.listdir(path) if f.endswith('_opt.xyz')]
    if len(opt_files) == 1:
        filename = opt_files[0]
        if verbose:
            print(f"ℹ️ : Found and selected geometry from file: {filename}")
        return os.path.abspath(os.path.join(path, filename))
    elif len(opt_files) > 1:
        raise ValueError(f"AmbiguityError: Found {len(opt_files)} '*_opt.xyz' files in '{path}'. Please keep only one.")
     # 2. Next prioritize the temp '_tmp.xyz' file
    opt_files = [f for f in os.listdir(path) if f.endswith('_tmp.xyz')]
    if len(opt_files) == 1:
        filename = opt_files[0]
        if verbose:
            print(f"ℹ️ : Found and selected geometry from file: {filename}")
        return os.path.abspath(os.path.join(path, filename))
    elif len(opt_files) > 1:
        raise ValueError(f"AmbiguityError: Found {len(opt_files)} '*_tmp.xyz' files in '{path}'. Please keep only one.")
    
    # 3. If no optimized file, fall back to standard '.xyz' file
    # Ensure we don't accidentally match an '_opt.xyz' file here
    xyz_files = [f for f in os.listdir(path) if f.endswith('.xyz') and not f.endswith('_opt.xyz') and not f.endswith('_tmp.xyz')]

    if len(xyz_files) == 1:
        filename = xyz_files[0]
        if verbose:
            print(f"ℹ️ : No optimized xyz file found. Selected standard file: {filename}")
        return os.path.abspath(os.path.join(path, filename))
    elif len(xyz_files) > 1:
        raise ValueError(f"AmbiguityError: Found {len(xyz_files)} standard '*.xyz' files in '{path}'. Please keep only one.")
    
    # 3. If no files of either type are found
    raise FileNotFoundError(f"InputError: No suitable '*.xyz' or '*_opt.xyz' file found in '{path}'.")
def get_cell_filename(path="./",verbose=True):
    """
    Finds a unique geometry file, prioritizing the optimized ('*_opt.xyz') version.

    The function first searches for an optimized file. If exactly one is found,
    it's used. Otherwise, it searches for a standard '*.xyz' file. A message
    is printed to clarify which file is selected.
    
    Args:
        path (str): The directory path to search in.
        
    Returns:
        str: The full path to the selected .xyz file.
        
    Raises:
        ValueError: If multiple files of the same type are found.
        FileNotFoundError: If no suitable geometry file is found at all.
    """
    # 1. Prioritize the optimized '_opt.cell' file
    opt_files = [f for f in os.listdir(path) if f.endswith('_opt.cell')]
    if len(opt_files) == 1:
        filename = opt_files[0]
        if verbose:
            print(f"ℹ️ : Found and selected cell from file: {filename}")
        return os.path.abspath(os.path.join(path, filename))
    elif len(opt_files) > 1:
        raise ValueError(f"AmbiguityError: Found {len(opt_files)} '*_opt.cell' files in '{path}'. Please keep only one.")

    # 2. If no optimized file, fall back to standard '.xyz' file
    # Ensure we don't accidentally match an '_opt.xyz' file here
    cell_files = [f for f in os.listdir(path) if f.endswith('.cell') and not f.endswith('_opt.cell')]

    if len(cell_files) == 1:
        filename = cell_files[0]
        if verbose:
            print(f"ℹ️ : No optimized cell file found. Selected standard file: {filename}")
        return os.path.abspath(os.path.join(path, filename))
    elif len(cell_files) > 1:
        raise ValueError(f"AmbiguityError: Found {len(cell_files)} standard '*.cell' files in '{path}'. Please keep only one.")
    
    # 3. If no files of either type are found
    raise FileNotFoundError(f"InputError: No suitable '*.cell' or '*_opt.cell' file found in '{path}'.")
def read_cell(path):
    """
    Parse a CP2K cell line string (from parse_cell_line format).

    Parameters:
        line : str
            A single line with: itimes, time, h(1,1)...h(3,3), vol

    Returns:
        dict with keys:
            - itimes (int)
            - time (float)
            - h (3x3 numpy array, rows = lattice vectors A,B,C)
            - vol (float)
    """
    with open(path,"r") as f:
        line=f.readline()
    parts = line.split()

    # Next 9 values: h matrix (column-major in file)
    h_colwise = list(map(float, parts[2:11]))
    h = np.array(h_colwise).reshape(3,3, order="F")  # column-major → 3x3 matrix

    return h
def read_atomic_coordinates(path):
    ##Reads in the atomic coordinates from a provided xyz file (these coordinates are independent from cell vectors)! 
    ## input:
    ## (opt.)   folder              path to the folder of the .xyz file         (string)
    ## output:  Atoms               list of sublists. 
    ##                              Each of the sublists has five elements. 
    ##                              Sublist[0] contains the atomorder as a int.
    ##                              Sublist[1] contains the symbol of the atom.
    ##                              Sublist[2:] containst the x y z coordinates.
    Atoms=[]
    with open(path) as f:
        lines=f.readlines()
        it=1
        for l in lines[2:]:
            Atoms.append([it,l.split()[0],float(l.split()[1]),float(l.split()[2]),float(l.split()[3])])
            it+=1
    f.close()
    return Atoms
def read_coordinates_and_masses(filename):
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
    #Get the Atomic coordinates 
    AtomicCoordinates=read_atomic_coordinates(filename)
    #Parse the Coordinates & the Masses
    coordinates=[]
    masses=[]
    atomicsymbols=[]
    for coords in AtomicCoordinates:
        mass=StandardAtomicWeights[coords[1]]
        atomicsymbols.append(coords[1])
        masses.append(mass)
        coordinates.append(np.array([coords[2],coords[3],coords[4]]))
    return coordinates,masses,atomicsymbols
"####################################################################################"
"#########                           END xyz-Read                        ############"
"####################################################################################"

"####################################################################################"
"#########                           CP2K- Read                          ############"
"####################################################################################"

####################################################################################
#########                    General input and Output                   ############
####################################################################################
def get_inp_filename(path="./", verbose=True):
    """
    Finds a unique CP2K input file (*.inp) in the given directory.

    Args:
        path (str): Directory path to search in.
        verbose (bool): Whether to print informative messages.
        
    Returns:
        str: Full path to the selected .inp file.
        
    Raises:
        ValueError: If multiple .inp files are found.
        FileNotFoundError: If no .inp file is found.
    """
    inp_files = [f for f in os.listdir(path) if f.endswith('.inp')]

    if len(inp_files) == 1:
        filename = inp_files[0]
        if verbose:
            print(f"ℹ️ : Selected input file: {filename}")
        return os.path.abspath(os.path.join(path, filename))
    elif len(inp_files) > 1:
        raise ValueError(f"AmbiguityError: Found {len(inp_files)} '*.inp' files in '{path}'. Please keep only one.")
    else:
        raise FileNotFoundError(f"InputError: No '*.inp' file found in '{path}'.")

def get_out_filename(path="./", verbose=True):
    """
    Finds the unique .out file in the given directory.

    Args:
        path (str): Directory path to search in.
        verbose (bool): Whether to print informative messages.

    Returns:
        str: Full path to the selected .out file.

    Raises:
        ValueError: If multiple .out files are found.
        FileNotFoundError: If no .out file is found.
    """
    out_files = [f for f in os.listdir(path) if f.endswith('.out')]

    if len(out_files) == 1:
        filename = out_files[0]
        if verbose:
            print(f"ℹ️ : Selected output file: {filename}")
        return os.path.abspath(os.path.join(path, filename))
    elif len(out_files) > 1:
        raise ValueError(f"AmbiguityError: Found {len(out_files)} '*.out' files in '{path}'. Please keep only one.")
    else:
        raise FileNotFoundError(f"InputError: No '*.out' file found in '{path}'.")

####################################################################################
#########                    End General input and Output               ############
####################################################################################

####################################################################################
#########                    Forces, Hessian & Vibrations               ############
####################################################################################
def read_forces(folder):
    ## Reads the last instance of atomic forces from the Forces file in folder
    filename = folder + "/Forces"
    forces_blocks = []  # to keep all blocks
    
    with open(filename, "r") as f:
        lines = f.readlines()

    reading = False
    current_forces = []

    for line in lines:
        parts = line.split()
        if len(parts) >= 2:
            if parts[0] == "#" and parts[1] == "Atom":
                # Start new block
                reading = True
                current_forces = []
                continue
            
            if parts[0] == "SUM" and parts[1] == "OF":
                # End of current block
                reading = False
                if current_forces:
                    forces_blocks.append(current_forces)
                continue

            if reading:
                # Parse forces from columns 4,5,6 (0-based indexing 3,4,5)
                try:
                    fx = float(parts[3])
                    fy = float(parts[4])
                    fz = float(parts[5])
                    current_forces.append([fx, fy, fz])
                except (IndexError, ValueError):
                    # Malformed line, skip
                    continue

    # If no forces found by first method, try alternative format
    if not forces_blocks:
        reading = False
        current_forces = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                if parts[0] == "FORCES|" and parts[1] == "Atom":
                    reading = True
                    current_forces = []
                    continue
                if parts[0] == "FORCES|" and parts[1] == "Sum":
                    reading = False
                    if current_forces:
                        forces_blocks.append(current_forces)
                    continue
                if reading:
                    try:
                        fx = float(parts[2])
                        fy = float(parts[3])
                        fz = float(parts[4])
                        current_forces.append([fx, fy, fz])
                    except (IndexError, ValueError):
                        continue

    # Return the last forces block or empty list if none found
    if forces_blocks:
        return forces_blocks[-1]
    else:
        return []
def read_basis_vectors(parentfolder="./"):
    """
    Reads in the normalized Basis vectors in which the Hessian is represented.
    These are the directions in which the atoms have been displaced.
    Also returns the displacement factors (deltas) and the unit of the displacement factor.
    
    Parameters:
    parentfolder (str): Path to the folder containing the BasisHessian file.
    
    Returns:
    BasisVectors (list of np.arrays): List of normalized displaced vectors.
    deltas (np.array): Array of displacement factors.
    unit (str): Unit of the displacement factor, either 'Bohr' or 'sqrt(u)*Bohr'.
    """

    xyzfilename=get_xyz_filename(parentfolder,verbose=False)
    numofatoms = 0
    with open(xyzfilename) as g:
        lines = g.readlines()
        numofatoms = int(lines[0])

    # Read in the Basis Vectors and deltas
    BasisVectors = []
    deltas = []
    with open(os.path.join(parentfolder, "BasisHessian")) as g:
        lines = g.readlines()

        # Parse the delta line
        if lines[0].startswith("delta="):
            delta_values = [float(val) for val in lines[0].split("=")[1].split(",")]
        else:
            raise ValueError('InputError: delta not provided in the expected format.')

        # If only one delta is provided, use it for all basis vectors
        if len(delta_values) == 1:
            single_delta = delta_values[0]
        else:
            single_delta = None

        # Parse the unit line
        if lines[1].startswith("unit="):
            unit = lines[1].split("=")[1].strip()
            if unit not in ["Bohr", "sqrt(u)*Bohr"]:
                raise ValueError("InputError: Unit must be either 'Bohr' or 'sqrt(u)*Bohr'")
        else:
            raise ValueError('InputError: Unit not provided in the expected format.')

        # Parse the basis vectors from line 4 onward
        for line in lines[4:]:
            if line.strip():
                if line.startswith("Basisvector"):
                    if "basevector" in locals():
                        BasisVectors.append(basevector)
                    basevector = np.zeros(3 * numofatoms)
                    vector_index = int(line.split()[1]) - 1  # zero-indexed for Python
                    
                    # Assign delta to each basis vector as specified, or use default if single delta
                    if single_delta is not None:
                        deltas.append(single_delta)
                    elif vector_index < len(delta_values):
                        deltas.append(delta_values[vector_index])
                    else:
                        raise ValueError("InputError: Insufficient delta values provided for basis vectors.")
                    
                    it = 0
                else:
                    basevector[it] = float(line.split()[1])
                    basevector[it + 1] = float(line.split()[2])
                    basevector[it + 2] = float(line.split()[3])
                    it += 3
        BasisVectors.append(basevector)  # Append the last basis vector

        # Ensure deltas is a numpy array for easier handling
        deltas = np.array(deltas)

    return BasisVectors, deltas

def read_hessian(path="./"):
    try:
        Hessian=np.load("Hessian.npy")
    except:
        #Read in the basis vectors of the finite displacements:
        BasisVectors,deltas=read_basis_vectors(path+"/")
        #Built the T matrix from b basis 
        T=np.zeros((len(deltas),len(deltas)))
        for salpha in range(len(deltas)):
            for lambd in range(len(deltas)):
                T[salpha][lambd]=BasisVectors[lambd][salpha]
        #invert the Tmatrix:
        Tinv=np.linalg.inv(T)
        ##########################################
        #get the subdirectories
        ##########################################
        Hessian=np.zeros((len(deltas),len(deltas)))
        subdirectories=[f for f in os.listdir(path) if f.startswith('vector=')]
        if len(subdirectories)!=2*len(deltas):
            raise ValueError('InputError: Number of subdirectories does not match the number of needed Ones!')
        
        partialFpartialY=np.zeros((len(deltas),len(deltas)))
        for lambd in range(len(deltas)):
            folderplus='vector='+str(lambd+1)+'sign=+'
            folderminus='vector='+str(lambd+1)+'sign=-'
            #Project out the Force components into direction of rotation and translation D.O.F.
            try:
                Fplus=np.array(read_forces(path+"/"+folderplus))
                if len(Fplus)==0:
                    print("Error in folder: "+path+"/"+folderplus)
                
            except:
                print("Error in folder: "+path+"/"+folderplus)
                exit()
            try:
                Fminus=np.array(read_forces(path+"/"+folderminus))
                if len(Fminus)==0:
                    print("Error in folder: "+path+"/"+folderminus)
            except:
                print("Error in folder: "+path+"/"+folderminus)
                exit()
            try:
                diffofforces=(Fplus-Fminus)/(2*deltas[lambd])
            except:
                print("Cannot take difference of Forces in: "+folderplus+" & "+folderminus)
                exit()
            for s1alpha1 in range(len(deltas)):
                partialFpartialY[s1alpha1][lambd]=diffofforces.flatten()[s1alpha1]
        Hessian=-partialFpartialY@Tinv
        np.save("Hessian.npy",Hessian)
    return Hessian

def read_vibrations(parentfolder="./"):
    try:
        VibrationalFrequencies=np.load(parentfolder+"/Normal-Mode-Energies.npy")
        CarthesianDisplacements=np.load(parentfolder+"/normalized-Carthesian-Displacements.npy")
        normfactors=np.load(parentfolder+"/Norm-Factors.npy")
    except:   
        #Open the Molden file
        mol_files = [f for f in os.listdir(parentfolder) if f.endswith('.mol')]
        if len(mol_files) != 1:
            raise ValueError('InputError: There should be only one mol file in the current directory')
        f = open(parentfolder+"/"+mol_files[0], "r")
        Frequencyflag=False
        Vibrationflag=False
        Normfactorsflag=False
        numvib=0
        CarthesianDisplacements=[]
        normfactors=[]
        mode=[]
        VibrationalFrequencies=[]
        for line in f.readlines():
            if line.split()[0]=="[FREQ]":
                Frequencyflag=True
            if line.split()[0]=="[NORM-FACTORS]":
                Normfactorsflag=True
            if line.split()[0][0]=="[" and Frequencyflag and line.split()[0]!="[FREQ]":
                Frequencyflag=False
            if line.split()[0][0]=="[" and Normfactorsflag and line.split()[0]!="[NORM-FACTORS]":
                Normfactorsflag=False
            if line.split()[0]=="[FR-NORM-COORD]":
                Vibrationflag=True
            if Frequencyflag and line.split()[0]!="[FREQ]":
                VibrationalFrequencies.append(float(line.split()[0]))
            if Normfactorsflag and line.split()[0]!="[NORM-FACTORS]":
                normfactors.append(float(line.split()[0]))
            if Vibrationflag and line.split()[0]!="[FR-NORM-COORD]":
                if line.split()[0]=="vibration":
                    numvib=int(line.split()[1])
                    if numvib!=1:
                        CarthesianDisplacements.append(np.array(mode))
                    mode=[]
                else:
                    mode.append(float(line.split()[0]))
                    mode.append(float(line.split()[1]))
                    mode.append(float(line.split()[2]))
            if line.split()[0][0]=="[" and Vibrationflag and line.split()[0]!="[FR-NORM-COORD]":
                Vibrationflag=False
        f.close()
        CarthesianDisplacements.append(np.array(mode))
    return VibrationalFrequencies,CarthesianDisplacements,normfactors

def read_stress(folder,filename="Stress_Tensor"):
    """
    Parse CP2K output for the analytical stress tensor [GPa].

    Parameters
    ----------
    filename : str
        Path to CP2K output file.

    Returns
    -------
    stress : np.ndarray, shape (3,3)
        Stress tensor in GPa.
    """
    with open(folder+"/"+filename, "r") as f:
        lines = f.readlines()

    stress = None
    for i, line in enumerate(lines):
        if "STRESS| Analytical stress tensor" in line:
            # the tensor is always in the next 3 lines (x, y, z rows)
            mat = []
            for l in lines[i+2:i+5]:
                parts = l.split()
                # Example line: "STRESS|      x   -6.92 ... -1.31 ... 5.52e-04"
                row = [float(val) for val in parts[2:5]]
                mat.append(row)
            stress = np.array(mat, dtype=float)
            break

    if stress is None:
        raise ValueError("Stress tensor not found in file.")

    return stress
####################################################################################
#########                    END Forces, Hessian & Vibrations               ########
####################################################################################

####################################################################################
#########                    Cell-Vectors, Periodicity                  ############
####################################################################################
def read_cell_vectors(path="./",verbose=True):
    ## Function read out the cell size from the .inp file
    ## input: (opt.)   path   path to the folder of the calculation         (string)
    ## output:  -                                                           (void)

    #Get the .inp file
    inp_file=get_inp_filename(path=path, verbose=False)
    if verbose:
        print(f"ℹ️ : Reading cell information from file: {inp_file}")
    cellvectors=[np.zeros((3,1)),np.zeros((3,1)),np.zeros((3,1))]
    with open(inp_file) as f:
        lines = f.readlines()
        Cellflag=False
        for l in lines:
            if len(l.split())>=1:
                if l.split()[0]=='&CELL':
                    Cellflag=True
                if len(l.split())>=2:
                    if l.split()[0]=='&END' and l.split()[1]=='CELL':
                        Cellflag=False
                if Cellflag:
                    if l.split()[0]=='ABC':
                        cellvectors[0]=np.array([float(l.split()[1]),0.0,0.0])
                        cellvectors[1]=np.array([0.0,float(l.split()[2]),0.0])
                        cellvectors[2]=np.array([0.0,0.0,float(l.split()[3])])
                    if l.split()[0]=='A':
                        cellvectors[0]=np.array([float(l.split()[1]),float(l.split()[2]),float(l.split()[3])])
                    if l.split()[0]=='B':
                        cellvectors[1]=np.array([float(l.split()[1]),float(l.split()[2]),float(l.split()[3])])
                    if l.split()[0]=='C':
                        cellvectors[2]=np.array([float(l.split()[1]),float(l.split()[2]),float(l.split()[3])])
    #check cell volume 
    det=np.linalg.det(np.array(cellvectors))
    if np.abs(det)<10**(-3):
        ValueError("Cell Vectors do not span a unit cell!")
    return cellvectors

def read_periodicity(path="./",verbose=True):
    ## Function read out the cell periodicity from the .inp file
    ## input: (opt.)   path   path to the folder of the calculation         (string)
    ## output:  -                                                           (void)

    inp_file=get_inp_filename(path=path, verbose=False)
    if verbose:
        print(f"ℹ️ : Reading periodicity settings from file: {inp_file}")
    Periodic=True
    with open(inp_file) as f:
        lines = f.readlines()
        Cellflag=False
        for l in lines:
            if len(l.split())>=1:
                if l.split()[0]=='&CELL':
                    Cellflag=True
                if l.split()[0]=='END' and l.split()[1]=='CELL':
                    Cellflag=False
                if Cellflag:
                    if l.split()[0]=='PERIODIC':
                        if l.split()[1]=="NONE":
                             Periodic=False
                        else:
                             Periodic=True
    if verbose:
        if Periodic:
            print(f"ℹ️ : Periodicity detected")
        else:
            print(f"ℹ️ : No periodicity detected")

    return Periodic

####################################################################################
#########                End Cell-Vectors, Periodicity                  ############
####################################################################################

####################################################################################
#########                       System Configurations                   ############
####################################################################################

def check_uks(path="./",verbose=True):
    ##opens the .inp file in the directory and checks, if Multiplicity is 1
    ## input:
    ## (opt.)   folder              path to the folder of the .inp file         (string)
    ## output:  mul    (int)        the multiplicity of the system. 

    inp_file=get_inp_filename(path=path, verbose=False)
    if verbose:
        print(f"ℹ️ : Reading spin settings (RKS/UKS) from file: {inp_file}")
    UKS=False
    with open(inp_file,"r") as f:
        for line in f:
            if len(line.split())>=1:
                if line.split()[0]=="LSD":
                    UKS=True
    if verbose:
        if UKS:
            print(f"ℹ️ : Detected UKS calculation.")
        else:
            print(f"ℹ️ : Detected RKS calculation.")
    return UKS
def read_multiplicity(path="./",verbose=True):
    ##opens the .inp file in the directory and checks, if Multiplicity is 1
    ## input:
    ## (opt.)   folder              path to the folder of the .inp file         (string)
    ## output:  mul    (int)        the multiplicity of the system. 

    #get the Projectname
    inp_file=get_inp_filename(path=path, verbose=False)
    if verbose:
        print(f"ℹ️ : Reading spin multiplicity settings from file: {inp_file}")
    mul=1
    with open(inp_file,"r") as f:
        for line in f:
            if len(line.split())>1:
                if line.split()[0]=="MULTIPLICITY":
                    mul=int(line.split()[1])
    if verbose:
        print(f"ℹ️ : Found spin multiplicity to be: {int(mul)}")
    return mul

def get_number_of_electrons(parentfolder="./",verbose=True):
    filename=get_xyz_filename(path=parentfolder,verbose=False)
    Atoms=read_atomic_coordinates(filename)
    #get the Projectname
    inp_file=get_inp_filename(path=parentfolder, verbose=False)
    if verbose:
        print(f"ℹ️ : Reconstructing number of electrons from file: {inp_file}")
    #Calculate the HOMO 
    NumberOfElectrons={}
    Charge=0
    with open(inp_file,'r') as f:
        lines=f.readlines()
        for line in lines:
            if len(line.split())>=2:
                if line.split()[0]=="CHARGE":
                    Charge=int(line.split()[1])
                if line.split()[0]=="&KIND":
                    atomtype= line.split()[1]
                if line.split()[0]=="POTENTIAL":
                    PotentialName=line.split()[1]
                    splitedPotentialName=PotentialName.split('-')
                    if splitedPotentialName[0]=="GTH" or splitedPotentialName[1]=="GTH":
                        numstring=splitedPotentialName[-1]
                        numofE=int(numstring[1:])
                        NumberOfElectrons[atomtype]=numofE
                    else:
                        ValueError("Yet only GTH Potentials implemented")
    numofE=0
    for atom in Atoms:
        atomsymbol=atom[1]
        numofE+=NumberOfElectrons[atomsymbol]
    if verbose:
        print(f"ℹ️ : Number of (explicit) electrons in calculation: {numofE}")
        if Charge!=0:
            print(f"ℹ️ : System is charged with total charge: {Charge}")
        else:
            print(f"ℹ️ : System is charge-neutral")
    return numofE,Charge

def read_homo_index(parentfolder="./"):
    ##   Function to get the index of the HOMO orbital, if energyeigenvalues 0,1,2,...Homoit are ordered acendingly
    ##   input:   parentfolder:         (string)            absolute/relative path, where the geometry optimized .xyz file lies 
    ##                                                      in the subfolders there we find the electronic structure at displaced geometries                      
    ##   output:  HOMOit                (int)               the index of the HOMO orbital (python convention)
    numofE,Charge=get_number_of_electrons(parentfolder,verbose=False)
    numofE+=Charge
    remainder=numofE%2
    iter=np.floor(numofE/2)
    HOMOit=iter-1+remainder
    return int(HOMOit)
####################################################################################
#########                     END  System Configurations                ############
####################################################################################

####################################################################################
#########                     Ground state properties                   ############
####################################################################################
def read_total_energy(path="./", verbose=True):
    """
    Reads the total ground state energy from a CP2K output file.

    Args:
        path (str): Directory containing the .out file.
        verbose (bool): Whether to print informative messages.

    Returns:
        float: Ground state energy in Hartree.

    Raises:
        ValueError: If total energy could not be determined from the output file.
    """
    outfilename = get_out_filename(path=path, verbose=False)
    if verbose:
        print(f"ℹ️ : Reading total ground state energy from file: {outfilename}")

    GSEnergy = None  # Start with None to detect if it was found
    with open(os.path.join(path, outfilename), 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) > 8 and parts[0] == "ENERGY|" and parts[1] == "Total" and parts[2] == "FORCE_EVAL":
                try:
                    GSEnergy = float(parts[8])
                except ValueError:
                    raise ValueError(f"Could not convert energy value '{parts[8]}' to float in file '{outfilename}'")

    if GSEnergy is None:
        raise ValueError(f"Total ground state energy could not be determined from file '{outfilename}'")

    if verbose:
        print(f"ℹ️ : Found total ground state energy to be: {GSEnergy} [Ha]")

    return GSEnergy

def read_ground_state_dipole_moment(path="./",verbose=True):
    """
    Reads the dipole moment coordinates (X, Y, Z) from a text file and returns them as a NumPy array.

    Parameters:
        file_path (str): The path to the input file.

    Returns:
        np.ndarray: A NumPy array containing the dipole moment coordinates [X, Y, Z].
    """

    outfilename = get_out_filename(path=path, verbose=False)
    if verbose:
        print(f"ℹ️ : Reading ground state dipole moment from file: {outfilename}")
    dipole_moment = None
    readflag=False
    with open(outfilename, 'r') as file:
        for line in file:
            if len(line.split())>0:
                if readflag:
                    dx=line.split()[1]
                    dy=line.split()[3]
                    dz=line.split()[5]
                    dipole_moment=np.array([float(dx),float(dy),float(dz)])*ConversionFactors['A->a.u.']
                    break
                if line.split()[0]=="Dipole" and line.split()[1]=="moment" and line.split()[2]=="[Debye]":
                    readflag=True

    if dipole_moment is None:
        raise ValueError(f"ℹ️ :Dipole moment data not found in the file {outfilename}")
    if verbose:
        print(f"ℹ️ : Found ground state dipole moment to be: {dx,dy,dz} [e a_0]")
    return dipole_moment



####################################################################################
#########                     END Ground state properties               ############
####################################################################################


####################################################################################
#########                         AO Matrices                           ############
####################################################################################
def read_ks_matrices(parentfolder="./",filename='KSHamiltonian'):
    ## Reads the Kohn-Sham Hamiltonian (for spin-species alpha/-beta) and the overlapmatrix from a provided file
    ## input:
    ## (opt.)   filename            path to the Hamiltonian file        (string)
    ## output:  KSHamiltonian       Kohn-Sham Hamiltonian   symmetric np.array(NumBasisfunctions,Numbasisfunction)
    ##                              In case of Multiplicity 2 KSHamiltonian is the KS_Hamiltonian of the alpha spin!
    ##          OLM                 Overlapmatrix           symmetric np.array(NumBasisfunctions,Numbasisfunction)
    UKS=check_uks(parentfolder)
    if not UKS:
        try:
            KSlines,OLMlines,NumBasisfunctions=read_matrices_from_file_multiplicity_one(parentfolder,filename)
            KSHamiltonian=np.zeros((NumBasisfunctions,NumBasisfunctions))
            OLM=np.zeros((NumBasisfunctions,NumBasisfunctions))
            for l in KSlines:
                if len(l.split())<5:
                    jindices=[int(j)-1 for j in l.split()]
                else:
                    iindix=int(l.split()[0])-1
                    iter=4
                    for jindex in jindices:
                        KSHamiltonian[iindix][jindex]=float(l.split()[iter])
                        iter+=1
            for l in OLMlines:
                if len(l.split())<5:
                    jindices=[int(j)-1 for j in l.split()]
                else:
                    iindix=int(l.split()[0])-1
                    iter=4
                    for jindex in jindices:
                        OLM[iindix][jindex]=float(l.split()[iter])
                        iter+=1
            np.save(parentfolder+"/"+"KSHamiltonian",KSHamiltonian)
            np.save(parentfolder+"/"+"OLM",OLM)
            KSHamiltonian_alpha=KSHamiltonian
            KSHamiltonian_beta=KSHamiltonian_alpha
            os.remove(parentfolder+"/"+"KSHamiltonian")
        except:
            KSHamiltonian_alpha=np.load(parentfolder+"/KSHamiltonian.npy")
            KSHamiltonian_beta=KSHamiltonian_alpha
            OLM=np.load(parentfolder+"/OLM.npy")
    else:
        try:
            KSlines_alpha,KSlines_beta,OLMlines,NumBasisfunctions=read_matrices_from_file_multiplicity_two(parentfolder,filename)
            KSHamiltonian_alpha=np.zeros((NumBasisfunctions,NumBasisfunctions))
            KSHamiltonian_beta=np.zeros((NumBasisfunctions,NumBasisfunctions))
            OLM=np.zeros((NumBasisfunctions,NumBasisfunctions))
            for l in KSlines_alpha:
                if len(l.split())<5:
                    jindices=[int(j)-1 for j in l.split()]
                else:
                    iindix=int(l.split()[0])-1
                    iter=4
                    for jindex in jindices:
                        KSHamiltonian_alpha[iindix][jindex]=float(l.split()[iter])
                        iter+=1
            for l in KSlines_beta:
                if len(l.split())<5:
                    jindices=[int(j)-1 for j in l.split()]
                else:
                    iindix=int(l.split()[0])-1
                    iter=4
                    for jindex in jindices:
                        KSHamiltonian_beta[iindix][jindex]=float(l.split()[iter])
                        iter+=1
            for l in OLMlines:
                if len(l.split())<5:
                    jindices=[int(j)-1 for j in l.split()]
                else:
                    iindix=int(l.split()[0])-1
                    iter=4
                    for jindex in jindices:
                        OLM[iindix][jindex]=float(l.split()[iter])
                        iter+=1
            np.save(parentfolder+"/"+"KSHamiltonian_alpha",KSHamiltonian_alpha)
            np.save(parentfolder+"/"+"KSHamiltonian_beta",KSHamiltonian_beta)
            np.save(parentfolder+"/"+"OLM",OLM)
            KSHamiltonian=KSHamiltonian_alpha
            os.remove(parentfolder+"/"+"KSHamiltonian")
        except:
            KSHamiltonian_alpha=np.load(parentfolder+"/KSHamiltonian_alpha.npy")
            KSHamiltonian_beta=np.load(parentfolder+"/KSHamiltonian_beta.npy")
            OLM=np.load(parentfolder+"/OLM.npy")
            KSHamiltonian=KSHamiltonian_alpha
    return KSHamiltonian_alpha,KSHamiltonian_beta,OLM




def read_matrices_from_file_multiplicity_one(parentfolder="./",filename="KSHamiltonian"):
    ##Reads in the overlapmatrix and the KSHamiltonian in case of Spin multiplicity 1
    ## input:
    ## (opt.)   folder              path to the folder of the KSHamiltonian file         (string)
    ## output:  KSlines             list of strings of the converges KS Hamiltonian
    ##          OLMlines            list of strings of the overlapmatrix
    ##          NumBasisfunctions   int number of (spherical) Basisfunctions in Basis set
    with open(parentfolder+"/"+filename,'r+') as f:
        OLMlines=[]
        OLMFlag=False
        Niter=0
        Nlines=0
        for line in f:
            if len(line.split())>=1:
                if line.split()[0]=="KOHN-SHAM" and  line.split()[1]=="MATRIX":
                    OLMFlag=False
                    Niter+=1
                if line.split()[0]=="OVERLAP" and  line.split()[1]=="MATRIX" and OLMFlag:
                    OLMFlag=False
                if OLMFlag:
                    OLMlines.append(line)
                elif line.split()[0]=="OVERLAP" and  line.split()[1]=="MATRIX":
                    OLMFlag=True
            Nlines+=1
    NumBasisfunctions=0
    with open(parentfolder+"/"+filename,'r+') as f:
        KSlines=[]
        KSFlag=False
        Niter2=0
        Nlines2=0
        for line in f:
            if len(line.split())>=1:
                if line.split()[0]=="OVERLAP" and  line.split()[1]=="MATRIX":
                    KSFlag=False
                if line.split()[0]=="KOHN-SHAM" and  line.split()[1]=="MATRIX":
                    Niter2+=1
                    KSFlag=False
                if KSFlag:
                    KSlines.append(line)
                if line.split()[0]=="KOHN-SHAM" and  line.split()[1]=="MATRIX" and Niter2==Niter:
                    KSFlag=True
                if Nlines2==Nlines-3:
                    NumBasisfunctions=int(line.split()[0])
            Nlines2+=1
    return KSlines,OLMlines,NumBasisfunctions

def read_matrices_from_file_multiplicity_two(parentfolder,filename="KSHamiltonian"):
    with open(parentfolder+"/"+filename,'r+') as f:
        OLMlines=[]
        OLMFlag=False
        Niter0=0
        Niter1=0
        Nlines=0
        for line in f:
            if len(line.split())>=1:
                if line.split()[0]=="KOHN-SHAM" and  line.split()[1]=="MATRIX":
                    if line.split()[2]=="FOR" and  line.split()[3]=="ALPHA":
                        OLMFlag=False
                        Niter0+=1
                if line.split()[0]=="KOHN-SHAM" and  line.split()[1]=="MATRIX":
                    if line.split()[2]=="FOR" and  line.split()[3]=="BETA":
                        OLMFlag=False
                        Niter1+=1
                if line.split()[0]=="OVERLAP" and  line.split()[1]=="MATRIX" and OLMFlag:
                    OLMFlag=False
                if OLMFlag:
                    OLMlines.append(line)
                elif line.split()[0]=="OVERLAP" and  line.split()[1]=="MATRIX":
                    OLMFlag=True
            Nlines+=1
    NumBasisfunctions=0
    with open(parentfolder+"/"+filename,'r+') as f:
        KSlines_alpha=[]
        KSFlag1=False
        Niter2=0
        Nlines2=0
        for line in f:
            if len(line.split())>=1:
                if line.split()[0]=="OVERLAP" and  line.split()[1]=="MATRIX":
                    KSFlag1=False
                if line.split()[0]=="KOHN-SHAM" and  line.split()[1]=="MATRIX":
                    if line.split()[2]=="FOR" and  line.split()[3]=="BETA":
                        KSFlag1=False
                if line.split()[0]=="KOHN-SHAM" and  line.split()[1]=="MATRIX":
                    if line.split()[2]=="FOR" and  line.split()[3]=="ALPHA":
                        KSFlag1=False
                        Niter2+=1
                if KSFlag1:
                    KSlines_alpha.append(line)
                if line.split()[0]=="KOHN-SHAM" and  line.split()[1]=="MATRIX":
                    if line.split()[2]=="FOR" and  line.split()[3]=="ALPHA" and Niter2==Niter0:
                        KSFlag1=True
                if Nlines2==Nlines-3:
                    NumBasisfunctions=int(line.split()[0])
            Nlines2+=1
    with open(parentfolder+"/"+filename,'r+') as f:
        KSlines_beta=[]
        KSFlag1=False
        Niter2=0
        Nlines2=0
        for line in f:
            if len(line.split())>=1:
                if line.split()[0]=="OVERLAP" and  line.split()[1]=="MATRIX":
                    KSFlag1=False
                if line.split()[0]=="KOHN-SHAM" and  line.split()[1]=="MATRIX":
                    if line.split()[2]=="FOR" and  line.split()[3]=="ALPHA":
                        KSFlag1=False
                if line.split()[0]=="KOHN-SHAM" and  line.split()[1]=="MATRIX":
                    if line.split()[2]=="FOR" and  line.split()[3]=="BETA":
                        KSFlag1=False
                        Niter2+=1
                if KSFlag1:
                    KSlines_beta.append(line)
                if line.split()[0]=="KOHN-SHAM" and  line.split()[1]=="MATRIX":
                    if line.split()[2]=="FOR" and  line.split()[3]=="BETA" and Niter2==Niter1:
                        KSFlag1=True
            Nlines2+=1
    return KSlines_alpha,KSlines_beta,OLMlines,NumBasisfunctions


def read_mos_ao(parentfolder="./"):
    ## Reads the Molecular Orbitals from a provided file in the AO basis valid for Multiplicity 1
    ## input:
    ## (opt.)   filename            path to the MOs file        (string)
    ## output:  MOs                symmetric np.array(NumBasisfunctions,Numbasisfunction)       Expansion coefficients of the MOs in terms of AO's 
    ## Example: MOs[:,0] are the expansion coefficients of the MO 0 in the canonically ordered atomic Basis
    lastMOstart=0
    with open(parentfolder+"/MOs") as f:
        lineiter=0
        for line in f:
            if len(line.split())>6:
                if line.split()[0]=="MO|" and line.split()[1]=="EIGENVALUES," and line.split()[2]=="OCCUPATION" and line.split()[3]=="NUMBERS," and line.split()[4]=="AND" and line.split()[5]=="SPHERICAL" and line.split()[6]=="EIGENVECTORS":
                    lastMOstart=lineiter
            lineiter+=1
    MOstring=[]
    BasisFKTIndex=-10**(-20)
    with open(parentfolder+"/MOs") as f:
        lineiter=0
        for line in f:
            if lineiter>=lastMOstart:
                MOstring.append(line)
            if len(line.split())>5:
                if line.split()[1]=='E(Fermi):':
                    BasisFKTIndex=lineiter-2
            lineiter+=1
    BasisFKTIndex-=lastMOstart
    NUM_BASIS_FKT=int(MOstring[BasisFKTIndex].split()[1])
    MOs=np.zeros((NUM_BASIS_FKT,NUM_BASIS_FKT))
    HOMOid=read_homo_index(parentfolder)
    Basenumber=0
    MOindices=[]
    for line in MOstring:
        splited_line=line.split()[1:]
        if len(splited_line)>=5:
            if splited_line[0].isdigit() and splited_line[1].isdigit() and splited_line[2].isalpha():
                aoBasisindex=int(splited_line[0])-1
                iterator=0
                for number_string in splited_line[4:]:
                    number=float(number_string)
                    moindex=Basenumber+iterator
                    MOindices.append(moindex-HOMOid)
                    MOs[aoBasisindex,moindex]=number
                    iterator+=1
                if aoBasisindex==NUM_BASIS_FKT-1:
                    Basenumber+=4
    np.save("MOs",MOs)
    np.save("MOindices",MOindices)
    os.remove(parentfolder+"/MOs")
    return MOs,MOindices
def read_mos(parentfolder="./"):
    ## Reads the Molecular Orbitals from a provided file
    ## input:
    ## (opt.)   filename            path to the MOs file        (string)
    ## output:  MOs                symmetric np.array(NumBasisfunctions,Numbasisfunction)       Expansion coefficients of the MOs in terms of AO's 
    ## Example: MOs[:,0] are the expansion coefficients of the MO 0 in the canonically ordered atomic Basis
    try:
        MOs=np.load("MOs.npy")
        MOindices=np.load("MOindices.npy")
    except:
        #check if MO file exists
        MOfiles= [f for f in os.listdir(parentfolder) if f.endswith('MOs')]
        if len(MOfiles)==1:
            MOs,MOindices=read_mos_ao(parentfolder)
    return MOs,MOindices
####################################################################################
#########                        END AO Matrices                        ############
####################################################################################

####################################################################################
#########                         Excited States                        ############
####################################################################################
def read_excited_states(path,minweight=1e-6,verbose=True):
    ## Parses the excited states from a TDDFPT calculation done in CP2K  
    ## input:
    ## (opt.)   minweight           minimum amplitude to consider
    ## output: 
	#get the output file 
    def represents_int(s):
        try: 
            int(s)
        except ValueError:
            return False
        else:
            return True
    outfilename = get_out_filename(path=path, verbose=False)
    if verbose:
        print(f"ℹ️ : Reading TD-DFT excited states from file: {outfilename}")

    try:
        readflag=False
        energies=[]
        with open(outfilename, "r") as f:
            lines = f.readlines()
            for line in lines:
                if len(line.split())>4:
                    if line.split()[0]=="TDDFPT" and line.split()[1]==":" and line.split()[2]=="CheckSum" and line.split()[3]=="=":
                        readflag=False
                    if readflag:
                        energy=line.split()[2]
                        energies.append(float(energy))
                    if line.split()[0]=="number" and line.split()[1]=="energy" and line.split()[2]=="(eV)" and line.split()[3]=="x" and line.split()[4]=="y":
                        readflag=True
        states=[]
        stateiterator=[]
        stateiteratorflag=False
        homoid=read_homo_index(path)
        with open(outfilename,'r') as f:
            for line in f:
                if len(line.split())>0 and stateiteratorflag:
                    if line.split()[0]=="-------------------------------------------------------------------------------":
                        stateiteratorflag=False
                        states.append([energies[it],stateiterator])
                if stateiteratorflag and len(line.split())==3 and represents_int(line.split()[1]):
                    if abs(float(line.split()[2]))>=minweight:
                        stateiterator.append([int(line.split()[0])-homoid-1,int(line.split()[1])-homoid-1,float(line.split()[2])])
                if len(line.split())==3:
                    if represents_int(line.split()[0]) and not represents_int(line.split()[1]) and line.split()[2]=="eV":
                        if int(line.split()[0])<=len(energies):
                            it=int(line.split()[0])-1
                            if np.abs(energies[it]-float(line.split()[1]))<0.00001:
                                if stateiterator:
                                    states.append([energies[it-1],stateiterator])
                                    stateiterator=[]
                                stateiteratorflag=True                
        return states
    except Exception as e:
        raise IOError("Could not read file: " + outfilename + " (" + str(e) + ")")

def read_transition_dipole_moments(path="./",verbose=True):
    ## Parses the excited states & DipoleMoments from a TDDFPT calculation done in CP2K  
    ## input:
    ## (opt.)   minweight           minimum amplitude to consider
    ## output: 
	#get the output file 
    outfilename = get_out_filename(path=path, verbose=False)
    if verbose:
        print(f"ℹ️ : Reading TD-DFT excited states from file: {outfilename}")
    readflag=False
    energies=[]
    TransitionDipolevectors=[];Oscillatorstrenghts=[]
    with open(outfilename,'r') as f:
        for line in f:
            if len(line.split())>4:
                if line.split()[0]=="TDDFPT" and line.split()[1]==":" and line.split()[2]=="CheckSum" and line.split()[3]=="=":
                    readflag=False
                if readflag:
                    energy=line.split()[2]
                    energies.append(float(energy))
                    dx=line.split()[3]
                    dy=line.split()[4]
                    dz=line.split()[5]
                    TransitionDipolevectors.append(np.array([float(dx),float(dy),float(dz)]));Oscillatorstrenghts.append(float(line.split()[6]))
                if line.split()[0]=="number" and line.split()[1]=="energy" and line.split()[2]=="(eV)" and line.split()[3]=="x" and line.split()[4]=="y":
                    readflag=True
    np.save(path+"/"+"transition_dipole_vectors",TransitionDipolevectors);np.save(path+"/"+"oscillatorstrengths",Oscillatorstrenghts)
    np.save(path+"/"+"excited_state_energies",energies)

def read_G0W0_energies(path="./",verbose=True):
    ## Reads in the G0W0 energies and the respective Orbitals
    ## input: 
    ## (opt.)   parentfolder    path to the folder of the BasisHessian file         (string)
    ## output:  
    ##          BasisVectors    normalized displaced vectors                        (list of np.arrays)     
    ##          delta           displacementfactor                                  (float)
    ##          unit            unit of the displacementfactor                      (string, either 'Bohr' or 'sqrt(u)*Bohr') 
    outfilename = get_out_filename(path=path, verbose=False)
    if verbose:
        print(f"ℹ️ : Reading G0W0 excited state energies from file: {outfilename}")
    readflag=False
    G0W0flag=False
    orbitals=[]
    E_SCF=[]
    Sig_C=[]
    Sigxmvxc=[]
    E_QP=[]
    with open(outfilename) as f:
        lines=f.readlines()
        for line in lines:
            if len(line.split())>1:
                if line.split()[0]=="GW" and line.split()[1]=="HOMO-LUMO":
                    G0W0flag=False
                if readflag and G0W0flag:
                    orbitals.append(int(line.split()[0])-1)
                    E_SCF.append(float(line.split()[4]))
                    Sig_C.append(float(line.split()[5]))
                    Sigxmvxc.append(float(line.split()[6]))
                    E_QP.append(float(line.split()[7]))
                if line.split()[0]=="G0W0" and line.split()[1]=="results":
                    G0W0flag=True
                if line.split()[0]=="Molecular" and line.split()[1]=="orbital" and line.split()[2]=="E_SCF":
                    readflag=True
    return  orbitals,np.array(E_SCF),np.array(Sig_C),np.array(Sigxmvxc),np.array(E_QP)

####################################################################################
#########                       END  Excited States                     ############
####################################################################################
"####################################################################################"
"#########                       END CP2K- Read                          ############"
"####################################################################################"

"####################################################################################"
"#########                       WFN Data Format                          ###########"
"####################################################################################"
def read_cube_file(filename,parentfolder="./"):
    '''Function to write a .cube file , the origin is assumed to be [0.0,0.0,0.0]
       input:   filename:         (str)                   name of the file 
       (opt.)   parentfolder:     (str)                   path to the .inp file of the cp2k calculation to read in the cell dimensions                               
       output:  data              (np.array)              Nx x Ny x Nz numpy array where first index is x, second y and third is z
    '''
    predata=[]
    with open(parentfolder+"/"+filename,"r") as file:
        lines=file.readlines()
        it=0
        Cubedata=False
        for line in lines[2:]:
            if it==1:
                Nx=int(line.split()[0])
            if it==2:
                Ny=int(line.split()[0])
            if it==3:
                Nz=int(line.split()[0])
            if len(line.split())==6:
                Cubedata=True
            if Cubedata:
                for da in line.split():
                    predata.append(float(da))
            it+=1
    data=np.zeros((Nx,Ny,Nz))
    it=0
    for itx in range(Nx):
        for ity in range(Ny):
            for itz in range(Nz):
                data[itx][ity][itz]=predata[it]
                it+=1
    return data






    
