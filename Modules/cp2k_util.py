
from . import Read 
from .coordinate_tools import compute_center_of_geometry_coordinates,get_principle_axis_coordinates
import numpy as np
import os
import shutil
import subprocess
import ast
from typing import List, Tuple
num_of_e_atom = {
    "H": 1,
    "He": 2,
    "Li": 1,
    "Be": 3,
    "B": 5,
    "C": 4,
    "N": 5,
    "O": 6,
    "F": 7,
    "Ne": 8,
    "Na": 1,
    "Mg": 2,
    "Al": 3,
    "Si": 4,
    "P": 5,
    "S": 6,
    "Cl": 7,
    "Ar": 8
}
def get_cp2k_exec(cluster="local"):
    if cluster == "slurm":
        # trust module environment
        cp2k_exec = shutil.which("cp2k.psmp")
        if cp2k_exec is None:
            raise RuntimeError("cp2k.psmp not found in PATH, did you load the CP2K module in your jobscript?")
        return cp2k_exec
    else:  # local
        pathtocp2k = os.environ.get("cp2kpath")
        if pathtocp2k is None:
            raise RuntimeError("Environment variable cp2kpath not set for local execution")
        return os.path.join(pathtocp2k, "exe/local/cp2k.popt")
    
def write_pos_file(name, atoms, coordinates, cell, energy, forces, step, path):
    """
    Write a list of XYZ coordinates to a file with cell information.
    If the file exists, append the coordinates; otherwise, create it.
    
    Parameters:
    name (str): Name of the system
    atoms (list): List of atom symbols
    coordinates (list): List of (x, y, z) tuples/lists
    cell (numpy.ndarray): 3x3 numpy array representing the unit cell
    energy (float): Energy of the system
    forces (list): List of force vectors
    step (int): Optimization step number
    path (str): Path to the file
    """
    with open(path, "a") as f:
        f.write(f"Optimization Step: {step}\n")
        
        # Write cell information
        f.write(f"Cell\n")
        for i in range(3):
            f.write(f"{cell[i, 0]:.12f} {cell[i, 1]:.12f} {cell[i, 2]:.12f}\n")
        f.write(f"End Cell\n")
        
        # Write coordinates
        f.write(f"Coordinates\n")
        f.write(f"{len(atoms)}\n")
        f.write(f"{name}\n")
        for it, coord in enumerate(coordinates):
            f.write(f"{atoms[it]} {coord[0]:.12f} {coord[1]:.12f} {coord[2]:.12f}\n")
        f.write(f"End Coordinates\n")
        
        # Write energy
        f.write(f"Energy\n")
        f.write(f"E={energy}\n")
        f.write(f"End Energy\n")
        
        # Write forces
        f.write(f"Forces\n")
        for it, force in enumerate(forces):
            f.write(f"{atoms[it]} {force[0]:.12f} {force[1]:.12f} {force[2]:.12f}\n")
        f.write(f"End Forces\n")
        f.write("\n")  # Add blank line between steps for readability
def parse_last_optimization_step(path):
    """
    Parse the last optimization step from the file written by `write_pos_file`.
    Returns:
    dict with keys:
    - step (int)
    - atoms (list of str)
    - coordinates (list of [x,y,z])
    - cell (numpy.ndarray): 3x3 array representing the unit cell
    - energy (float)
    - forces (list of [fx,fy,fz])
    """
    import numpy as np
    
    last_block = []
    # Read all lines
    with open(path, "r") as f:
        lines = f.readlines()
    
    # Find the last "Optimization Step"
    for i, line in enumerate(lines):
        if line.startswith("Optimization Step:"):
            last_block = lines[i:] # from here to the end
    
    if not last_block:
        raise ValueError("No optimization step found in file.")
    
    # Parse step number
    step_line = last_block[0].strip()
    step = int(step_line.split(":")[1])
    
    atoms = []
    coordinates = []
    forces = []
    energy = None
    cell = None
    name = None
    
    i = 0
    while i < len(last_block):
        line = last_block[i].strip()
        
        if line == "Cell":
            # Parse the 3x3 cell matrix
            cell = []
            for j in range(3):
                parts = last_block[i+1+j].split()
                cell.append([float(parts[0]), float(parts[1]), float(parts[2])])
            cell = np.array(cell)
            i += 4  # Skip "Cell" + 3 lines + "End Cell" will be handled by the loop
            
        elif line == "Coordinates":
            n_atoms = int(last_block[i+1].strip())
            name = last_block[i+2].strip() # molecule/system name
            atoms = []
            coordinates = []
            for j in range(n_atoms):
                parts = last_block[i+3+j].split()
                atoms.append(parts[0])
                coordinates.append([float(parts[1]), float(parts[2]), float(parts[3])])
            i += 3 + n_atoms
            
        elif line == "Energy":
            energy_line = last_block[i+1].strip()
            energy = float(energy_line.split("=")[1])
            i += 2
            
        elif line == "Forces":
            atoms_forces = []
            forces = []
            j = i+1
            while j < len(last_block) and not last_block[j].startswith("End Forces"):
                parts = last_block[j].split()
                atoms_forces.append(parts[0])
                forces.append([float(parts[1]), float(parts[2]), float(parts[3])])
                j += 1
            i = j # move index to "End Forces"
            
        i += 1
    
    return {
        "name": name,
        "step": step,
        "atoms": atoms,
        "coordinates": coordinates,
        "cell": cell,
        "energy": energy,
        "forces": forces
    }

def geo_opt(
    cluster="local", 
    path="./", 
    tol_max=5e-4, 
    tol_drm=1e-4, 
    optimizer="bfgs_step"
):
    """
    Geometry optimization wrapper.

    Parameters
    ----------
    cluster : str
        Cluster type ("local", "slurm", etc.).
    path : str
        Base path containing input files.
    tol_max : float
        Maximum force convergence threshold.
    tol_drm : float
        RMS force convergence threshold.
    optimizer : str or callable
        Optimizer step function name or function. 
        Defaults to "bfgs_step", but can be swapped out.
    """

    # -------------------------------
    # 1. Input & setup
    # -------------------------------
    geo_opt_path = os.path.join(path, "Geo_Optimization")
    xyz_filepath = Read.get_xyz_filename(path, verbose=True)

    with open(xyz_filepath, "r") as f:
        input_data = f.readlines()

    # Parse CP2K config & execution setup
    config = parse_cp2k_config(input_data)


    cp2k_exec = get_cp2k_exec(cluster=cluster)

    # Resolve optimizer function if passed as string
    if isinstance(optimizer, str):
        optimizer_func = globals()[optimizer]
    else:
        optimizer_func = optimizer

    # -------------------------------
    # 2. Initialization / Restart
    # -------------------------------
    if not os.path.exists(geo_opt_path):
        os.makedirs(geo_opt_path, exist_ok=True)

        # Prepare CP2K input and run first force evaluation
        config=prepare_force_eval(
            path=geo_opt_path, config=config
        )
        energy, forces = run_force_eval(
            path=geo_opt_path,
            cp2k_exec=cp2k_exec,
            runner="slurm" if cluster == "slurm" else "local"
        )
        name=config["name"]
        atoms=config["atoms"]
        coordinates=config["coordinates"]
        cell=config["cell"]
        it = 0
        Hm1 =  np.eye(3 * len(atoms))  # Initial Hessian guess
        write_pos_file(
            name, atoms, coordinates,cell, energy, forces, 
            step=it, path=os.path.join(geo_opt_path, f"{name}.pos")
        )
        print(f"Starting fresh optimization for {name}\n", flush=True)

    else:
        # Continue from previous optimization
        results = parse_last_optimization_step(
            path=os.path.join(geo_opt_path, f"{name}.pos")
        )

        if results["name"] == name:
            it = results["step"]
            forces = results["forces"]
            energy = results["energy"]
            coordinates = results["coordinates"]
            #cell0=results["cell"]

            print(f"Continuing optimization from step {it}\n", flush=True)

            try:
                Hm1 = np.load(os.path.join(geo_opt_path, "Hm1.npy"))
                print("Loaded inverse Hessian from checkpoint\n", flush=True)
            except:
                Hm1 =  np.eye(3 * len(atoms))
                print("No Hessian found, using initial guess.\n", flush=True)

    # Print header for optimization progress
    print("-" * 52, flush=True)
    print(f"{'Iter':>4} | {'Energy':>10} | {'RMS Force':>8} | {'Max Force':>8}", flush=True)
    print("-" * 52, flush=True)

    # -------------------------------
    # 3. Optimization loop
    # -------------------------------
    def forces_summary(forces):
        """Return max and RMS force norms."""
        fmax = np.max(np.abs(forces))
        frms = np.sum(np.abs(forces)) / (3 * np.shape(coordinates)[0])
        return fmax, frms

    force_max, drm_force = forces_summary(forces)
    print(f"{it:4d} | {energy:10.8f} | {drm_force:8.6f} | {force_max:8.6f}", flush=True)

    while force_max > tol_max or drm_force > tol_drm:
        it += 1

        # Single optimizer step (BFGS or user-provided)
        coordinates, forces, energy, Hm1 = optimizer_func(
            path=geo_opt_path,
            name=name,
            atoms=atoms,
            coordinates=coordinates,
            forces=forces,
            energy=energy,
            Hm1=Hm1,
            cp2k_exec=cp2k_exec,
            runner="slurm" if cluster == "slurm" else "local",
        )

        # Save checkpoint (Hessian, positions)
        np.save(os.path.join(geo_opt_path, "Hm1.npy"), Hm1)
        write_pos_file(
            name, atoms, coordinates,cell, energy, forces, 
            step=it, path=os.path.join(geo_opt_path, f"{name}.pos")
        )

        # Update convergence metrics
        force_max, drm_force = forces_summary(forces)
        print(f"{it:4d} | {energy:10.8f} | {drm_force:8.6f} | {force_max:8.6f}", flush=True)

    print("-" * 52, flush=True)

    # -------------------------------
    # 4. Finalize optimized structure
    # -------------------------------
    shutil.copyfile(
        os.path.join(geo_opt_path, f"{name}_tmp.xyz"),
        os.path.join(geo_opt_path, f"{name}_opt.xyz")
    )
    
def bfgs_step(
    path,
    name,
    atoms,
    coordinates,
    forces,
    energy,
    Hm1,
    cp2k_exec="./cp2k.popt",
    runner="local",
    max_line_search_steps=5
):
    """
    Perform one BFGS optimization step with line search.
    """

    # ---------------------------------
    # 1. Flatten inputs for vector math
    # ---------------------------------
    forces_vec = np.array(forces).flatten()
    x0 = np.array(coordinates).flatten()

    # Sanity check: valid forces
    if np.any(np.isnan(forces_vec)) or np.any(np.isinf(forces_vec)):
        raise ValueError("Invalid forces encountered at starting point.")

    # ---------------------------------
    # 2. Compute search direction
    # ---------------------------------
    try:
        p = np.dot(Hm1, forces_vec)

        # Step length control (avoid huge displacements)
        max_step = 0.2  # Å
        p_reshaped = p.reshape(-1, 3)
        max_p_norm = np.max(np.linalg.norm(p_reshaped, axis=1))
        if max_p_norm > max_step:
            p *= (max_step / max_p_norm)

    except Exception as e:
        raise RuntimeError(f"Hessian multiplication failed: {e}")

    if np.linalg.norm(p) < 1e-10:
        raise RuntimeError("Search direction too small (likely converged or singular Hessian).")

    if np.linalg.norm(p) > 1.0:
        print("||p||=", np.linalg.norm(p))
        p = p / np.linalg.norm(p) * 1.0

    # ---------------------------------
    # 3. Line search (Armijo backtracking)
    # ---------------------------------
    alpha = 1.0
    c1 = 1e-4
    phi_0 = energy
    phi_prime_0 = -forces_vec @ p

    for ls_step in range(max_line_search_steps):
        # Candidate new position
        x_new = x0 + alpha * p
        n_atoms = len(coordinates)
        coordinates_new = x_new.reshape((n_atoms, 3)).tolist()

        # update the xyz file in the path 
        file_to_update = os.path.join(path, f"{name}_tmp.xyz")
        with open(file_to_update, "w") as f:
            f.write(str(len(atoms)) + "\n")
            f.write(str(ls_step) + "\n")
            for atom, coord in zip(atoms, coordinates_new):
                f.write(f"{atom:2s} {coord[0]:20.12f} {coord[1]:20.12f} {coord[2]:20.12f}\n")

        try:
            # Re-run CP2K evaluation at new coordinates
            energy_new, forces_new = run_force_eval(
                path=path, cp2k_exec=cp2k_exec, runner=runner
            )
            forces_new_vec = np.array(forces_new).flatten()

            # Reject if forces/energy are invalid
            if np.any(np.isnan([energy_new])) or np.any(np.isnan(forces_new_vec)):
                alpha *= 0.5
                continue

        except Exception as e:
            alpha *= 0.5
            if alpha < 1e-6:
                raise RuntimeError(f"Line search failed: {e}")
            continue

        # Armijo condition check
        phi = energy_new
        if phi <= phi_0 + c1 * alpha * phi_prime_0:
            break  # Accept step

        alpha *= 0.5
        if alpha < 1e-6:
            raise RuntimeError("Line search failed - step too small.")

    else:
        # <== This triggers if the for-loop finishes WITHOUT a `break`
        raise RuntimeError("Line search failed - no acceptable step found.")

    # ---------------------------------
    # 4. BFGS Hessian update
    # ---------------------------------
    s = alpha * p
    y = forces_vec - forces_new_vec
    Hm1_new = update_hessian_bfgs(Hm1, s, y, damping=True)

    # ---------------------------------
    # 5. Return updated state
    # ---------------------------------
    return coordinates_new, forces_new, energy_new, Hm1_new

def update_hessian_bfgs(H, s, y, damping=True):
    """
    Perform a (damped) BFGS inverse Hessian update.

    Parameters
    ----------
    H : ndarray
        Current inverse Hessian approximation.
    s : ndarray
        Step vector (x_{k+1} - x_k).
    y : ndarray
        Gradient difference (∇E_{k+1} - ∇E_k).
    damping : bool
        Whether to apply Powell's damping modification for stability.

    Returns
    -------
    H_new : ndarray
        Updated inverse Hessian.
    """
    y_dot_s = np.dot(y, s)
    sHs = np.dot(s, np.dot(H, s))

    # Apply Powell damping if curvature condition not met
    if damping and y_dot_s < 0.2 * sHs:
        theta = (0.8 * sHs) / (sHs - y_dot_s)
        y = theta * y + (1 - theta) * np.dot(H, s)
        y_dot_s = np.dot(y, s)

    rho = 1.0 / y_dot_s
    I = np.eye(len(H))
    V = I - rho * np.outer(s, y)

    return V @ H @ V.T + rho * np.outer(s, s)


def write_cell(write_path,name,cell):
    cell_path = os.path.join(write_path, f"{name}_tmp.cell")
    cell = np.array(cell, dtype=float)
    h_colwise = cell.T.flatten()
    vol = abs(np.linalg.det(cell))
    values = [1, 1] + h_colwise.tolist() + [vol]
    line = " ".join(f"{v:20.12f}" if isinstance(v, float) else str(v) for v in values)
    with open(cell_path,"w") as f:
        f.write(line)
def prepare_force_eval(path,config):
    name=config["name"]
    cell = config["cell"]
    atoms=config["atoms"]
    coordinates=config["coordinates"]
    cell_files= [f for f in os.listdir(path) if f.endswith('_tmp.cell')]
    
    if len(cell_files) == 0:
        if np.linalg.matrix_rank(cell)==0:
            padding = 10.0 # Define a clear padding value (e.g., 5 Å on each side)
            # 1. Center the molecule at the origin.
            centered_coords, _ = get_principle_axis_coordinates(coordinates)
            centered_coords = np.array(centered_coords)

            # 2. Calculate the full extent of the molecule in each dimension.
            #    This is 2 * the maximum absolute coordinate since it's centered.
            mol_extent_x = 2 * np.max(np.abs(centered_coords[:, 0]))
            mol_extent_y = 2 * np.max(np.abs(centered_coords[:, 1]))
            mol_extent_z = 2 * np.max(np.abs(centered_coords[:, 2]))
            max_extend=np.max([mol_extent_x,mol_extent_y,mol_extent_z])
            
            # 3. Define cell vectors as the molecular extent plus padding.
            cell = np.array([
                [max_extend + padding, 0, 0],
                [0, max_extend + padding, 0],
                [0, 0, max_extend + padding]
            ])

            write_cell(path, name, cell)

            # 4. Shift the centered coordinates to the center of the new box. This part is correct.
            cell_center = 0.5 * np.array([cell[0][0], cell[1][1], cell[2][2]])
            coordinates = centered_coords + cell_center
        elif np.linalg.matrix_rank(cell)==1:
            _, R = np.linalg.qr(cell)
            independent_index = np.where(np.abs(np.diag(R)) > 1e-10)
            independent_vectors=cell[independent_index[0],:]
            independent_vector=independent_vectors[0]
            print(independent_vector)
            cell_length_x=np.linalg.norm(independent_vector)
            e_x=independent_vector/cell_length_x
            # Create orthonormal basis with e_x as the first vector
            # Find two orthogonal vectors to complete the basis
            if abs(e_x[0]) < 0.9:
                e_y = np.array([1, 0, 0]) - np.dot([1, 0, 0], e_x) * e_x
            else:
                e_y = np.array([0, 1, 0]) - np.dot([0, 1, 0], e_x) * e_x
            e_y = e_y / np.linalg.norm(e_y)
            
            e_z = np.cross(e_x, e_y)
            e_z = e_z / np.linalg.norm(e_z)
            # Rotation matrix to align independent vector with x-axis
            rotation_matrix = np.column_stack([e_x, e_y, e_z]).T
            # Rotate coordinates to align with new coordinate system
            rotated_coords = np.dot(np.array(coordinates), rotation_matrix.T)

            # Center the molecule along the periodic direction (x)
            x_center = np.mean(rotated_coords[:, 0])
            rotated_coords[:, 0] -= x_center-cell_length_x/2
            
            # Get principal axis coordinates for y and z directions (non-periodic)
            temp_coords_yz = rotated_coords[:, 1:]  # y and z coordinates
            centered_coords_yz, _ = get_principle_axis_coordinates(temp_coords_yz)
            centered_coords_yz = np.array(centered_coords_yz)
            
            # Combine x (periodic, centered) with y,z (principal axes, centered)
            final_coords = np.column_stack([
                rotated_coords[:, 0],  # x: already centered for periodicity
                centered_coords_yz    # y,z: principal axis centered
            ])
            
            # Calculate extents in non-periodic directions
            mol_extent_y = 2 * np.max(np.abs(centered_coords_yz[:, 0]))
            mol_extent_z = 2 * np.max(np.abs(centered_coords_yz[:, 1]))
            max_extent_yz = np.max([mol_extent_y, mol_extent_z])
            
            # Define padding
            padding = 10.0
            
            # Create new cell: periodic in x, large box in y and z
            new_cell = np.array([
                [cell_length_x, 0, 0],  # Keep original periodic length
                [0, max_extent_yz + padding, 0],  # Large box for y
                [0, 0, max_extent_yz + padding]   # Large box for z
            ])
            
            # Transform back to original coordinate system
            final_coords_original = np.dot(final_coords, rotation_matrix)
            
            # Shift to center of the new box (only for non-periodic directions)
            cell_center_y = 0.5 * new_cell[1, 1]
            cell_center_z = 0.5 * new_cell[2, 2]
            
            # Apply shifts in the rotated coordinate system
            shift_vector = np.array([0, cell_center_y, cell_center_z])
            shift_vector_original = np.dot(shift_vector, rotation_matrix)
            
            coordinates = final_coords_original + shift_vector_original
            
            # Transform cell back to original coordinate system
            cell = np.dot(rotation_matrix.T, np.dot(new_cell, rotation_matrix))
            
            write_cell(path, name, cell)
            
            
            #reorient coordinate system s.t. x direction points to independent vector

    elif len(cell) == 0 and len(cell_files) == 1:
        # This assumes the cell file is already in the correct path, so no action needed.
        pass
    else:
        # Use the user-provided cell.
        write_cell(path, name, cell)

    # Write the (now correctly centered) coordinates to the XYZ file.
    xyz_path = os.path.join(path, f"{name}_tmp.xyz")
    with open(xyz_path, "w") as f:
        f.write(f"{len(atoms)}\n")
        f.write("Generated by force_eval\n")
        for atom, coord in zip(atoms, coordinates):
            f.write(f"{atom:2s} {coord[0]:20.12f} {coord[1]:20.12f} {coord[2]:20.12f}\n")

    

    generate_input(
        config=config,
        write_path=path,
        accuracy="fine"
    )
    #update cell and coordinates in config
    config["coordinates"]=coordinates
    config["cell"]=cell
    return config
def run_force_eval(
    path: str,
    cp2k_exec: str = "./cp2k.popt",
    input_file: str = "input_file.inp",
    output_file: str = "output_file.out",
    runner: str = "local"
) -> Tuple[float, List[List[float]]]:
    """
    Runs a CP2K calculation for given coordinates and atoms.

    If config_string is provided:
        1. Updates the XYZ file in `path`
        2. Generates the CP2K input file from config_string
    Otherwise:
        Assumes input_file.inp already exists in path.

    Then:
        3. Runs CP2K
        4. Reads forces and energy

    Parameters
    ----------
    path : str
        Working directory.
    name : str
        Base name for XYZ file.
    coordinates : list of list of float
        Atomic coordinates.
    atoms : list of str
        Atom symbols.
    config_string : str or None
        CP2K config string (optional).
    cp2k_exec : str
        Path to CP2K executable.
    input_file : str
        CP2K input file name.
    output_file : str
        CP2K output file name.
    np : int
        Number of MPI processes.
    mpirun_cmd : str
        MPI command.

    Returns
    -------
    E0 : float
        Ground state energy.
    forces : list of list of float
        Atomic forces.
    """
    # Step 3: Build command
    if runner == "local":
        cmd = ["mpirun", "-np", str(4), cp2k_exec, "-i", input_file, "-o", output_file]
    elif runner == "slurm":
        # In SLURM allocation, srun handles MPI ranks automatically
        cmd = ["srun", cp2k_exec, "-i", input_file, "-o", output_file]
    else:
        raise ValueError("runner must be 'local' or 'slurm'")
    # Step 3: Run CP2K
    output_path = os.path.join(path, output_file)
    with open(output_path, "w") as out_f:
        process = subprocess.Popen(
            cmd,
            cwd=path,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        # --- List of phrases to ignore ---
        ignore_phrases = [
            "Ignoring PCI device with non-16bit domain",
            "Pass --enable-32bits-pci-domain to configure to support such devices",
            "(warning: it would break the library ABI, don't enable unless really needed)",
            "SIRIUS 7.6.1, git hash:"
        ]

        for line in process.stdout:
            # --- Check if any of the ignored phrases are in the current line ---
            if any(phrase in line for phrase in ignore_phrases):
                continue
                
            print(line, end="")
            out_f.write(line)

        retcode = process.wait()
        if retcode != 0:
            raise RuntimeError(f"CP2K failed with return code {retcode}")

    # Step 4: Extract results
    forces = Read.read_forces(folder=path)
    E0 = Read.read_total_energy(path=path,verbose=False)
    return E0, forces
    
def parse_cp2k_config(config) -> dict:
    """
    Parses a CP2K-style configuration string.

    Example:
    "acetone;geo@CAM-B3LYP;DZVP/HFX_BASIS;GTH/GTH_POTENTIALS;cell=(22, 22, 22)"
    or:
    "acetone;geo@CAM-B3LYP;DZVP/HFX_BASIS;GTH/GTH_POTENTIALS;cell=((1,0,0)(0,1,0)(0,0,1))"
    """
    result = {
        "name": None,
        "run_mode": None,
        "xc_functional": None,
        "basis": None,
        "basis_file": None,
        "potential": None,
        "potential_file": None,
        "charge": 0,
        "multiplicity": 1,
        "UKS": False,
        "cell":np.zeros((3,3)),
        "atoms":None,
        "coordinates":None
    }
    config_string=config[1][1:]

    parts = config_string.strip().split(';')
    if len(parts) < 4:
        raise ValueError("Expected at least 4 parts: name; run_mode@xc; basis/file; potential/file")

    # 1. name
    result["name"] = parts[0].strip()

    # 2. run_mode@xc_functional
    if '@' not in parts[1]:
        raise ValueError("Second part must contain '@' separating run_mode and xc_functional")
    run_mode, xc_functional = parts[1].split('@')
    result["run_mode"] = run_mode.strip()
    if xc_functional.strip()[0] == "U":
        result["UKS"] = True
        result["xc_functional"] = xc_functional.strip()[1:]
    else:
        result["UKS"] = False
        result["xc_functional"] = xc_functional.strip()

    # 3. basis/file
    if '/' not in parts[2]:
        raise ValueError("Third part must be in the form basis/file")
    basis, basis_file = parts[2].split('/')
    result["basis"] = basis.strip()
    result["basis_file"] = basis_file.strip()

    # 4. potential/file
    if '/' not in parts[3]:
        raise ValueError("Fourth part must be in the form potential/file")
    potential, potential_file = parts[3].split('/')
    result["potential"] = potential.strip()
    result["potential_file"] = potential_file.strip()

    # 5. optional settings
    for field in parts[4:]:
        if field.startswith("charge="):
            result["charge"] = int(field.split("=")[1])
        elif field.startswith("m="):
            result["multiplicity"] = int(field.split("=")[1])
        elif field.startswith("cell="):
            cell_data = field[len("cell="):].strip()
            cell_data =cell_data.split(";")[0]
            cell = np.array(ast.literal_eval(cell_data), dtype=float)
            result["cell"] = cell
    coordinates=[]
    atoms=[]
    for line in config[2:]:
        atoms.append(line.split()[0])
        coordinates.append(np.array([float(line.split()[1]),float(line.split()[2]),float(line.split()[3])]))
    result["atoms"]=atoms
    result["coordinates"]=coordinates
    return result
def generate_input(
    config,
    write_path="./",
    accuracy="fine"
):
    """
    Generate a CP2K input file using given coordinates, atoms, and a CP2K config string.
    """
    name=config["name"]
    atoms=config["atoms"]
    # 2. Start building input
    string = global_section(molecule_name=name, run_mode=config["run_mode"])
    string += dft_basis_set_section(basis_set_file=config["basis_file"], potential_file=config['potential_file'])
    string += dft_cutoff_rel_cutoff(accuracy=accuracy)

    hybrid = config["xc_functional"].split("_")[0] in ["CAM-B3LYP", "LCwPBE", "B3LYP", "PBEO"]
    string += qs_section(method="GPW", accuracy=accuracy, extrapolation="ASPC", hybrid=hybrid)

    # Charge / multiplicity logic
    if config["UKS"]:
        string += dft_charge_multiplicity(charge=config["charge"], multiplicity=config["multiplicity"], LSD=True)
    else:
        num_of_electrons = sum(num_of_e_atom[atom] for atom in atoms)
        if (num_of_electrons - int(config["charge"])) % 2 != 0:
            config["multiplicity"] = 2
            string += dft_charge_multiplicity(charge=config["charge"], multiplicity=config["multiplicity"], LSD=True)
        else:
            string += dft_charge_multiplicity(charge=config["charge"], multiplicity=config["multiplicity"], LSD=False)
    cell=config["cell"]
    # Cell / periodicity
    if np.sum(np.abs(cell)**2)<1e-10:
        string += poisson_section(periodic="NONE", psolver="WAVELET")
    else:
        string += poisson_section(periodic="XYZ", psolver="PERIODIC")

    if config["run_mode"] == "energy":
        string += print_AO_section(filename="KSHamiltonian")

    string += scf_section(method="OT")

    string += xc_section(
        xc_functional=config["xc_functional"],
        density_cutoff=1e-10,
        gradient_cutoff=1e-10,
        tau_cutoff=1e-10,
        eps_schwarz=1e-9,
        max_memory=30000,
        eps_storage_scaling=1e-1
    )

    periodic = "XYZ"
    cell=config["cell"]
    if np.linalg.matrix_rank(cell)==0:
        periodic = "NONE"
    xc_func = config["xc_functional"].split("_")[0]
    if xc_func in ["CAM-B3LYP", "B3LYP"]:
        func = "BLYP"
    elif xc_func in ["LCwPBE", "PBE", "PBE0"]:
        func = "PBE"
    elif xc_func in ["LDA"]:
        func = "LDA"
    else:
        raise ValueError(f"Unknown XC functional: {config['xc_functional']}")

    kinds = [{
        "element": atom,
        "basis_set": config["basis"],
        "potential": f"{config['potential']}-{func}-q{num_of_e_atom[atom]}"
    } for atom in sorted(set(atoms))]

    # Always write coordinates from given list
    string += subsys_section(
        cell_file_name=name+"_tmp.cell",
        cell_file_format="CP2K",
        periodic=periodic,
        coord_file_name=name+"_tmp.xyz",
        coord_file_format="XYZ",
        kinds=kinds
    )

    if config["run_mode"] == "geo" or config["run_mode"]=="force":
        string += force_stress_eval_print_section(
            forces_on=True,
            forces_filename="=Forces",
            ndigits=15
        )
    elif config["run_mode"]=="cell":
        string += force_stress_eval_print_section(
            forces_on=True,
            stresses_on=True,
            forces_filename="=Forces",
            stresses_filename="=Stress_Tensor",
            ndigits=15
        )

    string += "&END FORCE_EVAL\n"

    # 3. Write input file
    with open(os.path.join(write_path, "input_file.inp"), "w") as f:
        f.write(string)
def global_section(molecule_name, run_mode="energy"):
    """
    Create the GLOBAL section of a CP2K input file.

    Parameters:
        molecule_name (str): Name of the molecule or project.
        run_mode (str): Type of run - 'energy', 'va', 'force', or 'geo'.
        iolevel (str): IOLEVEL setting in CP2K.
        print_level (str, optional): Optional PRINT_LEVEL setting.

    Returns:
        str: A formatted GLOBAL section.
    """
    run_mode = run_mode.lower()

    if run_mode == "energy":
        RUN_TYPE = "ENERGY"
    elif run_mode in ["va", "force"]:
        RUN_TYPE = "ENERGY_FORCE"
    elif run_mode == "geo":
        RUN_TYPE = "ENERGY_FORCE"
    else:
        raise ValueError(f"Unsupported run_mode: '{run_mode}'. Choose from 'energy', 'va', 'force', or 'geo'.")

    string  = "&GLOBAL\n"
    string += f"   PROJECT {molecule_name}_{run_mode}\n"
    string += f"   RUN_TYPE {RUN_TYPE}\n"
    string += "   IOLEVEL  MEDIUM\n"
    string += "&END GLOBAL\n"
    string +="\n"

    return string
def qs_section(
    method="GPW",
    accuracy="fine",
    extrapolation="ASPC",
    hybrid=False
):
    """
    Generate the &QS section for CP2K input.

    Parameters:
        method (str): Method used for electronic structure (e.g., GPW, GAPW).
        eps_default (float): Global numerical threshold for accuracy.
        eps_pgf_orb (float): Precision for Gaussian orbitals.
        eps_filter_matrix (float): Threshold for filtering matrices.
        extrapolation (str): Extrapolation scheme for MD or SCF.

    Returns:
        str: A formatted &QS section.
    """
    eps_default=1e-12
    if accuracy=="coarse":
        eps_default=1e-10
    elif accuracy=="ultra":
        eps_default=1e-14
        
    section  = "&QS\n"
    section += f"  METHOD {method.upper()}\n"
    section += f"  EPS_DEFAULT {eps_default:.1E}\n"
    section += f"  EXTRAPOLATION {extrapolation.upper()}\n"
    if hybrid:
        section += f"MIN_PAIR_LIST_RADIUS -1\n"
    section += "&END\n"
    return section
def geo_opt_section(accuracy="fine",
    optimizer="BFGS"
):
    """
    Generate the &MOTION / &GEO_OPT section for geometry optimization in CP2K.

    Parameters:
        max_force (float): Maximum force convergence criterion.
        max_iter (int): Maximum number of geometry optimization steps.
        optimizer (str): Optimization algorithm (e.g., BFGS, CG, LBFGS).
        keep_space_group (bool): Whether to preserve the space group.
        spgr_print_atoms (bool): Whether to print equivalent atoms.

    Returns:
        str: A formatted &GEO_OPT section.
    """
    if accuracy=="ultra":
        max_force=5.0e-6
        max_iter=600
        trust_radius=0.05
    elif accuracy=="fine":
        max_force=1.0e-4
        max_iter=300
        trust_radius=0.05
    elif accuracy=="coarse":
        max_force=1.0E-3
        max_iter=200
        trust_radius=0.15
    else:
        raise ValueError(f"Unknown accuracy level: {accuracy}")
    
    string  = "&MOTION\n"
    string += "  &GEO_OPT\n"
    string += "    TYPE MINIMIZATION\n"
    string += f"    MAX_FORCE {max_force:.7E}\n"
    string += f"    MAX_ITER {max_iter}\n"
    string += f"    OPTIMIZER {optimizer.upper()}\n"
    string +=  "    &BFGS\n"
    string += f"        TRUST_RADIUS {trust_radius}\n"
    string +=  "    &END BFGS\n"
    string += "  &END GEO_OPT\n"
    string += "&END MOTION\n"
    return string
def dft_basis_set_section(basis_set_file="HFX_BASIS", potential_file="GTH_POTENTIALS",stress_tensor="NONE"):
    """
    Create the DFT section of a CP2K input file.

    Parameters:
        charge (int): Total charge of the system.
        multiplicity (int): Spin multiplicity.
        cutoff (float): Plane wave cutoff energy.
        rel_cutoff (float): Relative cutoff energy.
        ngrids (int): Number of grids.
        basis_set_file (str): Path or name of the basis set file.
        potential_file (str): Path or name of the pseudopotential file.

    Returns:
        str: A formatted DFT section.
    """
    string="&FORCE_EVAL\n"
    string+= "METHOD Quickstep\n"
    string+=f"STRESS_TENSOR {stress_tensor}"
    string+="\n"
    string += "&DFT\n"
    string += "   ! basis sets and pseudopotential files can be found in cp2k/data\n"
    string += f"   BASIS_SET_FILE_NAME {basis_set_file}\n"
    string += f"   POTENTIAL_FILE_NAME {potential_file}\n"
    string += "\n"
    return string
def dft_charge_multiplicity(charge=0, multiplicity=1,LSD=False):
    """
    Create the DFT section of a CP2K input file.

    Parameters:
        charge (int): Total charge of the system.
        multiplicity (int): Spin multiplicity.
        cutoff (float): Plane wave cutoff energy (Ry).
        rel_cutoff (float): Relative cutoff energy (Ry).
        ngrids (int): Number of grids.
        basis_set_file (str): Path or name of the basis set file.
        potential_file (str): Path or name of the pseudopotential file.

    Returns:
        str: A formatted DFT section.
    """
    string = "   ! Charge and multiplicity\n"
    string += f"   CHARGE {charge}\n"
    string += f"   MULTIPLICITY {multiplicity}\n"
    if LSD:
        string += "   LSD\n"
    string +="\n"
    return string
def dft_cutoff_rel_cutoff(accuracy="fine"):
    """
    Create the DFT section of a CP2K input file.

    Parameters:
        charge (int): Total charge of the system.
        multiplicity (int): Spin multiplicity.
        cutoff (float): Plane wave cutoff energy (Ry).
        rel_cutoff (float): Relative cutoff energy (Ry).
        ngrids (int): Number of grids.
        basis_set_file (str): Path or name of the basis set file.
        potential_file (str): Path or name of the pseudopotential file.

    Returns:
        str: A formatted DFT section.
    """
    if accuracy=="fine":
        cutoff=450; rel_cutoff=70; ngrids=5
    elif accuracy=="ultra":
        cutoff=600; rel_cutoff=70; ngrids=6
    elif accuracy=="coarse":
        cutoff=300; rel_cutoff=50; ngrids=4
    string = "   &MGRID\n"
    string += "      ! PW cutoff ... depends on the element (basis)\n"
    string += f"      CUTOFF {cutoff}\n"
    string += f"      REL_CUTOFF {rel_cutoff}\n"
    string += f"      NGRIDS {ngrids}\n"
    string += "   &END MGRID\n"
    string +="\n"
    return string
def print_AO_section(filename="KSHamiltonian"):
    """
    Create the PRINT section of a CP2K input file for SCF output.

    Parameters:.
        filename (str): Base filename for AO_MATRICES output.

    Returns:
        str: A formatted PRINT section.
    """
    string  = "&PRINT\n"
    string += f"   &AO_MATRICES \n"
    string += "      ADD_LAST SYMBOLIC\n"
    string += "      KOHN_SHAM_MATRIX .TRUE.\n"
    string += "      OVERLAP .TRUE.\n"
    string += f"      FILENAME ={filename}\n"
    string += "      NDIGITS 15\n"
    string += "   &END AO_MATRICES\n\n"

    string += "&END PRINT\n"
    string +="\n"
    return string
def xc_section(
    xc_functional="CAM-B3LYP_omega=0.33",
    density_cutoff=1e-12,
    gradient_cutoff=1e-12,
    tau_cutoff=1e-12,
    eps_schwarz=1e-12,
    max_memory=30000,
    eps_storage_scaling=1e-1
):
    """
    Create the &XC section of a CP2K input file with support for common hybrid and GGA functionals.

    Supported functionals:
        - "PBE"
        - "B3LYP"
        - "PBE0"
        - "CAM-B3LYP_omega=<value>"
        - "LC-WPBE_omega=<value>"

    Returns:
        str: Formatted &XC section string.
    """

    string = "&XC\n"
    string += f"   DENSITY_CUTOFF {density_cutoff:.16E}\n"
    string += f"   GRADIENT_CUTOFF {gradient_cutoff:.16E}\n"
    string += f"   TAU_CUTOFF {tau_cutoff:.16E}\n"

    if "CAM-B3LYP" in xc_functional:
        try:
            omega = float(xc_functional.split("omega=")[1])
        except:
            omega = 0.33

        string += "   &XC_FUNCTIONAL\n"
        string += "      &HYB_GGA_XC_CAM_B3LYP\n"
        string += f"         _OMEGA {omega}\n"
        string += "      &END\n"
        string += "   &END XC_FUNCTIONAL\n"

        string += "   &HF\n"
        string += "      &MEMORY\n"
        string += f"         EPS_STORAGE_SCALING {eps_storage_scaling:.1E}\n"
        string += f"         MAX_MEMORY {max_memory:.0f}\n"
        string += "      &END MEMORY\n"
        string += "      &SCREENING\n"
        string += f"         EPS_SCHWARZ {eps_schwarz:.16E}\n"
        string += "      &END SCREENING\n"
        string += "      &INTERACTION_POTENTIAL\n"
        string += "         POTENTIAL_TYPE MIX_CL\n"
        string += f"         OMEGA {omega}\n"
        string += f"         SCALE_LONGRANGE 0.46\n"
        string += f"         SCALE_COULOMB 0.19\n"
        string += "      &END INTERACTION_POTENTIAL\n"
        string += "   &END HF\n"

    elif "LCwPBE" in xc_functional:
        try:
            omega = float(xc_functional.split("omega=")[1])
        except:
            omega = 0.4

        string += "   &XC_FUNCTIONAL\n"
        string += "      &HYB_GGA_XC_LC_WPBEH_WHS\n"
        string += f"         _OMEGA {omega}\n"
        string += "      &END HYB_GGA_XC_LC_WPBEH_WHS\n"
        string += "   &END XC_FUNCTIONAL\n"

        string += "   &HF\n"
        string += "      &SCREENING\n"
        string += f"         EPS_SCHWARZ {eps_schwarz:.1E}\n"
        string += "      &END SCREENING\n"
        string += "      &MEMORY\n"
        string += f"         EPS_STORAGE_SCALING {eps_storage_scaling:.1E}\n"
        string += f"         MAX_MEMORY {max_memory:.0f}\n"
        string += "      &END MEMORY\n"
        string += "      &INTERACTION_POTENTIAL\n"
        string += "         POTENTIAL_TYPE LONGRANGE\n"
        string += f"         OMEGA {omega}\n"
        string += "      &END INTERACTION_POTENTIAL\n"
        string += "   &END HF\n"

    elif xc_functional.upper() == "PBE":
        string += "   &XC_FUNCTIONAL\n"
        string += "      &PBE\n"
        string += "      &END\n"
        string += "   &END XC_FUNCTIONAL\n"

    elif xc_functional.upper() == "BLYP":
        string += "   &XC_FUNCTIONAL\n"
        string += "      &BLYP\n"
        string += "      &END\n"
        string += "   &END XC_FUNCTIONAL\n" 

    elif xc_functional.upper() == "LDA":
        string += "   &XC_FUNCTIONAL\n"
        string += "      &PADE\n"
        string += "      &END\n"
        string += "   &END XC_FUNCTIONAL\n"  

    elif xc_functional.upper() == "B3LYP":
        string += "   &XC_FUNCTIONAL\n"
        string += "      &GGA_C_LYP\n"
        string += "         SCALE 0.81\n"
        string += "      &END\n"
        string += "      &GGA_X_B88\n"
        string += "         SCALE 0.72\n"
        string += "      &END\n"
        string += "      &VWN\n"
        string += "         FUNCTIONAL_TYPE VWN5\n"
        string += "         SCALE_C 0.19\n"
        string += "      &END\n"
        string += "      &XALPHA\n"
        string += "         SCALE_X 0.08\n"
        string += "      &END\n"
        string += "   &END XC_FUNCTIONAL\n"

        string += "   &HF\n"
        string += "      FRACTION 0.2\n"
        string += "      &MEMORY\n"
        string += f"         EPS_STORAGE_SCALING {eps_storage_scaling:.1E}\n"
        string += f"         MAX_MEMORY {max_memory:.0f}\n"
        string += "      &END MEMORY\n"
        string += "      &INTERACTION_POTENTIAL\n"
        string += "         POTENTIAL_TYPE COULOMB\n"
        string += "      &END INTERACTION_POTENTIAL\n"
        string += "      &SCREENING\n"
        string += f"         EPS_SCHWARZ {eps_schwarz:.16E}\n"
        string += "      &END SCREENING\n"
        string += "   &END HF\n"

    elif xc_functional.upper() == "PBE0":
        string += "   &XC_FUNCTIONAL\n"
        string += "      &PBE\n"
        string += "         SCALE_X 0.75\n"
        string += "         SCALE_C 1.0\n"
        string += "      &END\n"
        string += "   &END XC_FUNCTIONAL\n"

        string += "   &HF\n"
        string += "      FRACTION 0.25\n"
        string += "      &MEMORY\n"
        string += f"         EPS_STORAGE_SCALING {eps_storage_scaling:.1E}\n"
        string += f"         MAX_MEMORY {max_memory:.0f}\n"
        string += "      &END MEMORY\n"
        string += "      &SCREENING\n"
        string += f"         EPS_SCHWARZ {eps_schwarz:.1E}\n"
        string += "      &END SCREENING\n"
        string += "      &INTERACTION_POTENTIAL\n"
        string += "         POTENTIAL_TYPE COULOMB\n"
        string += "      &END\n"
        string += "   &END HF\n"

    else:
        raise ValueError(f"Unsupported XC functional: {xc_functional}")
    string+= "&END XC\n"
    string+="&END DFT"
    string +="\n"
    return string
def scf_section(
    scf_guess="RESTART",
    max_scf_standard=300,
    max_scf_ot=20,
    max_scf_outer=20,
    eps_scf_ot=1.0e-6,
    eps_scf_standard=1.0e-7,
    method="STANDARD",  # OT or STANDARD
    # OT-specific
    preconditioner="FULL_ALL",
    minimizer="CG",
    # STANDARD-specific
    mixing_method="BROYDEN_MIXING",
    mixing_alpha=0.4,
    mixing_nbuffer=8,
    diagonalization_algorithm="STANDARD",
    restart_print=True,
):
    """
    Generate the &SCF section of a CP2K input file with OT or STANDARD method.

    Returns:
        str: CP2K-formatted SCF section.
    """

    string = "&SCF\n"

    if method.upper() == "OT":
        string += f"   SCF_GUESS {scf_guess}\n"
        string += f"   MAX_SCF {max_scf_ot}\n"
        string += f"   EPS_SCF {eps_scf_ot:.1E}\n"
        string += "   &OT\n"
        string += f"      PRECONDITIONER {preconditioner}\n"
        string += f"      MINIMIZER {minimizer}\n"
        string += "   &END OT\n"
        string += "   &OUTER_SCF \n"
        string += f"   MAX_SCF {max_scf_outer} \n"
        string += f"   EPS_SCF {eps_scf_ot} \n"
        string += "   &END OUTER_SCF\n"

    elif method.upper() == "STANDARD":
        string += f"   SCF_GUESS {scf_guess}\n"
        string += f"   MAX_SCF {max_scf_standard}\n"
        string += f"   EPS_SCF {eps_scf_standard:.1E}\n"
        string += "   &DIAGONALIZATION\n"
        string += f"      ALGORITHM {diagonalization_algorithm}\n"
        string += "   &END DIAGONALIZATION\n"
        string += "   &MIXING\n"
        string += f"      METHOD {mixing_method}\n"
        string += f"      ALPHA {mixing_alpha:.2f}\n"
        string += f"      NBROYDEN {mixing_nbuffer}\n"
        string += "   &END MIXING\n"

    else:
        raise ValueError(f"Unsupported SCF method '{method}'. Choose 'OT' or 'STANDARD'.")

    if restart_print:
        string += "   &PRINT\n"
        string += "      &RESTART ON\n"
        string += "      &END RESTART\n"
        string += "   &END PRINT\n"

    string += "&END SCF\n"
    return string
def poisson_section(periodic="NONE", psolver="ANALYTIC"):
    """
    Create the &POISSON section for a CP2K input file.

    Parameters:
        periodic (str): Periodicity setting (e.g., NONE, XYZ, XY, Z).
        psolver (str): Poisson solver method (e.g., WAVELET, MULTIPOLE, FFT).

    Returns:
        str: Formatted &POISSON section.
    """
    string = "&POISSON\n"
    string += f"   PERIODIC {periodic}\n"
    string += f"   PSOLVER {psolver}\n"
    string += "&END\n"
    return string
def subsys_section(
    cell_file_name="2-Tetracene.cell",
    cell_file_format="CP2K",
    periodic="NONE",
    coord_file_name="2-Tetracene.xyz",
    coord_file_format="XYZ",
    kinds=None
):
    """
    Create the &SUBSYS section for a CP2K input file.

    Parameters:
        abc (tuple): Unit cell lengths (a, b, c).
        periodic (str): Periodicity setting (e.g., NONE, XYZ, XY, Z).
        coord_file_name (str): External coordinate file name.
        coord_file_format (str): Format of coordinate file (e.g., XYZ, PDB).
        kinds (list of dict): List of atom kinds, each dict contains:
            'element' (str): Element symbol, e.g., 'H', 'C'.
            'basis_set' (str): Basis set name.
            'potential' (str): Pseudopotential name.

    Returns:
        str: Formatted &SUBSYS section.
    """
    if kinds is None:
        kinds = [
            {"element": "H", "basis_set": "cc-TZV2P-GTH", "potential": "GTH-PBE-q1"},
            {"element": "C", "basis_set": "cc-TZV2P-GTH", "potential": "GTH-PBE-q4"},
        ]

    string = "&SUBSYS\n"
    string += "   &CELL\n"
    string += f"      CELL_FILE_NAME {cell_file_name}\n"
    string += f"      CELL_FILE_FORMAT {cell_file_format}\n"
    string += f"      PERIODIC {periodic}\n"
    string += "   &END CELL\n\n"
    string += "   &TOPOLOGY\n"
    string += f"      COORD_FILE_NAME {coord_file_name}\n"
    string += f"      COORD_FILE_FORMAT {coord_file_format}\n"
    string += "   &END\n\n"

    for kind in kinds:
        string += f"   &KIND {kind['element']}\n"
        string += f"      BASIS_SET {kind['basis_set']}\n"
        string += f"      POTENTIAL {kind['potential']}\n"
        string += "   &END KIND\n"

    string += "&END SUBSYS\n"
    return string
def force_stress_eval_print_section(
    forces_on=True,
    stresses_on=False,
    forces_filename="=Forces",
    stresses_filename="=Stresses",
    ndigits=15
):
    """
    Create the &PRINT section (with &FORCES) for the &FORCE_EVAL block in a CP2K input file.

    Parameters:
        forces_filename (str): Filename to output forces.
        ndigits (int): Number of significant digits in the output.
        forces_on (bool): Whether to include the &FORCES section.

    Returns:
        str: Formatted &PRINT section with optional &FORCES and &END FORCE_EVAL.
    """
    string = "&PRINT\n"
    
    if forces_on:
        string += "   &FORCES ON\n"
        string += f"      FILENAME {forces_filename}\n"
        string += f"      NDIGITS {ndigits}\n"
        string += "   &END FORCES\n"
    if stresses_on:
        string += "   &STRESS_TENSOR ON\n"
        string += f"      FILENAME {stresses_filename}\n"
        string += "   &END STRESS_TENSOR\n"

    string += "&END PRINT\n"
    return string 

def create_summary_file(input_file: str, output_name: str):
    """
    A sample function that reads an input and writes a summary.
    """
    # (Your function logic here)
    print(f"Reading from {input_file} and writing to {output_name}...")
    with open(output_name, 'w') as f:
        f.write(f"This is a summary of {input_file}.\n")
    print("Done.")

