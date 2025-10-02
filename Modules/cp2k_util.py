
from . import Read 
from .PhysConst import StandardAtomicWeights,ConversionFactors
from .coordinate_tools import backtransform_iterative_refinement,getTransAndRotEigenvectors,get_principle_axis_coordinates,cartesian_to_internal_coordinates,generate_internal_representation,get_B_non_redundant_internal,assign_bond_orders
from scipy.optimize import least_squares
import numpy as np
import os
import shutil
import subprocess
import ast
from datetime import datetime # Needed for timestamping the log
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
    
def write_pos_file(
    name,
    atoms,
    coordinates_bohr,
    cell_bohr,
    energy_Ha,
    forces_Ha_bohr,
    step,
    path,
    stress_Ha_bohr_3=None,
):
    """
    Write atomic structure, cell, energy, forces (and optionally stress) to a file.
    Appends if file exists.

    Parameters
    ----------
    name : str
        Name of the system
    atoms : list
        List of atom symbols
    coordinates_bohr : list
        List of (x, y, z) tuples/lists in Bohr
    cell_bohr : numpy.ndarray
        3x3 numpy array representing the unit cell (Bohr)
    energy_Ha : float
        Energy of the system (Hartree)
    forces_Ha_bohr : list
        List of force vectors in Ha/Bohr
    step : int
        Optimization step number
    path : str
        Path to the file
    stress_Ha_bohr_3 : numpy.ndarray, optional
        3x3 stress tensor in Ha/Bohr^3
    """
    with open(path, "a") as f:
        f.write(f"Optimization Step: {step}\n")
        
        # Write cell information
        f.write("Cell\n")
        for i in range(3):
            f.write(f"{cell_bohr[i,0]:.12f} {cell_bohr[i,1]:.12f} {cell_bohr[i,2]:.12f}\n")
        f.write("End Cell\n")
        
        # Write coordinates
        f.write("Coordinates\n")
        f.write(f"{len(atoms)}\n")
        f.write(f"{name}\n")
        for it, coord in enumerate(coordinates_bohr):
            f.write(f"{atoms[it]} {coord[0]:.12f} {coord[1]:.12f} {coord[2]:.12f}\n")
        f.write("End Coordinates\n")
        
        # Write energy
        f.write("Energy\n")
        f.write(f"E={energy_Ha:.12f}\n")
        f.write("End Energy\n")
        
        # Write forces
        f.write("Forces\n")
        for it, force in enumerate(forces_Ha_bohr):
            f.write(f"{atoms[it]} {force[0]:.12f} {force[1]:.12f} {force[2]:.12f}\n")
        f.write("End Forces\n")
        
        # Write stress if provided
        if stress_Ha_bohr_3 is not None and len(stress_Ha_bohr_3) == 3:
            f.write("Stress\n")
            for i in range(3):
                f.write(f"{stress_Ha_bohr_3[i,0]:.12f} {stress_Ha_bohr_3[i,1]:.12f} {stress_Ha_bohr_3[i,2]:.12f}\n")
            f.write("End Stress\n")
        
        f.write("\n")  # Blank line between steps

def write_geo_log_file(name, atoms, coordinates_bohr, energy_Ha, step, path):
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
        f.write(f"{len(atoms):.0f}\n")
        f.write(f"Optimization of {name} | Step: {step} | Energy[Ha]={energy_Ha:.12f}\n")
        for it, coord in enumerate(coordinates_bohr):
            coord=np.array(coord)*ConversionFactors["a.u.->A"]
            f.write(f"{atoms[it]} {coord[0]:.12f} {coord[1]:.12f} {coord[2]:.12f}\n")
        f.write("\n")  # Add blank line between steps for readability
def parse_last_optimization_step(path):
    """
    Parse the last optimization step from the file written by `write_pos_file`.

    Returns
    -------
    dict with keys:
    - name (str): System name
    - step (int): Optimization step number
    - atoms (list of str): Atom symbols
    - coordinates_bohr (list of [x,y,z]): Atomic positions in Bohr
    - cell_bohr (numpy.ndarray): 3x3 unit cell in Bohr
    - energy_Ha (float): Energy in Hartree
    - forces_Ha_bohr (list of [fx,fy,fz]): Forces in Ha/Bohr
    - stress_Ha_bohr_3 (numpy.ndarray or None): 3x3 stress tensor in Ha/Bohr^3, if present
    """
    import numpy as np

    last_block = []
    # Read all lines
    with open(path, "r") as f:
        lines = f.readlines()
    
    # Find the last "Optimization Step"
    for i, line in enumerate(lines):
        if line.startswith("Optimization Step:"):
            last_block = lines[i:]  # from here to the end
    
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
    stress = None
    
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
            i += 4  # move past "Cell" block
            
        elif line == "Coordinates":
            n_atoms = int(last_block[i+1].strip())
            name = last_block[i+2].strip()
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
            i = j  # move index to "End Forces"
        
        elif line == "Stress":
            stress = []
            for j in range(3):
                parts = last_block[i+1+j].split()
                stress.append([float(parts[0]), float(parts[1]), float(parts[2])])
            stress = np.array(stress)
            i += 4  # move past "Stress" block
        
        i += 1
    
    return {
        "name": name,
        "step": step,
        "atoms": atoms,
        "coordinates_bohr": coordinates,
        "cell_bohr": cell,
        "energy_Ha": energy,
        "forces_Ha_bohr": forces,
        "stress_Ha_bohr_3": stress
    }

def geo_opt(
    cluster="local",
    path="./",
    tol_max=5e-4,
    tol_drm=5e-4,
    max_iter=200,  #Add a maximum iteration limit
    stall_patience=4, # Number of steps with no progress before declaring a stall
    stall_energy_thresh=1e-7, #Energy change threshold for stalling
):
    """
    Geometry optimization wrapper with robust stall handling.
    """
    # ... (Your existing setup code remains the same) ...
    # -------------------------------
    # 1. Input & setup
    # -------------------------------
    geo_opt_path = os.path.join(path, "Geo_Optimization")
    xyz_filepath = Read.get_xyz_filename(path, verbose=True)

    with open(xyz_filepath, "r") as f:
        input_data = f.readlines()

    config = parse_cp2k_config(input_data)
    cp2k_exec = get_cp2k_exec(cluster=cluster)

    # -------------------------------
    # 2. Initialization / Restart
    # -------------------------------
    if not os.path.exists(geo_opt_path):
        os.makedirs(geo_opt_path, exist_ok=True)
        config=prepare_force_eval(path=geo_opt_path, config=config)
        energy_Ha, forces_Ha_bohr = run_energy_force_eval(
            path=geo_opt_path,
            cp2k_exec=cp2k_exec,
            runner="slurm" if cluster == "slurm" else "local"
        )
        name=config["name"]
        atoms=config["atoms"]
        masses=config["masses"]
        coordinates_A=config["coordinates_A"]
        coordinates_bohr=ConversionFactors["A->a.u."]*coordinates_A
        cell_A=config["cell_A"]
        cell_bohr=ConversionFactors["A->a.u."]*cell_A
        it = 0
        hessian = []
        write_pos_file(
            name, atoms, coordinates_bohr,cell_bohr, energy_Ha, forces_Ha_bohr,
            step=it, path=os.path.join(geo_opt_path, f"{name}.pos")
        )
        write_geo_log_file(name, atoms, coordinates_bohr, energy_Ha, step=it, path=os.path.join(geo_opt_path, f"{name}-pos.xyz"))
        print(f"Starting fresh optimization for {name}\n", flush=True)

    else:
        name=config["name"]
        atoms=config["atoms"]
        results = parse_last_optimization_step(path=os.path.join(geo_opt_path, f"{name}.pos"))
        if results["name"] == name:
            it = results["step"]
            forces_Ha_bohr = results["forces_Ha_bohr"]
            energy_Ha = results["energy_Ha"]
            coordinates_bohr = results["coordinates_bohr"]
            cell_bohr=results["cell_bohr"]
            print(f"Continuing optimization from step {it}\n", flush=True)
            try:
                hessian = np.load(os.path.join(geo_opt_path, "Hessian.npy"))
                print("Loaded Hessian from checkpoint\n", flush=True)
            except:
                hessian = []
                print("No Hessian found, using initial guess.\n", flush=True)

    _,rot_subspace=getTransAndRotEigenvectors(coordinates=coordinates_bohr,masses=masses)
    rot_projector=np.eye(len(rot_subspace[0]))
    for rotation in rot_subspace:
        rot_projector-=np.outer(rotation,rotation)
    force_history=[]

    # ... (Your printing header remains the same) ...
    print("=" * 135, flush=True)
    print(
        f"{'iter':>4} | {'E_tot':>12} | {'ΔE_tot':>12} | {'ΔE_pred':>12} | "
        f"{'rms force':>10} | {'max force':>10} | {'step size':>10} | "
        f"{'p_e [%]':>8} | {'rho':>10} | {'tru.rad.':>10} | {'acc?':>5} |",
        flush=True
    )
    print("=" * 135, flush=True)
    # -------------------------------
    # 3. Optimization loop
    # -------------------------------
    def forces_summary(forces):
        fmax = np.max(np.abs(forces))
        frms = np.sqrt(np.sum(np.array(forces)**2) / len(forces))
        return fmax, frms

    forces_Ha_bohr=rot_projector@np.array(forces_Ha_bohr).flatten()
    forces_Ha_bohr=forces_Ha_bohr.reshape((len(atoms), 3))
    force_max, drm_force = forces_summary(forces_Ha_bohr)
    delta = 0.25
    stall_counter = 0 # NEW: Centralized stall counter
    hessian_reset_active = False # NEW: Flag to track if we've already tried resetting H

    bonds, angles, dihedrals = generate_internal_representation(
        atomic_symbols=atoms, coordinates_bohr=coordinates_bohr,
        cell_bohr=cell_bohr
    )
    internals=(bonds,angles,dihedrals)
    
    print(f"{it:4d} | {energy_Ha:12.8f} | {'--------':>12} | {'--------':>12} | "
        f"{drm_force:10.6f} | {force_max:10.6f} | {'--------':>10} | {0.0:8.1f} | "
        f"{'--------':>10} | {delta:10.6f} | {'Y':>5} |", flush=True)

    # MODIFIED: Added max_iter check to the loop condition
    while (force_max > tol_max or drm_force > tol_drm) and it < max_iter:
        it += 1
        energy_Ha_prev = energy_Ha

        # Always try the BFGS step first unless forces are extremely small
        if drm_force > 1e-5:
            # MODIFIED: Call the updated bfgs_step function
            coordinates_bohr, forces_Ha_bohr, energy_Ha, hessian, delta, accepted, info = bfgs_step_coord_internal(
                path=geo_opt_path,
                name=name,
                atoms=atoms,
                coordinates_bohr=coordinates_bohr,
                internals=internals,
                forces_Ha_bohr=forces_Ha_bohr,
                energy_Ha=energy_Ha,
                hessian=hessian,
                delta=delta,
                cp2k_exec=cp2k_exec,
                runner="slurm" if cluster == "slurm" else "local",
            )
            forces_Ha_bohr = rot_projector @ np.array(forces_Ha_bohr).flatten()
            forces_Ha_bohr = forces_Ha_bohr.reshape((len(atoms), 3))

            # NEW: Stall detection logic is now here
            energy_change = abs(energy_Ha - energy_Ha_prev)

            if energy_change < stall_energy_thresh:
                stall_counter += 1
            else:
                # If we make progress, reset stall counters
                stall_counter = 0
                hessian_reset_active = False

            # NEW: Tiered stall recovery logic
            if stall_counter >= stall_patience:
                print(f"\nWarning: Stalled for {stall_counter} steps. Taking corrective action.", flush=True)
                if not hessian_reset_active:
                    print(" -> Action: Resetting Hessian matrix.", flush=True)
                    hessian = [] # Reset Hessian, it will be re-initialized in the next step
                    hessian_reset_active = True
                else:
                    print(" -> Action: Hessian already reset. Forcing a DIIS step to escape.", flush=True)
                    # Force a DIIS step as a more drastic measure
                    coordinates_bohr, forces_Ha_bohr, energy_Ha = diis_step_coord_cart(
                         path=geo_opt_path, name=name, atoms=atoms, force_history=force_history,
                         cp2k_exec=cp2k_exec, runner="slurm" if cluster == "slurm" else "local"
                    )
                    hessian_reset_active = False # Allow Hessian reset again after DIIS
                stall_counter = 0 # Reset counter after taking action

        else: # Switch to DIIS for final convergence
            coordinates_bohr, forces_Ha_bohr, energy_Ha = diis_step_coord_cart(
                path=geo_opt_path, name=name, atoms=atoms, force_history=force_history,
                cp2k_exec=cp2k_exec, runner="slurm" if cluster == "slurm" else "local"
            )

        if accepted:
            np.save(os.path.join(geo_opt_path, "Hessian.npy"), hessian)
            force_history.append((np.array(coordinates_bohr).flatten(),np.array(forces_Ha_bohr).flatten()))
            write_pos_file(
                name=name, atoms=atoms, coordinates_bohr=coordinates_bohr, cell_bohr=cell_bohr,
                energy_Ha=energy_Ha, forces_Ha_bohr=forces_Ha_bohr,
                step=it, path=os.path.join(geo_opt_path, f"{name}.pos")
            )
            write_geo_log_file(name=name, atoms=atoms, coordinates_bohr=coordinates_bohr,
                                energy_Ha=energy_Ha, step=it, path=os.path.join(geo_opt_path, f"{name}-pos.xyz"))

        force_max, drm_force = forces_summary(forces_Ha_bohr)
        # ... (Your print statement for the log remains the same) ...
        print(f"{it:4d} | {energy_Ha:12.8f} | {energy_Ha - energy_Ha_prev:12.8f} | {info.get('pred', '--------'):12.8f} | "
            f"{drm_force:10.6f} | {force_max:10.6f} | {info.get('step_size', '--------'):10.6f} | "
            f"{np.round(info.get('projection_error', 0)/max(force_max,1e-12)*100,0):8.1f} | "
            f"{info.get('rho', '--------'):10.6f} | {delta:10.6f} | "
            f"{'Y' if accepted else 'N':>5} |", flush=True)

    print("=" * 135, flush=True)
    
    # NEW: Final status message
    if force_max <= tol_max and drm_force <= tol_drm:
        print(f"Geometry optimization converged successfully in {it} steps.")
    elif it >= max_iter:
        print(f"Optimization stopped after reaching the maximum of {max_iter} iterations.")
    else:
        print("Optimization finished for other reasons.")


    # ... (Your finalization code remains the same) ...
    # -------------------------------
    # 4. Finalize optimized structure
    # -------------------------------
    shutil.copyfile(
        os.path.join(geo_opt_path, f"{name}_tmp.xyz"),
        os.path.join(geo_opt_path, f"{name}_opt.xyz")
    )





def bfgs_step_coord_internal(
    path,
    name,
    atoms,
    coordinates_bohr,
    internals,
    forces_Ha_bohr,
    energy_Ha,
    hessian,
    delta,
    # MODIFIED: stall_counter and related parameters are removed
    cp2k_exec="./cp2k.popt",
    runner="local",
    delta_min=0.005,
    delta_max=0.5,
):
    """
    Perform one Trust-Radius BFGS optimization step.
    (Stall detection is now handled by the parent geo_opt function).
    """
    # ... (Most of your existing code in this function remains identical) ...
    # ---------------------------------
    # 1. Define coordinate system at current point (k)
    # ---------------------------------
    forces_vec_cart_k = np.array(forces_Ha_bohr).flatten()
    x_k_cart = np.array(coordinates_bohr).flatten()
    if np.any(np.isnan(forces_vec_cart_k)) or np.any(np.isinf(forces_vec_cart_k)):
        raise ValueError("Invalid forces encountered at starting point.")

    bonds=internals[0]
    angles=internals[1]
    dihedrals=internals[2]
    
    PINV_RCOND_TOL = 1e-5

    q_k, B_prim = cartesian_to_internal_coordinates(np.array(coordinates_bohr), bonds, angles, dihedrals)
    B, U = get_B_non_redundant_internal(B_prim)
    B_plus_T = np.linalg.pinv(B,rcond=PINV_RCOND_TOL).T
    projection_error=np.max(np.abs(forces_vec_cart_k-(B.T @ B_plus_T)@forces_vec_cart_k))
    forces_q_k =B_plus_T @ forces_vec_cart_k
    grad_q_k = -forces_q_k
    
    if len(hessian) == 0:
        hessian=np.eye(len(grad_q_k))
    
    step_type = "Trust"
    
    # ... (The rest of the function for taking the step, evaluating, and updating Hessian is the same) ...
    #dx=solve_tr_subproblem_internal_coords(grad_q_k, hessian, B, delta, verbose=False)
    dq=solve_tr_subproblem_internal_coords(grad_q_k, hessian, B, delta, verbose=False)
    dx = B_plus_T.T@dq
    x_new_cart=x_k_cart+dx
    coordinates_new_bohr = np.real(x_new_cart.reshape((len(atoms), 3)))
    
    file_to_update = os.path.join(path, f"{name}_tmp.xyz")
    with open(file_to_update, "w") as f:
        f.write(str(len(atoms)) + "\n\n")
        for atom, coord in zip(atoms, coordinates_new_bohr):
            f.write(f"{atom:2s} {coord[0]*ConversionFactors['a.u.->A']:20.12f} {coord[1]*ConversionFactors['a.u.->A']:20.12f} {coord[2]*ConversionFactors['a.u.->A']:20.12f}\n")
    
    energy_new_Ha, forces_new_Ha_bohr = run_energy_force_eval(path=path, cp2k_exec=cp2k_exec, runner=runner)

    ared = energy_new_Ha-energy_Ha
    pred = (grad_q_k.T @ dq + 0.5 * dq.T @ hessian @ dq)
    rho = ared / pred
    delta_new, accepted = update_trust_rad(rho=rho, delta=delta,delta_min=delta_min,delta_max=delta_max)
    step_size = np.linalg.norm(dx)

    if accepted:
        coordinates_out = coordinates_new_bohr.tolist()
        q_k_plus_1, B_prim_new = cartesian_to_internal_coordinates(coordinates_new_bohr, bonds, angles, dihedrals)
        B_new, U_new = get_B_non_redundant_internal(B_prim_new)
        B_plus_new_T = np.linalg.pinv(B_new, rcond=PINV_RCOND_TOL).T
        forces_out = np.array(forces_new_Ha_bohr).flatten()
        forces_out=forces_out.reshape((len(atoms), 3))
        energy_out = energy_new_Ha
        
        s_k_actual_redundant = q_k_plus_1 - q_k
        dihedral_angle_start_index = len(bonds)+len(angles)
        for i in range(dihedral_angle_start_index, len(s_k_actual_redundant)):
            if abs(s_k_actual_redundant[i]) > np.pi:
                s_k_actual_redundant[i] -= np.sign(s_k_actual_redundant[i]) * 2*np.pi
        s_update = U_new.T @ s_k_actual_redundant
        if np.shape(U_new)==np.shape(U):
            dg_new = -B_plus_new_T@(np.array(forces_new_Ha_bohr).flatten()-np.array(forces_Ha_bohr).flatten())
        else:
            H_cart = B.T @ np.linalg.inv(hessian) @ B
            hessian = B_plus_new_T @ H_cart @ B_plus_new_T.T
            dg_new = -B_plus_new_T@(np.array(forces_new_Ha_bohr).flatten()-np.array(forces_Ha_bohr).flatten())
        hessian=bfgs_update(s_update,dg_new, hessian)
    else:
        coordinates_out = coordinates_bohr
        forces_out = forces_Ha_bohr
        energy_out = energy_Ha

    info = { "step_type":step_type, "rho": rho, "ared": ared, "pred":pred, "projection_error":projection_error, "step_size":step_size }
    
    # MODIFIED: Removed stall_counter from the return statement
    return coordinates_out, forces_out, energy_out, hessian, delta_new, accepted, info
def diis_step_coord_cart(
    path,
    name,
    atoms,
    force_history,
    N=5,
    cp2k_exec="./cp2k.popt",
    runner="local"
):
    """
    Perform one geometric DIIS step using force history in cart. coordinates.
    """

    if len(force_history) > N:
        force_history.pop(0)  # keep last N points


    # ---------------------------------
    # 3. Build DIIS matrix B_ij = <f_i|f_j>
    # ---------------------------------
    m = len(force_history)
    F = [f for _, f in force_history]
    Bmat = np.empty((m+1, m+1))
    Bmat[-1, :] = -1
    Bmat[:, -1] = -1
    Bmat[-1, -1] = 0
    for i in range(m):
        for j in range(m):
            Bmat[i, j] = np.dot(F[i], F[j])

    # ---------------------------------
    # 4. Solve for coefficients
    # ---------------------------------
    rhs = np.zeros(m+1)
    rhs[-1] = -1
    coeffs = np.linalg.solve(Bmat, rhs)[:-1]  # last element is Lagrange multiplier

    # ---------------------------------
    # 5. Extrapolate geometry
    # ---------------------------------
    Q = [q for q, _ in force_history]
    coords_new = sum(c * q for c, q in zip(coeffs, Q))
    coords_current = Q[-1]
    disp = coords_new - coords_current
    '''
    if norm_disp > max_step:
        disp *= max_step / norm_disp  # rescale displacement
        coords_new = coords_current + disp
    '''
    # Write temporary geometry for evaluation
    file_to_update = os.path.join(path, f"{name}_tmp.xyz")
    with open(file_to_update, "w") as f:
        f.write(str(len(atoms)) + "\n\n")
        for it,atom in enumerate(atoms):
            f.write(f"{atom:2s} {coords_new[3*it]*ConversionFactors['a.u.->A']:20.12f} "
                    f"{coords_new[3*it+1]*ConversionFactors['a.u.->A']:20.12f} "
                    f"{coords_new[3*it+2]*ConversionFactors['a.u.->A']:20.12f}\n")

    # ---------------------------------
    # 7. Evaluate new energy & forces
    # ---------------------------------
    energy_new, forces_new = run_energy_force_eval(path=path, cp2k_exec=cp2k_exec, runner=runner)
    coordinates_new_bohr = np.real(coords_new.reshape((len(atoms), 3)))
    return coordinates_new_bohr, forces_new, energy_new
# --- New Wrapper Function ---
def back_transform_iterative(
    x_initial_cart,       # Initial Cartesian coords (flat array, Bohr)
    q_initial_prim,       # Initial PRIMITIVE internal coords
    p_q_non_redundant,    # The desired step in NON-REDUNDANT internals
    B_plus_T,
    U_matrix,             # Transformation from non-redundant to primitive space
    internals,            # Definition of bonds, angles, etc.
    ftol=1e-6,            # Convergence tolerance for the solver
    xtol=1e-6
):
    """
    Iteratively finds the Cartesian coordinates that correspond to a
    step in internal coordinates using scipy.optimize.least_squares.
    """
    # 1. Define the target in the primitive (redundant) coordinate space
    p_q_prim = U_matrix @ p_q_non_redundant
    q_target = q_initial_prim + p_q_prim

    # Define indices for angle wrapping
    num_bonds = len(internals[0])
    num_angles = len(internals[1])
    dihedral_start_idx = num_bonds + num_angles

    # 2. Define the residual function for the least-squares solver
    # This function calculates: q(x_initial + dx) - q_target
    def residual_function(dx, x_initial, q_target_vec):
        """Calculates the vector of errors in internal coordinates."""
        # Current geometry based on the step dx
        x_current = x_initial + dx
        
        # Calculate the internal coordinates for the current geometry
        q_current, _ = cartesian_to_internal_coordinates(
            x_current.reshape(-1, 3), *internals
        )

        # Calculate the error, correctly wrapping dihedral angles
        error = q_current - q_target_vec
        for j in range(dihedral_start_idx, len(error)):
            # Wrap the error to be in the range [-pi, pi]
            while error[j] > np.pi: error[j] -= 2 * np.pi
            while error[j] < -np.pi: error[j] += 2 * np.pi
            
        return error

    # 3. Define the Jacobian function for the solver
    # The Jacobian of the residual is simply the Wilson B-matrix
    def jacobian_function(dx, x_initial,q_target):
        """Calculates the Wilson B-matrix at the current geometry."""
        x_current = x_initial + dx
        _, B_prim = cartesian_to_internal_coordinates(
            x_current.reshape(-1, 3), *internals
        )
        return B_prim

    # 4. Set the initial guess for the step 'dx'
    # A zero vector is a safe and simple initial guess.
    dx_initial_guess = B_plus_T.T@p_q_non_redundant
    if np.linalg.norm(dx_initial_guess)>0.025:
        # 5. Run the least-squares optimization
        result = least_squares(
            fun=residual_function,
            x0=dx_initial_guess,
            jac=jacobian_function,
            args=(x_initial_cart,q_target), # Extra args for our functions
            method='trf',  # Trust Region Reflective is robust
            ftol=ftol,
            xtol=xtol,
            verbose=0 # Change to 1 or 2 for debugging
        )

        # 6. Check for success and return the final geometry
        if not result.success:
            print(f"Warning: Back-transformation did not converge. Message: {result.message}")

        final_dx = result.x
        x_final = x_initial_cart + final_dx
    else:
        x_final=x_final = x_initial_cart + dx_initial_guess
    
    return x_final

def solve_tr_subproblem_internal_coords(g_q, H_q, B, delta, verbose=False):
    """
    Solves the TR subproblem in internal coordinates with a physically
    meaningful (Cartesian) metric and returns the Cartesian step.

    Parameters:
    - g_q: Gradient in internal coordinates.
    - H_q: Hessian in internal coordinates.
    - B: The Wilson B-matrix (your Bq).
    - delta: Trust radius, defined as the RMS norm in Cartesian space.
    - verbose: Verbosity flag.

    Returns:
    - p_x: The optimal step in Cartesian coordinates.
    """
    # 1. Calculate the G matrix and its Cholesky factor L
    # We use pinv for stability in case of redundant coordinates
    G = B @ B.T
    try:
        # Use Cholesky if G is positive definite
        L = np.linalg.cholesky(G)
    except np.linalg.LinAlgError:
        # Fallback for semi-definite G (due to redundancies)
        # Use eigendecomposition: G = Q Lmbda Q.T => G^1/2 = Q sqrt(Lmbda) Q.T
        eigvals, eigvecs = np.linalg.eigh(G)
        eigvals[eigvals < 1e-9] = 0 # Clamp small/negative eigenvalues
        L = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


    # 2. Transform gradient and Hessian to the scaled system 's'
    g_s = L.T @ g_q
    H_s = L.T @ H_q @ L
    
    if verbose:
        print("Solving TR subproblem in scaled internal coordinates.")

    # 3. Solve the standard TR subproblem in the 's' coordinate system
    # The norm of p_s corresponds to the Cartesian norm of the step.
    p_s = more_sorensen_secular(g_s, H_s, delta, verbose=verbose)

    # 4. Back-transform the step to get the Cartesian step p_x
    # p_s -> p_q -> p_x
    dq = L @ p_s
    
    ## Use the pseudoinverse of B to get the Cartesian step
    #B_inv = np.linalg.pinv(B)
    #p_x = B_inv @ p_q

    #if verbose:
    #    print(f"Final ||p_x|| = {np.linalg.norm(p_x):.6f} (target δ={delta})")

    return dq
def more_sorensen_secular(g_q, H_q, delta, tol=1e-12, max_iter=50, verbose=False):
    """
    Solve the trust-region subproblem using the More–Sorensen algorithm
    with the secular equation for guaranteed convergence.

    Parameters
    ----------
    g : ndarray
        Gradient vector (n,).
    B : ndarray
        Symmetric positive definite Hessian approximation (n x n).
    delta : float
        Trust-region radius.
    tol : float
        Tolerance for norm convergence.
    max_iter : int
        Maximum number of Newton iterations on the secular equation.
    verbose : bool
        If True, prints convergence messages.

    Returns
    -------
    p : ndarray
        Optimal step solving min g^T p + 0.5 p^T B p with ||p|| <= delta.
    """

    # Try unconstrained minimizer first
    try:
        p_star = -np.linalg.solve(H_q, g_q)
        if np.linalg.norm(p_star) <= delta:
            if verbose:
                print("Unconstrained minimizer inside trust region.")
            return p_star
    except np.linalg.LinAlgError:
        pass

    # Eigen decomposition for secular equation
    # B = Q Λ Q^T, transform problem to diagonal coordinates
    eigvals, Q = np.linalg.eigh(H_q)
    g_hat = Q.T @ g_q


    # Newton iteration on secular equation φ(λ)=0, λ >= 0
    # initial guess: shift away from negative eigvals
    lmbda = max(0, -np.min(eigvals) + 1e-12)

    for k in range(max_iter):
        denom = eigvals + lmbda
        phi_val = np.sum((g_hat / denom)**2) - delta**2
        if abs(phi_val) < tol:
            break
        # derivative φ'(λ) = -2 Σ (g_i/(λ_i+λ)^3 g_i)
        phi_prime = -2 * np.sum((g_hat**2) / (denom**3))
        lmbda -= phi_val / phi_prime
        lmbda = max(lmbda, 0)  # stay feasible

    # Compute final p(λ)
    denom = eigvals + lmbda
    p_hat = -(g_hat / denom)
    p = Q @ p_hat

    if verbose:
        print(f"More–Sorensen (secular) converged in {k+1} iterations, λ={lmbda:.3e}")
        print(f"||p|| = {np.linalg.norm(p):.6f} (target δ={delta})")

    return p

def bfgs_update(dx: np.ndarray, 
                dg: np.ndarray, 
                hess_mat: np.ndarray,
                min_curv: float = 0.2
               ) -> np.ndarray:
    """
    Perform BFGS Hessian update with Powell-style damping.
    
    H_new = H_old + (theta*dg⊗dg)/(dx·dg) - (H*dx⊗H*dx)/(dx·H*dx)
    where theta adjusts dg to enforce sufficient curvature.

    Parameters
    ----------
    dx : np.ndarray
        Step vector (change in positions)
    dg : np.ndarray
        Gradient difference vector (change in gradients)
    hess_mat : np.ndarray
        Current Hessian approximation
    min_curv : float
        Minimum curvature ratio to enforce

    Returns
    -------
    np.ndarray
        Updated Hessian matrix
    """
    
    work = hess_mat @ dx
    dxw = np.dot(dx, work)
    gdx = np.dot(dg, dx)
    
    if abs(gdx) < 1e-8 or abs(dxw) < 1e-8:
        print("Warning: Small curvature, skipping BFGS update")
        return hess_mat

    # ---------------------
    # Damping condition (Powell)
    # ---------------------
    if gdx < min_curv * dxw:
        # Compute theta
        theta = (1 - min_curv) * dxw / (dxw - gdx)
        dg_damped = theta * dg + (1 - theta) * work
    else:
        dg_damped = dg

    
    # Outer products for BFGS update
    hess_mat += (np.outer(dg_damped, dg_damped) / np.dot(dg_damped, dx) \
                - np.outer(work, work) / dxw)
    hess_mat=0.5*(hess_mat+hess_mat.T)

    return hess_mat



def update_trust_rad(rho: float, delta: float, delta_min: float,delta_max: float) -> float:
    """
    Update trust-region radius based on the ratio of actual vs. predicted reduction (rho).

    This implementation follows a more standard algorithm (e.g., Nocedal & Wright, 
    "Numerical Optimization", Ch. 4).

    Parameters
    ----------
    rho : float
        Ratio of actual vs. predicted reduction (ρ = ΔE_actual / ΔE_pred).
    delta : float
        Current trust-region radius in Bohr.
    ediff : float
        Energy difference (negative if energy decreased).

    Returns
    -------
    float
        Updated trust-region radius in Bohr.
    """
    # Define boundaries and thresholds
    max_trust = delta_max
    min_trust = delta_min
    eta1 = 0.15  # Threshold for shrinking
    eta2 = 0.75  # Threshold for expansion
    
    gamma_shrink = 0.75  # Factor to shrink radius
    gamma_shrink_large = 0.25  # Factor to shrink radius
    gamma_expand = 1.5   # Factor to expand radius


    # Update radius based on model accuracy (rho)
    if rho<0:
        delta_new = delta * gamma_shrink_large
        return max(delta_new,min_trust),False
    elif rho < eta1:
        # Poor agreement: model is unreliable. Shrink the trust region.
        delta_new = delta * gamma_shrink
        return max(delta_new,min_trust),False
    elif rho > eta2 and rho < 1.5:
        # Excellent agreement: model is reliable. Expand the trust region.
        delta_new = min(delta * gamma_expand, max_trust)
    elif rho >2:
        delta_new=gamma_shrink*delta
    else:
        # Good agreement: leave the trust region as is.
        delta_new = delta

    # Ensure the new radius is not smaller than the minimum
    return max(delta_new, min_trust),True


# Reference single-bond force constants (k_ref) in Hartree/Bohr^2.
# Updated reference force constants 
REFERENCE_lnK_BOND = {
    frozenset({'H', 'H'}): 4.661,  
    frozenset({'C', 'C'}): 7.643,  
    frozenset({'N', 'N'}): 7.634,  
    frozenset({'O', 'O'}): 7.561,  
    frozenset({'F', 'F'}): 7.358,  
    frozenset({'Cl', 'Cl'}): 8.648,  
    frozenset({'Br', 'Br'}): 9.012,  
    frozenset({'I', 'I'}): 9.511,  
    frozenset({'P', 'P'}): 8.805,  
    frozenset({'S', 'S'}): 8.316,  
    frozenset({'H', 'C'}): 6.217,  
    frozenset({'H', 'N'}): 6.057,  
    frozenset({'H', 'O'}): 5.794,  
    frozenset({'H', 'F'}): 5.600,  
    frozenset({'H', 'Cl'}): 6.937,  
    frozenset({'H', 'Br'}): 7.301,  
    frozenset({'H', 'I'}): 7.802,  
    frozenset({'H', 'P'}): 7.257,  
    frozenset({'H', 'S'}): 7.018,  
    frozenset({'C', 'N'}): 7.504,  
    frozenset({'C', 'O'}): 7.347,  
    frozenset({'C', 'F'}): 7.227,  
    frozenset({'C', 'Cl'}): 8.241,  
    frozenset({'C', 'Br'}): 8.478,  
    frozenset({'C', 'I'}): 8.859,  
    frozenset({'C', 'P'}): 8.237,  
    frozenset({'C', 'S'}): 8.117,  
    frozenset({'N', 'O'}): 7.526,  
    frozenset({'N', 'F'}): 7.475,  
    frozenset({'N', 'Cl'}): 8.266,  
    frozenset({'N', 'Br'}): 8.593,  
    frozenset({'N', 'I'}): 8.963,  
    frozenset({'N', 'P'}): 8.212,  
    frozenset({'N', 'S'}): 8.073,  
    frozenset({'O', 'F'}): 7.375,  
    frozenset({'O', 'Cl'}): 8.097,  
    frozenset({'O', 'Br'}): 8.276,  
    frozenset({'O', 'I'}): 8.854,  
    frozenset({'O', 'P'}): 7.957,  
    frozenset({'O', 'S'}): 7.922,  
    frozenset({'F', 'Cl'}): 7.947,  
    frozenset({'Cl', 'I'}): 9.309,  
    frozenset({'Br', 'I'}): 9.380,  
    frozenset({'F', 'P'}): 7.592,  
    frozenset({'F', 'S'}): 7.733,  
    frozenset({'Cl', 'P'}): 8.656,  
    frozenset({'Cl', 'S'}): 8.619,  
    frozenset({'Br', 'P'}): 8.729,  
    frozenset({'Br', 'S'}): 8.728,  
    frozenset({'I', 'P'}): 9.058,  
    frozenset({'I', 'S'}): 9.161,  
    frozenset({'P', 'S'}): 8.465,
}
DEFAULT_lnK_BOND = 7.0  
def get_K_bond(atom1,atom2,r1_bohr,r2_bohr,m=4.5):
    r1_A=r1_bohr*ConversionFactors["a.u.->A"]
    r2_A=r2_bohr*ConversionFactors["a.u.->A"]
    conversion_factor_kcal_mol_A2_Ha_bohr2=0.00044626155
    ln_Kij = REFERENCE_lnK_BOND[frozenset({atom1, atom2})]
    Kr_Ha_bohr2=conversion_factor_kcal_mol_A2_Ha_bohr2*np.exp(ln_Kij)/(np.linalg.norm(r1_A-r2_A))**(m)
    return Kr_Ha_bohr2

REFERENCE_R0_BOND_A = {
    frozenset({'H', 'H'}): 0.738,  
    frozenset({'C', 'C'}): 1.526,  
    frozenset({'N', 'N'}): 1.441,  
    frozenset({'O', 'O'}): 1.460,  
    frozenset({'F', 'F'}): 1.406,  
    frozenset({'Cl', 'Cl'}): 2.031,  
    frozenset({'Br', 'Br'}): 2.337,  
    frozenset({'I', 'I'}): 2.836,  
    frozenset({'P', 'P'}): 2.324,  
    frozenset({'S', 'S'}): 2.038,  
    frozenset({'H', 'C'}): 1.090,  
    frozenset({'H', 'N'}): 1.010,  
    frozenset({'H', 'O'}): 0.960,  
    frozenset({'H', 'F'}): 0.920,  
    frozenset({'H', 'Cl'}): 1.280,  
    frozenset({'H', 'Br'}): 1.410,  
    frozenset({'H', 'I'}): 1.600,  
    frozenset({'H', 'P'}): 1.410,  
    frozenset({'H', 'S'}): 1.340,  
    frozenset({'C', 'N'}): 1.470,  
    frozenset({'C', 'O'}): 1.440,  
    frozenset({'C', 'F'}): 1.370,  
    frozenset({'C', 'Cl'}): 1.800,  
    frozenset({'C', 'Br'}): 1.940,  
    frozenset({'C', 'I'}): 2.160,  
    frozenset({'C', 'P'}): 1.830,  
    frozenset({'C', 'S'}): 1.820,  
    frozenset({'N', 'O'}): 1.420,  
    frozenset({'N', 'F'}): 1.420,  
    frozenset({'N', 'Cl'}): 1.750,  
    frozenset({'N', 'Br'}): 1.930,  
    frozenset({'N', 'I'}): 2.120,  
    frozenset({'N', 'P'}): 1.720,  
    frozenset({'N', 'S'}): 1.690,  
    frozenset({'O', 'F'}): 1.410,  
    frozenset({'O', 'Cl'}): 1.700,  
    frozenset({'O', 'Br'}): 1.790,  
    frozenset({'O', 'I'}): 2.110,  
    frozenset({'O', 'P'}): 1.640,  
    frozenset({'O', 'S'}): 1.650,  
    frozenset({'F', 'Cl'}): 1.648,  
    frozenset({'Cl', 'I'}): 2.550,  
    frozenset({'Br', 'I'}): 2.671,  
    frozenset({'F', 'P'}): 1.500,  
    frozenset({'F', 'S'}): 1.580,  
    frozenset({'Cl', 'P'}): 2.040,  
    frozenset({'Cl', 'S'}): 2.030,  
    frozenset({'Br', 'P'}): 2.240,  
    frozenset({'Br', 'S'}): 2.210,  
    frozenset({'I', 'P'}): 2.490,  
    frozenset({'I', 'S'}): 2.560,  
    frozenset({'P', 'S'}): 2.120,
}
C_Values={
"C":1.339,
"N":1.300,
"O":1.249,
"P":0.906,
"S":1.448
}
Z_Values= {
    'H': 0.784,
    'C': 1.183,
    'N': 1.212,
    'O': 1.219,
    'F': 1.166,
    'Cl': 1.272,
    'Br': 1.378,
    'I': 1.398,
    'P': 1.620,
    'S': 1.280,
}
REFERENCE_ANGLE_BOND_ORDER = {
    
    # ==================== HYDROGEN-CENTERED ANGLES ====================
    # Note: Hydrogen can only form one bond, so these are theoretical/transition state cases
    
    # ==================== CARBON-CENTERED ANGLES ====================
    
    # sp³ Carbon (tetrahedral, ~109.5°)
    ('H', 'C', 'H', (1, 1)): 109.5,
    ('H', 'C', 'C', (1, 1)): 109.5,
    ('H', 'C', 'N', (1, 1)): 109.5,
    ('H', 'C', 'O', (1, 1)): 109.5,
    ('H', 'C', 'F', (1, 1)): 109.5,
    ('H', 'C', 'Cl', (1, 1)): 109.5,
    ('H', 'C', 'Br', (1, 1)): 109.5,
    ('H', 'C', 'I', (1, 1)): 109.5,
    ('H', 'C', 'P', (1, 1)): 109.5,
    ('H', 'C', 'S', (1, 1)): 109.5,
    ('C', 'C', 'C', (1, 1)): 109.5,
    ('C', 'C', 'N', (1, 1)): 109.5,
    ('C', 'C', 'O', (1, 1)): 109.5,
    ('C', 'C', 'F', (1, 1)): 109.5,
    ('C', 'C', 'Cl', (1, 1)): 109.5,
    ('C', 'C', 'Br', (1, 1)): 109.5,
    ('C', 'C', 'I', (1, 1)): 109.5,
    ('C', 'C', 'P', (1, 1)): 109.5,
    ('C', 'C', 'S', (1, 1)): 109.5,
    ('N', 'C', 'N', (1, 1)): 109.5,
    ('N', 'C', 'O', (1, 1)): 109.5,
    ('O', 'C', 'O', (1, 1)): 109.5,
    
    # sp² Carbon (trigonal planar, ~120°)
    ('H', 'C', 'H', (1, 2)): 120.0,  # One double bond
    ('H', 'C', 'C', (1, 2)): 120.0,
    ('H', 'C', 'N', (1, 2)): 120.0,
    ('H', 'C', 'O', (1, 2)): 120.0,
    ('C', 'C', 'C', (1, 2)): 120.0,
    ('C', 'C', 'C', (2, 1)): 120.0,
    ('C', 'C', 'N', (1, 2)): 120.0,
    ('C', 'C', 'N', (2, 1)): 120.0,
    ('C', 'C', 'O', (1, 2)): 120.0,
    ('C', 'C', 'O', (2, 1)): 120.0,
    ('N', 'C', 'N', (2, 1)): 120.0,
    ('N', 'C', 'O', (2, 1)): 120.0,
    ('O', 'C', 'O', (2, 1)): 120.0,
    ('H', 'C', 'F', (1, 2)): 120.0,
    ('H', 'C', 'Cl', (1, 2)): 120.0,
    ('H', 'C', 'Br', (1, 2)): 120.0,
    ('H', 'C', 'I', (1, 2)): 120.0,
    
    # sp Carbon (linear, 180°)
    ('H', 'C', 'C', (1, 3)): 180.0,  # Triple bond
    ('H', 'C', 'N', (1, 3)): 180.0,
    ('C', 'C', 'C', (1, 3)): 180.0,
    ('C', 'C', 'C', (3, 1)): 180.0,
    ('C', 'C', 'N', (3, 1)): 180.0,
    ('C', 'C', 'N', (1, 3)): 180.0,
    
    # Aromatic Carbon (sp² with delocalized electrons, ~120°)
    ('H', 'C', 'C', (1, 1.5)): 120.0,
    ('C', 'C', 'H', (1.25, 1)):120.0,
    ('C', 'C', 'H', (1.75, 1)):120.0,
    ('H', 'C', 'N', (1, 1.5)): 120.0,
    ('C', 'C', 'C', (1.5, 1.5)): 120.0,
    ('C', 'C', 'C', (1.5, 1.25)):120.0,
    ('C', 'C', 'C', (1.25, 1.25)):120.0,
    ('C', 'C', 'C', (1.25, 1.375)):120.0,
    ('C', 'C', 'C', (1.25, 1.5)):120.0,
    ('C', 'C', 'C', (1.25, 1.75)):120.0,
    ('C', 'C', 'C', (1.75, 1.25)):120.0,
    ('C', 'C', 'N', (1.5, 1.5)): 120.0,
    ('C', 'C', 'O', (1.5, 1)): 120.0,
    ('C', 'C', 'F', (1.5, 1)): 120.0,
    ('C', 'C', 'Cl', (1.5, 1)): 120.0,
    ('C', 'C', 'Br', (1.5, 1)): 120.0,
    ('C', 'C', 'I', (1.5, 1)): 120.0,
    ('N', 'C', 'N', (1.5, 1.5)): 120.0,
    
    # ==================== NITROGEN-CENTERED ANGLES ====================
    
    # sp³ Nitrogen (pyramidal, ~107°) - lone pair effect
    ('H', 'N', 'H', (1, 1)): 107.0,
    ('H', 'N', 'C', (1, 1)): 107.0,
    ('C', 'N', 'C', (1, 1)): 107.0,  # Amines
    ('C', 'N', 'H', (1, 1)): 107.0,
    
    # sp² Nitrogen (trigonal planar, ~120°)
    ('C', 'N', 'C', (1, 2)): 120.0,  # Imines, amides
    ('C', 'N', 'C', (2, 1)): 120.0,
    ('C', 'N', 'H', (2, 1)): 120.0,
    ('C', 'N', 'O', (1, 2)): 120.0,  # Nitro compounds
    ('C', 'N', 'O', (2, 1)): 120.0,
    ('O', 'N', 'O', (2, 2)): 120.0,  # NO₂ group
    
    # sp Nitrogen (linear, 180°)
    ('C', 'N', 'N', (1, 3)): 180.0,  # Nitriles, diazo
    ('C', 'N', 'O', (3, 1)): 180.0,
    
    # Aromatic Nitrogen
    ('C', 'N', 'C', (1.5, 1.5)): 120.0,  # Pyridine-like
    ('C', 'N', 'H', (1.5, 1)): 120.0,
    
    # ==================== OXYGEN-CENTERED ANGLES ====================
    
    # sp³ Oxygen (bent, ~104.5°) - two lone pairs
    ('H', 'O', 'H', (1, 1)): 104.5,  # Water
    ('H', 'O', 'C', (1, 1)): 104.5,  # Alcohols
    ('C', 'O', 'C', (1, 1)): 104.5,  # Ethers
    ('C', 'O', 'H', (1, 1)): 104.5,
    
    # sp² Oxygen (bent, ~120°) - carbonyl-like
    ('C', 'O', 'C', (1, 2)): 120.0,
    ('C', 'O', 'H', (2, 1)): 120.0,
    
    # ==================== PHOSPHORUS-CENTERED ANGLES ====================
    
    # sp³ Phosphorus (pyramidal, ~102°)
    ('H', 'P', 'H', (1, 1)): 93.0,   # PH₃ (experimental value)
    ('C', 'P', 'C', (1, 1)): 102.0,
    ('C', 'P', 'H', (1, 1)): 102.0,
    ('C', 'P', 'O', (1, 1)): 102.0,
    ('O', 'P', 'O', (1, 1)): 102.0,
    ('C', 'P', 'F', (1, 1)): 102.0,
    ('C', 'P', 'Cl', (1, 1)): 102.0,
    
    # sp² Phosphorus
    ('C', 'P', 'C', (1, 2)): 120.0,
    ('C', 'P', 'O', (2, 1)): 120.0,
    
    # ==================== SULFUR-CENTERED ANGLES ====================
    
    # sp³ Sulfur (bent/pyramidal, ~92-104°)
    ('H', 'S', 'H', (1, 1)): 92.0,   # H₂S (experimental value)
    ('C', 'S', 'C', (1, 1)): 104.0,  # Thioethers
    ('H', 'S', 'C', (1, 1)): 104.0,
    ('C', 'S', 'H', (1, 1)): 104.0,
    ('C', 'S', 'O', (1, 1)): 104.0,
    ('O', 'S', 'O', (1, 1)): 104.0,
    
    # sp² Sulfur
    ('C', 'S', 'C', (1, 2)): 120.0,
    ('C', 'S', 'O', (2, 1)): 120.0,
    ('O', 'S', 'O', (2, 2)): 120.0,  # SO₂
    ('C', 'C', 'S', (1.0, 2.0)): 120.0,
    ('C', 'C', 'S', (2.0, 1.0)): 120.0,
    # Aromatic Sulfur (thiophene-like)
    ('C', 'S', 'C', (1.5, 1.5)): 92.0,  # Thiophene (5-membered ring strain)
    
    # ==================== HALOGEN-CENTERED ANGLES ====================
    # Note: Halogens typically form only one bond, but in hypervalent compounds:
    
    # Fluorine (rare, mostly in transition state or hypervalent compounds)
    ('C', 'F', 'C', (1, 1)): 180.0,  # Linear when it occurs
    
    # Chlorine, Bromine, Iodine (can be hypervalent)
    ('C', 'Cl', 'C', (1, 1)): 180.0,
    ('C', 'Br', 'C', (1, 1)): 180.0,
    ('C', 'I', 'C', (1, 1)): 180.0,
    ('F', 'Cl', 'F', (1, 1)): 180.0,  # ClF₂⁻ type compounds
    ('F', 'Br', 'F', (1, 1)): 180.0,
    ('F', 'I', 'F', (1, 1)): 180.0,
    
    # ==================== FLIPPED ATOM ORDER ENTRIES (for completeness) ====================
    # These are the same angles as above but with atom1 and atom3 positions swapped
    # and bond orders accordingly flipped: (atom3, central_atom, atom1, (bond_order3, bond_order1))
    
    # CARBON-CENTERED (flipped)
    ('C', 'C', 'H', (1, 1)): 109.5,
    ('N', 'C', 'H', (1, 1)): 109.5,
    ('O', 'C', 'H', (1, 1)): 109.5,
    ('F', 'C', 'H', (1, 1)): 109.5,
    ('Cl', 'C', 'H', (1, 1)): 109.5,
    ('Br', 'C', 'H', (1, 1)): 109.5,
    ('I', 'C', 'H', (1, 1)): 109.5,
    ('P', 'C', 'H', (1, 1)): 109.5,
    ('S', 'C', 'H', (1, 1)): 109.5,
    ('N', 'C', 'C', (1, 1)): 109.5,
    ('O', 'C', 'C', (1, 1)): 109.5,
    ('F', 'C', 'C', (1, 1)): 109.5,
    ('Cl', 'C', 'C', (1, 1)): 109.5,
    ('Br', 'C', 'C', (1, 1)): 109.5,
    ('I', 'C', 'C', (1, 1)): 109.5,
    ('P', 'C', 'C', (1, 1)): 109.5,
    ('S', 'C', 'C', (1, 1)): 109.5,
    ('O', 'C', 'N', (1, 1)): 109.5,
    ('O', 'C', 'O', (1, 1)): 109.5,
    
    # sp² Carbon (flipped)
    ('C', 'C', 'H', (2, 1)): 120.0,
    ('N', 'C', 'H', (2, 1)): 120.0,
    ('O', 'C', 'H', (2, 1)): 120.0,
    ('N', 'C', 'C', (2, 1)): 120.0,
    ('O', 'C', 'C', (2, 1)): 120.0,
    ('O', 'C', 'N', (1, 2)): 120.0,
    ('F', 'C', 'H', (1, 2)): 120.0,
    ('Cl', 'C', 'H', (1, 2)): 120.0,
    ('Br', 'C', 'H', (1, 2)): 120.0,
    ('I', 'C', 'H', (1, 2)): 120.0,
    
    # sp Carbon (flipped)
    ('C', 'C', 'H', (3, 1)): 180.0,
    ('N', 'C', 'H', (3, 1)): 180.0,
    ('N', 'C', 'C', (3, 1)): 180.0,
    
    ('O', 'C', 'N', (2, 1)): 120.0,
    ('N', 'C', 'O', (1, 2)): 120.0,
    # Aromatic Carbon (flipped)
    ('C', 'C', 'H', (1.5, 1)): 120.0,
    ('N', 'C', 'H', (1.5, 1)): 120.0,
    ('N', 'C', 'C', (1.5, 1.5)): 120.0,
    ('O', 'C', 'C', (1, 1.5)): 120.0,
    ('F', 'C', 'C', (1, 1.5)): 120.0,
    ('Cl', 'C', 'C', (1, 1.5)): 120.0,
    ('Br', 'C', 'C', (1, 1.5)): 120.0,
    ('I', 'C', 'C', (1, 1.5)): 120.0,
    
    # NITROGEN-CENTERED (flipped)
    ('C', 'N', 'H', (1, 1)): 107.0,
    ('H', 'N', 'C', (1, 1)): 107.0,
    ('C', 'N', 'C', (2, 1)): 120.0,
    ('H', 'N', 'C', (1, 2)): 120.0,
    ('O', 'N', 'C', (2, 1)): 120.0,
    ('O', 'N', 'C', (1, 2)): 120.0,
    ('N', 'N', 'C', (3, 1)): 180.0,
    ('O', 'N', 'C', (1, 3)): 180.0,
    ('H', 'N', 'C', (1, 1.5)): 120.0,
    
    # OXYGEN-CENTERED (flipped)
    ('C', 'O', 'H', (1, 1)): 104.5,
    ('C', 'O', 'C', (2, 1)): 120.0,
    ('H', 'O', 'C', (1, 2)): 120.0,
    
    # PHOSPHORUS-CENTERED (flipped)
    ('C', 'P', 'H', (1, 1)): 102.0,
    ('O', 'P', 'C', (1, 1)): 102.0,
    ('F', 'P', 'C', (1, 1)): 102.0,
    ('Cl', 'P', 'C', (1, 1)): 102.0,
    ('C', 'P', 'C', (2, 1)): 120.0,
    ('O', 'P', 'C', (1, 2)): 120.0,
    
    # SULFUR-CENTERED (flipped)
    ('C', 'S', 'H', (1, 1)): 104.0,
    ('H', 'S', 'C', (1, 1)): 104.0,
    ('O', 'S', 'C', (1, 1)): 104.0,
    ('C', 'S', 'C', (2, 1)): 120.0,
    ('O', 'S', 'C', (1, 2)): 120.0,
    ('S', 'C', 'C', (1.0, 2.0)): 120.0,

    # HALOGEN-CENTERED (flipped)
    ('C', 'F', 'C', (1, 1)): 180.0,
    ('C', 'Cl', 'C', (1, 1)): 180.0,
    ('C', 'Br', 'C', (1, 1)): 180.0,
    ('C', 'I', 'C', (1, 1)): 180.0,
    ('F', 'Cl', 'F', (1, 1)): 180.0,
    ('F', 'Br', 'F', (1, 1)): 180.0,
    ('F', 'I', 'F', (1, 1)): 180.0,
    # Note: The above ring strain entries use a different key format
    # You may want to handle these separately in your code
}
def get_K_theta(atom1,atom2,atom3,bond_order1,bond_order2):
    theta123_key=(atom1,atom2,atom3,(bond_order1,bond_order2))
    try:
        ref_angle_degree=REFERENCE_ANGLE_BOND_ORDER[theta123_key]
    except:
        ref_angle_degree=120.0
    ref_angle_rad=ref_angle_degree/360*2*np.pi
    REFERENCE_lnK_BOND[frozenset({atom1, atom2})]
    r_ij_eq_A=REFERENCE_R0_BOND_A[frozenset({atom1, atom2})]
    r_jk_eq_A=REFERENCE_R0_BOND_A[frozenset({atom2, atom3})]
    r_ij_p_r_jk_A=r_ij_eq_A+r_jk_eq_A
    D=(r_ij_eq_A-r_jk_eq_A)**2/r_ij_p_r_jk_A**2
    K_theta_ijk_kcal_mol_rad2=143.9*Z_Values[atom1]*C_Values[atom2]*Z_Values[atom3]*ref_angle_rad**(-2)*r_ij_p_r_jk_A**(-1)*np.exp(-2*D)
    #transform kcal/mol into Hartree
    K_theta_ijk_Ha_rad2=0.0015936*K_theta_ijk_kcal_mol_rad2
    K_theta_ijk_Ha_rad2_bohr2=K_theta_ijk_Ha_rad2
    return K_theta_ijk_Ha_rad2_bohr2

REFERENCE_DIHEDRAL_BO = {
    #unit kcal/mol, see W. Jorgensenet. al; J. Am. Chem. Soc. 1996, 118, 11225-11236
    # ==================== ALKANES (C-C single bonds) ====================
    (('H','C','C','H'), (1, 1, 1)): (0.000, 0.000, 0.318),
    (('H','C','C','C'), (1, 1, 1)): (0.000, 0.000, 0.366),
    (('C','C','C','C'), (1, 1, 1)): (1.740, -0.157, 0.279),
    (('C','C','C','H'), (1, 1, 1)): (0.000, 0.000, 0.366),  # Flipped
    
    # ==================== ALKENES (C=C double bonds) ====================
    (('H','C','C','C'), (1, 2, 1)): (0.000, 0.000, -0.372),
    (('C','C','C','H'), (1, 2, 1)): (0.000, 0.000, -0.372),  # Flipped
    (('H','C','C','H'), (1, 2, 1)): (0.000, 0.000, 0.000),   # Ethene-like
    (('C','C','C','C'), (1, 2, 1)): (0.000, 0.000, 0.000),   # Trans/cis barriers
    
    # ==================== AROMATIC SYSTEMS ====================
    # Ethylbenzene and aromatic substitution
    (('H','C','C','C'), (1, 1.5, 1.5)): (0.000, 0.000, 0.000),
    (('C','C','C','C'), (1, 1, 1.5)): (0.000, 0.000, 0.000),
    (('C','C','C','H'), (1.5, 1, 1)): (0.000, 0.000, 0.000),  # Flipped
    (('H','C','C','C'), (1, 1, 1.5)): (0.000, 0.000, 0.462),
    (('C','C','C','H'), (1.5, 1, 1)): (0.000, 0.000, 0.462),  # Flipped
    (('C','C','C','C'), (1.5, 1.5, 1.5)): (0.000, 0.000, 0.000),  # Aromatic-aromatic
    (('H','C','C','H'), (1, 1.5, 1)): (0.000, 0.000, 0.000),      # Aromatic H-H
    
    # ==================== ALCOHOLS (-OH groups) ====================
    (('H','C','O','H'), (1, 1, 1)): (0.000, 0.000, 0.450),
    (('H','O','C','H'), (1, 1, 1)): (0.000, 0.000, 0.450),  # Flipped
    (('C','C','O','H'), (1, 1, 1)): (-0.356, -0.174, 0.492),
    (('H','O','C','C'), (1, 1, 1)): (-0.356, -0.174, 0.492),  # Flipped
    (('H','C','C','O'), (1, 1, 1)): (0.000, 0.000, 0.468),
    (('O','C','C','H'), (1, 1, 1)): (0.000, 0.000, 0.468),  # Flipped
    (('C','C','C','O'), (1, 1, 1)): (1.711, -0.500, 0.663),
    (('O','C','C','C'), (1, 1, 1)): (1.711, -0.500, 0.663),  # Flipped
    
    # Polyol patterns
    (('C','C','O','C'), (1, 1, 1)): (-0.356, -0.174, 0.492),  # C-C-O-C
    (('O','C','O','H'), (1, 1, 1)): (0.000, 0.000, 0.900),    # Diol interactions
    (('H','O','C','O'), (1, 1, 1)): (0.000, 0.000, 0.900),    # Flipped
    
    # ==================== PHENOLS (aromatic -OH) ====================
    (('H','O','C','C'), (1, 1, 1.5)): (0.000, 1.682, 0.000),
    (('C','C','O','H'), (1.5, 1, 1)): (0.000, 1.682, 0.000),  # Flipped
    (('C','O','C','C'), (1, 1.5, 1.5)): (0.000, 1.682, 0.000), # O-aromatic-aromatic
    
    # ==================== THIOLS (-SH groups) ====================
    (('H','C','S','H'), (1, 1, 1)): (0.000, 0.000, 0.451),
    (('H','S','C','H'), (1, 1, 1)): (0.000, 0.000, 0.451),  # Flipped
    (('C','C','S','H'), (1, 1, 1)): (-0.759, -0.282, 0.603),
    (('H','S','C','C'), (1, 1, 1)): (-0.759, -0.282, 0.603),  # Flipped
    (('H','C','C','S'), (1, 1, 1)): (0.000, 0.000, 0.452),
    (('S','C','C','H'), (1, 1, 1)): (0.000, 0.000, 0.452),  # Flipped
    (('C','C','C','S'), (1, 1, 1)): (1.876, 0.000, 0.000),
    (('S','C','C','C'), (1, 1, 1)): (1.876, 0.000, 0.000),  # Flipped
    
    # ==================== SULFIDES (C-S-C) ====================
    (('H','C','S','C'), (1, 1, 1)): (0.000, 0.000, 0.647),
    (('C','S','C','H'), (1, 1, 1)): (0.000, 0.000, 0.647),  # Flipped
    (('C','C','C','S'), (1, 1, 1)): (2.619, -0.620, 0.258),
    (('S','C','C','C'), (1, 1, 1)): (2.619, -0.620, 0.258),  # Flipped
    (('C','C','S','C'), (1, 1, 1)): (0.925, -0.576, 0.677),
    (('C','S','C','C'), (1, 1, 1)): (0.925, -0.576, 0.677),  # Flipped
    
    # Aromatic sulfides
    (('C','C','S','C'), (1.5, 1, 1)): (0.925, -0.576, 0.677),
    (('C','S','C','C'), (1, 1, 1.5)): (0.925, -0.576, 0.677),
    
    # ==================== DISULFIDES (C-S-S-C) ====================
    (('C','S','S','C'), (1, 1, 1)): (0.000, -7.414, 1.705),
    (('H','C','S','S'), (1, 1, 1)): (0.000, 0.000, 0.558),
    (('S','S','C','H'), (1, 1, 1)): (0.000, 0.000, 0.558),  # Flipped
    (('C','C','S','S'), (1, 1, 1)): (1.941, -0.836, 0.935),
    (('S','S','C','C'), (1, 1, 1)): (1.941, -0.836, 0.935),  # Flipped
    
    # Cyclic disulfides (like cysteine bridges)
    (('S','C','C','S'), (1, 1, 1)): (0.000, -7.414, 1.705),
    
    # ==================== PRIMARY AMINES (1° -NH2) ====================
    (('H','C','N','H'), (1, 1, 1)): (0.000, 0.000, 0.400),
    (('H','N','C','H'), (1, 1, 1)): (0.000, 0.000, 0.400),  # Flipped
    (('H','C','C','N'), (1, 1, 1)): (-1.013, -0.709, 0.473),
    (('N','C','C','H'), (1, 1, 1)): (-1.013, -0.709, 0.473),  # Flipped
    (('C','C','N','H'), (1, 1, 1)): (-0.190, -0.417, 0.418),
    (('H','N','C','C'), (1, 1, 1)): (-0.190, -0.417, 0.418),  # Flipped
    (('C','C','C','N'), (1, 1, 1)): (2.392, -0.674, 0.550),
    (('N','C','C','C'), (1, 1, 1)): (2.392, -0.674, 0.550),  # Flipped
    
    # ==================== SECONDARY AMINES (2° -NH-) ====================
    (('C','C','N','C'), (1, 1, 1)): (-0.190, -0.417, 0.418),
    (('C','N','C','C'), (1, 1, 1)): (-0.190, -0.417, 0.418),  # Flipped
    (('H','C','N','C'), (1, 1, 1)): (0.000, 0.000, 0.400),
    (('C','N','C','H'), (1, 1, 1)): (0.000, 0.000, 0.400),   # Flipped
    
    # ==================== TERTIARY AMINES (3° -N<) ====================
    (('C','C','N','C'), (1, 1, 1)): (-0.190, -0.417, 0.418),
    (('C','N','C','C'), (1, 1, 1)): (-0.190, -0.417, 0.418),
    
    # ==================== ETHERS (C-O-C) ====================
    (('H','C','O','C'), (1, 1, 1)): (0.000, 0.000, 0.760),
    (('C','O','C','H'), (1, 1, 1)): (0.000, 0.000, 0.760),  # Flipped
    (('C','C','O','C'), (1, 1, 1)): (0.650, -0.250, 0.670),
    (('C','O','C','C'), (1, 1, 1)): (0.650, -0.250, 0.670),  # Flipped
    
    # Aromatic ethers
    (('C','C','O','C'), (1.5, 1, 1)): (0.650, -0.250, 0.670),
    (('C','O','C','C'), (1, 1, 1.5)): (0.650, -0.250, 0.670),
    
    # ==================== ACETALS/KETALS (C-O-C-O) ====================
    (('C','O','C','O'), (1, 1, 1)): (-0.574, -0.997, 0.0),
    (('O','C','O','C'), (1, 1, 1)): (-0.574, -0.997, 0.0),   # Flipped
    (('H','O','C','O'), (1, 1, 1)): (-0.574, -0.997, 0.0),
    (('O','C','O','H'), (1, 1, 1)): (-0.574, -0.997, 0.0),   # Flipped
    
    # ==================== CARBOXYLIC ACIDS (-COOH) ====================
    # Note: There was a duplicate key in original, assuming this is the correct one
    (('O','C','O','H'), (1, 1, 1)): (0.000, 4.830, 0.000),  # Carboxylic acid
    (('H','O','C','O'), (1, 1, 1)): (0.000, 4.830, 0.000),  # Flipped
    (('C','C','C','O'), (1, 1, 2)): (0.000, 0.000, 0.000),  # C=O rotation
    (('O','C','C','C'), (2, 1, 1)): (0.000, 0.000, 0.000),  # Flipped
    
    # ==================== ESTERS (-COO-) ====================
    (('C','C','O','C'), (1, 2, 1)): (0.000, 5.400, 0.000),  # Ester linkage
    (('C','O','C','C'), (1, 2, 1)): (0.000, 5.400, 0.000),  # Flipped
    (('O','C','O','C'), (2, 1, 1)): (0.000, 5.400, 0.000),  # C=O-O-C
    (('C','O','C','O'), (1, 1, 2)): (0.000, 5.400, 0.000),  # Flipped
    
    # ==================== AMIDES (-CONH-) ====================
    (('C','C','N','H'), (1, 2, 1)): (0.000, 10.000, 0.000), # Amide rotation barrier
    (('H','N','C','C'), (1, 2, 1)): (0.000, 10.000, 0.000), # Flipped
    (('C','C','N','C'), (1, 2, 1)): (0.000, 10.000, 0.000), # N-alkyl amides
    (('C','N','C','C'), (1, 2, 1)): (0.000, 10.000, 0.000), # Flipped
    (('O','C','N','H'), (2, 1, 1)): (0.000, 10.000, 0.000), # C=O-N-H
    (('H','N','C','O'), (1, 1, 2)): (0.000, 10.000, 0.000), # Flipped
    
    # ==================== IMINES (C=N) ====================
    (('C','C','N','C'), (1, 2, 1)): (0.000, 0.000, 0.000),  # C-C=N-C
    (('C','N','C','C'), (1, 2, 1)): (0.000, 0.000, 0.000),  # Flipped
    (('H','C','N','C'), (1, 2, 1)): (0.000, 0.000, 0.000),
    (('C','N','C','H'), (1, 2, 1)): (0.000, 0.000, 0.000),  # Flipped
    
    # ==================== PHOSPHORUS COMPOUNDS ====================
    (('C','C','P','C'), (1, 1, 1)): (0.000, 0.000, 0.200),  # Phosphines
    (('C','P','C','C'), (1, 1, 1)): (0.000, 0.000, 0.200),  # Flipped
    (('H','C','P','C'), (1, 1, 1)): (0.000, 0.000, 0.200),
    (('C','P','C','H'), (1, 1, 1)): (0.000, 0.000, 0.200),  # Flipped
    (('O','P','O','C'), (1, 1, 1)): (0.000, 0.000, 0.300),  # Phosphates
    (('C','O','P','O'), (1, 1, 1)): (0.000, 0.000, 0.300),  # Flipped
    
    # ==================== NITRILES (-CN) ====================
    (('C','C','C','N'), (1, 1, 3)): (0.000, 0.000, 0.000),  # C-C-C≡N
    (('N','C','C','C'), (3, 1, 1)): (0.000, 0.000, 0.000),  # Flipped
    (('H','C','C','N'), (1, 1, 3)): (0.000, 0.000, 0.000),
    (('N','C','C','H'), (3, 1, 1)): (0.000, 0.000, 0.000),  # Flipped
    
    # ==================== NITRO GROUPS (-NO2) ====================
    (('C','C','N','O'), (1, 1, 2)): (0.000, 0.000, 0.000),  # C-C-N=O
    (('O','N','C','C'), (2, 1, 1)): (0.000, 0.000, 0.000),  # Flipped
    (('O','N','C','H'), (2, 1, 1)): (0.000, 0.000, 0.000),
    (('H','C','N','O'), (1, 1, 2)): (0.000, 0.000, 0.000),  # Flipped
    
    # ==================== SPECIAL RING SYSTEMS ====================
    # 5-membered rings (furan, pyrrole, thiophene)
    (('C','C','O','C'), (1.5, 1, 1.5)): (0.000, 0.000, 0.000),  # Furan
    (('C','O','C','C'), (1.5, 1, 1.5)): (0.000, 0.000, 0.000),
    (('C','C','N','C'), (1.5, 1, 1.5)): (0.000, 0.000, 0.000),  # Pyrrole
    (('C','N','C','C'), (1.5, 1, 1.5)): (0.000, 0.000, 0.000),
    (('C','C','S','C'), (1.5, 1, 1.5)): (0.000, 0.000, 0.000),  # Thiophene
    (('C','S','C','C'), (1.5, 1, 1.5)): (0.000, 0.000, 0.000),
    
    # 6-membered rings (pyridine, pyrimidine)
    (('C','C','N','C'), (1.5, 1.5, 1.5)): (0.000, 0.000, 0.000),  # Pyridine
    (('N','C','C','N'), (1.5, 1.5, 1.5)): (0.000, 0.000, 0.000),  # Pyrimidine
    
    # ==================== HALOGENS ====================
    (('H','C','C','F'), (1, 1, 1)): (0.000, 0.000, 0.460),
    (('F','C','C','H'), (1, 1, 1)): (0.000, 0.000, 0.460),  # Flipped
    (('H','C','C','Cl'), (1, 1, 1)): (0.000, 0.000, 0.355),
    (('Cl','C','C','H'), (1, 1, 1)): (0.000, 0.000, 0.355),  # Flipped
    (('H','C','C','Br'), (1, 1, 1)): (0.000, 0.000, 0.340),
    (('Br','C','C','H'), (1, 1, 1)): (0.000, 0.000, 0.340),  # Flipped
    (('H','C','C','I'), (1, 1, 1)): (0.000, 0.000, 0.320),
    (('I','C','C','H'), (1, 1, 1)): (0.000, 0.000, 0.320),   # Flipped
    
    (('C','C','C','F'), (1, 1, 1)): (0.000, 0.000, 0.460),
    (('F','C','C','C'), (1, 1, 1)): (0.000, 0.000, 0.460),   # Flipped
    (('C','C','C','Cl'), (1, 1, 1)): (0.000, 0.000, 0.355),
    (('Cl','C','C','C'), (1, 1, 1)): (0.000, 0.000, 0.355),  # Flipped
    (('C','C','C','Br'), (1, 1, 1)): (0.000, 0.000, 0.340),
    (('Br','C','C','C'), (1, 1, 1)): (0.000, 0.000, 0.340),  # Flipped
    (('C','C','C','I'), (1, 1, 1)): (0.000, 0.000, 0.320),
    (('I','C','C','C'), (1, 1, 1)): (0.000, 0.000, 0.320),   # Flipped
}

# Helper function to get dihedral parameters with automatic flipping
def get_dihedral_params(atom1, atom2, atom3, atom4, bo1, bo2, bo3):
    """
    Get dihedral parameters with automatic handling of atom order.
    Returns (V1, V2, V3) Fourier coefficients.
    """
    key1 = ((atom1, atom2, atom3, atom4), (bo1, bo2, bo3))
    key2 = ((atom4, atom3, atom2, atom1), (bo3, bo2, bo1))  # Flipped
    
    if key1 in REFERENCE_DIHEDRAL_BO:
        return REFERENCE_DIHEDRAL_BO[key1]
    elif key2 in REFERENCE_DIHEDRAL_BO:
        return REFERENCE_DIHEDRAL_BO[key2]
    else:
        # Return default values if not found
        return (0.000, 0.000, 0.000)

def get_initial_Hessian(atoms,coordinates_bohr,bonds,angles,dihedrals,s=1):

    bond_order=assign_bond_orders(atoms=atoms,bonds=bonds)
    coordinates_bohr=np.array(coordinates_bohr)
    H_int_diag = np.zeros(len(bonds)+len(angles)+len(dihedrals))#+len(impropers))

    for it_bond,bond in enumerate(bonds):
        i_idx=bond[0]
        j_idx=bond[1]
        H_int_diag[it_bond]=get_K_bond(atoms[i_idx], atoms[j_idx], coordinates_bohr[i_idx],coordinates_bohr[j_idx])

    for it_angle,angle in enumerate(angles):
        i_idx=angle[0]
        j_idx=angle[1]
        k_idx=angle[2]
        bond_order1=bond_order[tuple(sorted((i_idx,j_idx)))]
        bond_order2=bond_order[tuple(sorted((j_idx,k_idx)))]
        H_int_diag[len(bonds)+it_angle]=get_K_theta(atoms[i_idx],atoms[j_idx],atoms[k_idx],bond_order1,bond_order2)
    for it_torsion,torsion in enumerate(dihedrals):
        i_idx=torsion[0]
        j_idx=torsion[1]
        k_idx=torsion[2]
        l_idx=torsion[3]
        bond_order1=bond_order[tuple(sorted((i_idx,j_idx)))]
        bond_order2=bond_order[tuple(sorted((j_idx,k_idx)))]
        bond_order3=bond_order[tuple(sorted((k_idx,l_idx)))]
        V1,V2,V3=get_dihedral_params(atoms[i_idx],atoms[j_idx],atoms[k_idx], atoms[l_idx], bond_order1, bond_order2, bond_order3)
        #kcals/mol -> Hartree
        V1*=0.0015936
        V2*=0.0015936
        V3*=0.0015936
        
        r_ij=coordinates_bohr[i_idx]-coordinates_bohr[j_idx]
        r_kj=coordinates_bohr[k_idx]-coordinates_bohr[j_idx]
        r_lk=coordinates_bohr[l_idx]-coordinates_bohr[k_idx]
        r_u = np.linalg.norm(r_ij)
        r_w = np.linalg.norm(r_kj)
        r_v = np.linalg.norm(r_lk)

        #scale_factor=1.0
        u = r_ij / r_u
        w = r_kj / r_w
        v = r_lk / r_v

        # Normals to the m-o-p and o-p-n planes
        normal_mop = np.cross(u, w)
        normal_opn = np.cross(v, -w) # Use -w to point away from central bond
        
        norm_mop_mag = np.linalg.norm(normal_mop)
        norm_opn_mag = np.linalg.norm(normal_opn)

        # Check for collinear atoms (which make planes undefined)
        if norm_mop_mag < 1e-8 or norm_opn_mag < 1e-8:
             H_int_diag[len(bonds)+len(angles)+it_torsion]=0
             continue

        # Calculate the dihedral angle using a stable atan2 method
        # The axis of rotation is the central bond vector w
        y = np.dot(np.cross(normal_mop, normal_opn), w)
        x = np.dot(normal_mop, normal_opn)
        phi = np.arctan2(y, x)
        Hphi1phi2=-0.5*V1*np.cos(phi)+2*V2*np.cos(2*phi)-4.5*V3*np.cos(3*phi)
        H_int_diag[len(bonds)+len(angles)+it_torsion]=Hphi1phi2

    # --- Regularization ---
    # Make all entries positive and avoid too small values
    min_threshold = max(np.min(H_int_diag[H_int_diag > 0.005]), 1e-3)
    H_int_diag = np.abs(H_int_diag) + min_threshold
    
    # Optional scaling factor
    H_int_diag *= s
    return H_int_diag*s
    


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
    cell = config["cell_A"]
    print(cell)
    atoms=config["atoms"]
    coordinates=config["coordinates_A"]
    cell_files= [f for f in os.listdir(path) if f.endswith('_tmp.cell')]
    if len(cell_files) == 0:
        if np.linalg.matrix_rank(cell)==0:
            padding = 10 # Define a clear padding value (e.g., 10 Å on each side)
            # 1. Center the molecule at the origin.
            centered_coords, _ = get_principle_axis_coordinates(coordinates)
            centered_coords = np.array(centered_coords)

            # 2. Calculate the full extent of the molecule in each dimension.
            #    This is 2 * the maximum absolute coordinate since it's centered.
            mol_extent_x = 2 * np.max(np.abs(centered_coords[:, 0]))
            mol_extent_y = 2 * np.max(np.abs(centered_coords[:, 1]))
            mol_extent_z = 2 * np.max(np.abs(centered_coords[:, 2]))
            mol_extent=np.max([mol_extent_x,mol_extent_y,mol_extent_z])
            # 3. Define cell vectors as the molecular extent plus padding.
            cell = np.array([
                [mol_extent + padding, 0, 0],
                [0, mol_extent + padding, 0],
                [0, 0, mol_extent + padding]
            ])

            write_cell(path, name, cell)

            # 4. Shift the centered coordinates to the center of the new box. This part is correct.
            cell_center = 0.5 * np.array([cell[0][0], cell[1][1], cell[2][2]])
            coordinates = centered_coords + cell_center
            config["periodic"]="None"
        elif np.linalg.matrix_rank(cell)==1:
            _, R = np.linalg.qr(cell)
            independent_index = np.where(np.abs(np.diag(R)) > 1e-10)
            independent_vectors=cell[independent_index[0],:]
            independent_vector=independent_vectors[0]
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

            # If points go outside [0, cell_length_x], adjust cell length
            for it in range(len(rotated_coords[:, 0])):
                if rotated_coords[it, 0] <0:
                    rotated_coords[it, 0]+=cell_length_x
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
            padding = 20
            
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
            config["periodic"]="x"
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
    config["coordinates_A"]=coordinates
    config["cell_A"]=cell
    return config
def run_energy_force_eval( path: str, cp2k_exec: str = "./cp2k.popt", input_file: str = "input_file.inp", output_file: str = "output_file.out", runner: str = "local" ) -> Tuple[float, List[List[float]]]: 
    """ Runs a CP2K calculation for given coordinates and atoms. 
        If config_string is provided: 
            1. Updates the XYZ file in path 
            2. Generates the CP2K input file from config_string 
            Otherwise: Assumes input_file.inp already exists in path. 
        Then: 
            3. Runs CP2K 4. Reads forces and energy 

        Parameters ---------- 
        path : str Working directory. 
        name : str Base name for XYZ file. 
        coordinates : list of list of float Atomic coordinates. 
        atoms : list of str Atom symbols. 
        config_string : str or None CP2K config string (optional). 
        cp2k_exec : str Path to CP2K executable. 
        input_file : str CP2K input file name. 
        output_file : str CP2K output file name. 
        np : int Number of MPI processes. 
        mpirun_cmd : str MPI command. 

        Returns ------- 
        E0 : float Ground state energy. 
        forces : list of list of float Atomic forces. 
    """ 
    if runner == "local": 
        cmd = ["mpirun", "-np", str(9), cp2k_exec, "-i", input_file, "-o", output_file] 
    elif runner == "slurm": # In SLURM allocation, srun handles MPI ranks automatically 
        cmd = ["srun", cp2k_exec, "-i", input_file, "-o", output_file] 
    else: 
        raise ValueError("runner must be 'local' or 'slurm'")

    output_path = os.path.join(path, output_file)
    with open(output_path, "w") as out_f:
        process = subprocess.run(cmd,
                                   cwd=path,
                                   capture_output=True,
                                   text=True )
    # --- 2. New: Append output to a log file ---
    log_file_path = os.path.join(path, "calculation.log")
    try:
        # Open the log file in append mode ('a')
        with open(log_file_path, "a") as log_f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_f.write(f"\n{'='*20} LOG ENTRY: {timestamp} {'='*20}\n")
            log_f.write(f"Command: {' '.join(cmd)}\n")
            log_f.write(f"Return Code: {process.returncode}\n\n")
            
            log_f.write("--- STDOUT ---\n")
            log_f.write(process.stdout)
            
            if process.stderr:
                log_f.write("\n--- STDERR ---\n")
                log_f.write(process.stderr)
            
            log_f.write(f"\n{'='*20} END OF LOG ENTRY {'='*20}\n")
    except IOError as e:
        print(f"Warning: Could not write to log file {log_file_path}. Error: {e}")

    # --- 3. Check for errors ---
    # The original output is still printed to the main SLURM output for convenience
    # --- 3b. Append to separate log files ---
    stdout_log = os.path.join(path, "std_out.log")
    stderr_log = os.path.join(path, "std_err.log")

    try:
        with open(stdout_log, "a") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"\n{'='*20} STDOUT: {timestamp} {'='*20}\n")
            f.write(process.stdout)
            f.write(f"\n{'='*20} END STDOUT {'='*20}\n")
    except IOError as e:
        print(f"Warning: Could not write to {stdout_log}. Error: {e}")

    if process.stderr:
        try:
            with open(stderr_log, "a") as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"\n{'='*20} STDERR: {timestamp} {'='*20}\n")
                f.write(process.stderr)
                f.write(f"\n{'='*20} END STDERR {'='*20}\n")
        except IOError as e:
            print(f"Warning: Could not write to {stderr_log}. Error: {e}")
        
    if process.returncode != 0:
        # Improved error message points to the log for detailed debugging
        raise RuntimeError(
            f"CP2K failed with return code {process.returncode}. "
            f"Check the detailed output in {log_file_path}"
        )

    # --- 4. Extract results as before ---
    forces = Read.read_forces(folder=path)
    E0 = Read.read_total_energy(path=path,verbose=False)

    return E0, forces
def run_energy_force_stress_eval(
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
        cmd = ["mpirun", "-np", str(9), cp2k_exec, "-i", input_file, "-o", output_file]
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
    stresses = Read.read_stress(folder=path)
    stresses*=2.94210e-5#conversion GPa -> Ha/a_0**3
    forces = Read.read_forces(folder=path)
    E0 = Read.read_total_energy(path=path,verbose=False)
    return E0, forces, stresses
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
        "cell_A":np.zeros((3,3)),
        "atoms":None,
        "coordinates_A":None,
        "masses":None
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
            if cell_data.endswith(";"):
                cell_data = cell_data[:-1]
            try:
                cell = np.array(ast.literal_eval(cell_data), dtype=float)
            except Exception as e:
                raise ValueError(f"Could not parse cell string: {cell_data}") from e
            result["cell_A"] = cell
    coordinates=[]
    atoms=[]
    for line in config[2:]:
        atoms.append(line.split()[0])
        coordinates.append(np.array([float(line.split()[1]),float(line.split()[2]),float(line.split()[3])]))
    result["atoms"]=atoms
    masses=[]
    for atom in atoms:
        masses.append(StandardAtomicWeights[atom])
    result["masses"]=masses
    result["coordinates_A"]=coordinates
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
    print(config["periodic"])
    if config["periodic"]=="None":
        string += dft_basis_set_section(basis_set_file=config["basis_file"], potential_file=config['potential_file'])
    else:
        string += dft_basis_set_section(basis_set_file=config["basis_file"], potential_file=config['potential_file'],stress_tensor="ANALYTICAL")
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
    cell=config["cell_A"]
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
        eps_schwarz=1e-12,
        max_memory=30000,
        eps_storage_scaling=1e-1
    )

    periodic = "XYZ"
    cell=config["cell_A"]
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
    elif run_mode == "cell":
        RUN_TYPE = "ENERGY_FORCE"
    else:
        raise ValueError(f"Unsupported run_mode: '{run_mode}'. Choose from 'energy', 'va', 'force', 'cell', or 'geo'.")

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
        cutoff=600; rel_cutoff=80; ngrids=6
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
    density_cutoff=1e-10,
    gradient_cutoff=1e-10,
    tau_cutoff=1e-10,
    eps_schwarz=1e-9,
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
    minimizer="DIIS",
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
def poisson_section(periodic="NONE", psolver="WAVELET"):
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

