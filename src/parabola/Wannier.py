import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from . import TDDFT, Geometry, Read, Write, electronics
from .PhysConst import ConversionFactors


def atoms_within_unit_cell(path="./"):
    # Parabola (Symmetry part) has issues processing the atoms that are outside the unit cell.
    # This function maps/'wraps' the atoms that are outside to their periodic copies that fall within the unit cell

    from ase.io import read, write

    xyzfile_name = Read.get_xyz_filename(path)
    xyzfile = read(xyzfile_name)
    xyzfile.set_pbc([True, True, True])
    xyzfile.wrap()
    write(xyzfile_name[:-4] + "_wrapped.xyz", images=xyzfile, format="extxyz")

    return None


def determine_k_point(mol, bloch_states, threshold=10**-15):
    # Can also be used to check if the bloch states have been made correctly!
    olm = mol.Electronics.OLM
    periodic_bool = np.array(mol.periodicity) > 1

    eigs = []
    for d in [0, 1, 2]:
        if periodic_bool[d]:
            T_op = mol.Electronics.electronic_symmetry.Symmetry_Generators["t" + str(d + 1)]
            eig = np.conjugate(bloch_states) @ T_op @ bloch_states.T
            eig_diag = np.diagonal(eig)
            eigs.append(eig_diag)

            if np.max(np.abs(eig - np.diag(eig_diag))) > threshold:
                print("Bloch states have not been correctly made!!")
                plt.matshow(np.abs(eig - np.diag(eig_diag)))
                plt.colorbar()
                plt.show()
        else:
            eigs.append((1.0 + 0.0j) * np.ones(bloch_states.shape[0]))

    full_phase = [[eigs[0][i], eigs[1][i], eigs[2][i]] for i in range(bloch_states.shape[0])]
    bloch_k_points = []
    for eigval in full_phase:
        k = []
        for direc in eigval:
            k.append(np.round((math.atan2(direc.imag, direc.real) / (2 * np.pi)), 6))
        k = np.array(k)
        k[np.isclose(k, 0.5, 1e-6)] = -0.5
        k[np.isclose(k, 0.0, 1e-6)] = 0.0
        print(k)
        bloch_k_points.append(k)

    k_resolved_bloch_dist = {}
    for state_ind in range(bloch_states.shape[0]):
        state = bloch_states[:, state_ind]
        key = tuple(bloch_k_points[state_ind])
        if key not in k_resolved_bloch_dist:
            k_resolved_bloch_dist[key] = []
        k_resolved_bloch_dist[key].append(state)

    return k_resolved_bloch_dist


def commute_check(mol, path="./"):
    # Calculating the symmetry-adapted KSH
    symsecs = list(mol.Electronics.electronic_symmetry.SymSectors.keys())
    sym_states, sym_state_ene = [], []
    for sym_sector in symsecs:
        for state in mol.Electronics.real_eigenstates["alpha"][sym_sector]:
            sym_states.append(state.a)
            sym_state_ene.append(state.energy)
    sym_states = np.array(sym_states)
    sym_state_ene = np.array(sym_state_ene)

    # ksh_orth = sym_states.T @ np.diag(sym_state_ene) @ sym_states
    olm = mol.Electronics.OLM
    # olmp12 = sp.fractional_matrix_power(olm,0.5)
    # ksh = olmp12 @ ksh_orth @ olmp12
    # Or use the DFT output ksh = mol.Electronics.KS_Hamiltonian_alpha

    T1 = mol.Electronics.electronic_symmetry.Symmetry_Generators["t1"]
    T2 = mol.Electronics.electronic_symmetry.Symmetry_Generators["t2"]
    # T3 = mol.Electronics.electronic_symmetry.Symmetry_Generators['t3']

    # ks_t1 = ksh @ T1
    # t1_ks = T1 @ ksh
    # print('ks_t1',np.max(np.abs(ks_t1-t1_ks)))

    # ks_t2 = ksh @ T2
    # t2_ks = T2 @ ksh
    # print('ks_t2',np.max(np.abs(ks_t2 - t2_ks)))

    # t1_t2 = T1 @ T2
    # t2_t1 = T2 @ T1
    # print('t1_t2', np.max(np.abs(t1_t2 - t2_t1)))

    trans1 = sym_states @ olm @ T1 @ sym_states.T
    print(np.max(np.imag(trans1)))
    plt.matshow(np.real(trans1))
    plt.colorbar()
    plt.show()

    return None


def filter_matrix(matrix, tolerance):
    Mreal, Mimag = np.real(matrix).copy(), np.imag(matrix).copy()
    mask1 = np.abs(Mreal) < tolerance
    Mreal[mask1] = 0

    mask2 = np.abs(Mimag) < tolerance
    Mimag[mask2] = 0

    return Mreal + (1.0j) * Mimag


def bloch_from_old(mol, path="./"):
    ksh = mol.Electronics.KS_Hamiltonian_alpha
    olm = mol.Electronics.OLM
    olmm1w = mol.Electronics.inverse_sqrt_OLM

    prim_gam_sym, non_prim_gam_sym = separating_symmetries(mol)
    periodic_bool = np.array(mol.periodicity) > 1

    for sym_sector in non_prim_gam_sym:
        print(sym_sector)
        sym_states_a, sym_state_ene = [], []
        for state in mol.Electronics.real_eigenstates["alpha"][sym_sector]:
            sym_states_a.append(state.a)
            sym_state_ene.append(state.energy)
        sym_states_a = np.array(sym_states_a)
        sym_state_ene = np.array(sym_state_ene)

        indices = unfolding.group_equal_indices(sym_state_ene)  # find degenerate states

        print(indices)

    return None


def threshold(M, threshold=1e-5):
    # function to put any real or imaginary part of the elements of Matrix M below a certain threshold to zero

    Mreal, Mimag = np.real(M).copy(), np.imag(M).copy()
    mask1 = np.abs(Mreal) < threshold
    Mreal[mask1] = 0

    mask2 = np.abs(Mimag) < threshold
    Mimag[mask2] = 0

    return Mreal + (1.0j) * Mimag


def testing_bloch(mol, path="./", name_of_file="general"):
    if "Bloch_states_" + name_of_file + ".npy" in os.listdir(path):
        print("Found pre-calculated bloch states!")
        bloch_eigenstates_full = np.load(path + "Bloch_states_" + name_of_file + ".npy")

    else:
        ksh = mol.Electronics.KS_Hamiltonian_alpha
        olm = mol.Electronics.OLM
        olmm12 = mol.Electronics.inverse_sqrt_OLM
        ksh_orth = olmm12 @ ksh @ olmm12
        prim_gam_sym, non_prim_gam_sym = separating_symmetries(mol)
        periodic_bool = np.array(mol.periodicity) > 1

        bloch_eigenstates_full = []
        for sym_sector in non_prim_gam_sym:
            print(sym_sector)
            sym_states_A = mol.Electronics.real_eigenstates["alpha"][sym_sector].T
            sym_states_a = []
            for ind in range(sym_states_A.shape[0]):
                sym_states_a.append(olmm12 @ sym_states_A[ind, :])

            sym_states_a = np.array(sym_states_a)

            periodic_bool = np.array(mol.periodicity) > 1
            full_matrix = sym_states_a @ ksh @ sym_states_a.T
            for d in [0, 1, 2]:
                if periodic_bool[d]:
                    T_op = mol.Electronics.electronic_symmetry.Symmetry_Generators["t" + str(d + 1)]
                    full_matrix += np.random.randn() * sym_states_a @ olm @ (T_op + T_op.T) @ sym_states_a.T
            val, vec = np.linalg.eigh(full_matrix)
            par_bloch_states = sym_states_a.T @ vec
            par_bloch_states = np.array(par_bloch_states).T

            T2 = mol.Electronics.electronic_symmetry.Symmetry_Generators["t2"]
            # T3 = mol.Electronics.electronic_symmetry.Symmetry_Generators["t3"]

            # print('commute - atomic basis: ', np.max(abs( olm @ T1 - T1 @ olm)), np.max(abs( T2 @ T1 - T1 @ T2)) , np.max(abs( ksh @ T2 - T2 @ ksh)))

            # full_matrix = (np.random.randn() * sym_states_a @ ksh_orth @ sym_states_a.T) + (np.random.randn() *sym_states_a @  (T1+T1.T) @ sym_states_a.T) + (np.random.randn() *sym_states_a @ (T2+T2.T) @ sym_states_a.T)
            # val, vec = np.linalg.eigh(full_matrix)
            # print('unitary', val[:4],np.max(np.abs( np.conjugate(vec).T@vec - np.eye(vec.shape[0]) )))
            # par_bloch_states = sym_states_a.T @ vec
            # par_bloch_states = np.array(par_bloch_states).T

            bloch_states = par_bloch_states.copy()
            for d in [0, 1, 2]:
                if periodic_bool[d]:
                    T_op = mol.Electronics.electronic_symmetry.Symmetry_Generators["t" + str(d + 1)]
                    eig = np.conjugate(bloch_states) @ olm @ T_op @ bloch_states.T
                    blocks = electronics.detect_block_sizes(np.abs(eig), tol=7.5e-2)
                    for block in blocks:
                        i, size = block

                        if size == 1:
                            pass
                        else:
                            this_block_bloch = (
                                np.conjugate(par_bloch_states[i : i + size + 1, :])
                                @ olm
                                @ T_op
                                @ par_bloch_states[i : i + size + 1, :].T
                            )
                            val, vec = np.linalg.eig(this_block_bloch)
                            if np.max(np.abs(np.conjugate(vec).T @ vec - np.eye(vec.shape[0]))) > 1e-5:
                                print(
                                    "bloch transformation not unitary",
                                    np.max(np.abs(np.conjugate(vec).T @ vec - np.eye(vec.shape[0]))),
                                )
                            this_block_bloch_states = par_bloch_states[i : i + size + 1, :].T @ vec
                            bloch_states[i : i + size + 1, :] = this_block_bloch_states.T
            for i in range(bloch_states.shape[0]):
                bloch_eigenstates_full.append(bloch_states[i, :])
            eig = np.conjugate(bloch_states) @ olm @ T2 @ bloch_states.T
            # eigval = np.real(np.conjugate(bloch_states) @ ksh @ bloch_states.T)
            eig_diag = np.diagonal(eig)
            if np.max(np.abs(eig - np.diag(eig_diag))) > 1e-4:
                print("Bloch states have not been correctly made!!")
                plt.matshow(np.abs(eig - np.diag(eig_diag)))
                #    plt.matshow(eigval)
                plt.colorbar()
                # plt.show()
                plt.savefig(path + str(sym_sector) + "bloch_issues.png", dpi=1200)

        bloch_eigenstates_full = np.array(bloch_eigenstates_full)
        # TODO: uncomment to save after debugging
        # np.save(path + "Bloch_states_" + name_of_file + ".npy", bloch_eigenstates_full)

    return bloch_eigenstates_full


def bloch_form(mol, path="./", name_of_file="general"):
    if "Bloch_states_" + name_of_file + ".npy" in os.listdir(path):
        print("Found pre-calculated bloch states!")
        bloch_eigenstates = np.load(path + "Bloch_states_" + name_of_file + ".npy")
        return bloch_eigenstates
    else:
        """
        ksh = mol.Electronics.KS_Hamiltonian_alpha
        olm = mol.Electronics.OLM
        prim_gam_sym, non_prim_gam_sym = separating_symmetries(mol)
        periodic_bool = np.array(mol.periodicity) > 1

        bloch_eigenstates = []
        sym_sector = non_prim_gam_sym[0]
        sym_states = []
        for state in mol.Electronics.real_eigenstates["alpha"][sym_sector]:
            sym_states.append(state.A)
        sym_states = np.array(sym_states)
        periodic_bool = np.array(mol.periodicity) > 1
        T1 = mol.Electronics.electronic_symmetry.Symmetry_Generators['t1']
        T2 = mol.Electronics.electronic_symmetry.Symmetry_Generators['t2']

        # commuting check
        ksh_in_symbasis = sym_states @ ksh @ sym_states.T
        t1_symbasis = sym_states @ T1 @ sym_states.T
        t2_symbasis = sym_states @ T2 @ sym_states.T
        print('ks_t1', np.max(np.abs(ksh_in_symbasis @ t2_symbasis - t2_symbasis @ ksh_in_symbasis)))
        print('t2_t1', np.max(np.abs(t1_symbasis @ t2_symbasis - t2_symbasis @ t1_symbasis)))


        full = ksh_in_symbasis + t1_symbasis + t2_symbasis

        val, vec = np.linalg.eig(full)

        print('check: ', np.max(abs( full - (vec @ np.diag(val) @ np.linalg.inv(vec) ))))
        bloch_states = sym_states.T @ vec

        eig = np.conjugate(bloch_states).T @ T1 @ bloch_states
        eig_diag = np.diagonal(eig)

        plt.matshow(np.abs(eig - np.diag(eig_diag)))
        plt.colorbar()
        plt.show()
        bloch_eigenstates = np.array(bloch_eigenstates)
        """
        ksh = mol.Electronics.KS_Hamiltonian_alpha
        olm = mol.Electronics.OLM
        prim_gam_sym, non_prim_gam_sym = separating_symmetries(mol)
        bloch_eigenstates = []

        for sym_sector in non_prim_gam_sym:
            sym_states = []
            for state in mol.Electronics.real_eigenstates["alpha"][sym_sector]:
                sym_states.append(state.a)
            sym_states = np.array(sym_states)

            periodic_bool = np.array(mol.periodicity) > 1
            full = sym_states @ ksh @ sym_states.T
            for d in [0, 1, 2]:
                if periodic_bool[d]:
                    T_op = mol.Electronics.electronic_symmetry.Symmetry_Generators["t" + str(d + 1)]
                    full += sym_states @ olm @ T_op @ sym_states.T
            val, vec = np.linalg.eig(full)
            bloch = sym_states.T @ vec
            print("unitary: ", np.max(np.abs(np.linalg.inv(vec) - np.conjugate(vec).T)))

            for i in range(bloch.shape[1]):
                bloch_eigenstates.append(bloch[:, i])

        bloch_eigenstates = np.array(bloch_eigenstates)
        # np.save(path+'Bloch_states_'+name_of_file+'.npy', bloch_eigenstates)
        print("here")
        T1 = mol.Electronics.electronic_symmetry.Symmetry_Generators["t1"]
        eig = np.conjugate(bloch_eigenstates) @ olm @ T1 @ bloch_eigenstates.T
        eig_diag = np.diagonal(eig)
        plt.matshow(np.abs(eig - np.diag(eig_diag)))
        plt.colorbar()
        plt.show()

        return None  # bloch_eigenstates.T


def separating_symmetries(mol):
    import re

    # Figuring out the translational symmetries in the structure:
    periodic_bool = np.array(mol.periodicity) > 1
    periodic = periodic_bool.astype(int)

    # Isolating Symmetry sectors that correspond to the primitive gamma point
    symsecs = list(mol.Electronics.electronic_symmetry.SymSectors.keys())

    patternT = re.compile(r"t(\d+)=([-+]?\d+)")
    prim_gam_sym = []
    for sym in symsecs:
        if "Id=1" in sym:
            check = []
            for dir in patternT.findall(sym):
                if int(dir[1]) == 1:
                    check.append(True)
                else:
                    check.append(False)
            if all(check):
                prim_gam_sym.append(sym)

    non_prim_gam_sym = list(set(symsecs) - set(prim_gam_sym))
    return prim_gam_sym, non_prim_gam_sym


def cel_periodic_overlap_calc(mol, path="./"):
    import re

    pattern = re.compile(r"^OLM_cell_per[123]\.npy$")
    already_stored_flag = False
    for file in os.listdir(path):
        if pattern.match(file):
            already_stored_flag = True

    if already_stored_flag:
        print("Found already calculated and stored phases!")
        return None
    else:
        periodic_bool = np.array(mol.periodicity) > 1
        periodic = periodic_bool.astype(int)
        basis = mol.Electronics.Basis
        xyz_filepath = Read.get_xyz_filename(path)
        atoms = Read.read_atomic_coordinates(xyz_filepath)
        q_points, _, _, unit_vectors = get_q_points(mol)
        # cellvectors = Geometry.getNeibouringCellVectors(cell=mol.cellvectors, m=periodic[0], n=periodic[1], l=periodic[2])
        cellvectors = Geometry.getNeibouringCellVectors(cell=mol.cellvectors, m=1, n=2, l=2)
        # assuming the supercell is large enough so that just the first neighbouring cells are enough; otherwise a convergence check would be needed!
        for d in [0, 1, 2]:
            if periodic_bool[d]:
                print("Calculating ", str(d + 1), " direction.")
                time1 = time.time()
                phase = AtomicBasis.get_phase_operators(
                    atoms, basis, q_vector=unit_vectors[d], cell_vectors=cellvectors, cutoff_radius=25
                )
                np.save(path + "OLM_cell_per" + str(d + 1) + ".npy", phase)
                print("time taken to make this matrix", time.time() - time1)
        return None


def recommended_kpath_bandstruc(mol, path="./"):
    # K-path for bandstructure plot to check the working of band indexing
    # Written in CP2K input format for ease
    periodic_bool = np.array(mol.periodicity) > 1
    dimension = np.sum(periodic_bool.astype(int))
    edges = [np.array([0.500, 0.000, 0.000]), np.array([0.000, 0.500, 0.000]), np.array([0.000, 0.000, 0.500])]
    gamma = np.array([0.000, 0.000, 0.000])
    file = open(path + "kpoint_set.txt", "w")
    if dimension == 1:
        line = []
        line.append(-1 * edges[np.where(periodic_bool)[0][0]])
        line.append(gamma)
        line.append(edges[np.where(periodic_bool)[0][0]])
        file.writelines("&KPOINT_SET\n")
        file.writelines("  UNITS B_VECTOR\n")
        for point in line:
            file.writelines("SPECIAL_POINT " + str(point[0]) + " " + str(point[1]) + " " + str(point[2]) + "\n")
        file.writelines("NPOINTS 20\n")
        file.writelines("&END KPOINT_SET\n")

    elif dimension == 2:
        set1, set2, set3, set4 = [], [], [], []
        set1.append(-1 * edges[np.where(periodic_bool)[0][0]])
        set1.append(gamma)
        set1.append(edges[np.where(periodic_bool)[0][0]])

        set2.append(-1 * edges[np.where(periodic_bool)[0][1]])
        set2.append(gamma)
        set2.append(edges[np.where(periodic_bool)[0][1]])

        set3.append(-1 * edges[np.where(periodic_bool)[0][0]] - 1 * edges[np.where(periodic_bool)[0][1]])
        set3.append(gamma)
        set3.append(edges[np.where(periodic_bool)[0][0]] + edges[np.where(periodic_bool)[0][1]])

        set4.append(-1 * edges[np.where(periodic_bool)[0][0]] + edges[np.where(periodic_bool)[0][1]])
        set4.append(gamma)
        set4.append(edges[np.where(periodic_bool)[0][0]] - edges[np.where(periodic_bool)[0][1]])

        for set in [set1, set2, set3, set4]:
            file.writelines("&KPOINT_SET\n")
            file.writelines("  UNITS B_VECTOR\n")
            for point in set:
                file.writelines("  SPECIAL_POINT " + str(point[0]) + " " + str(point[1]) + " " + str(point[2]) + "\n")
            file.writelines("NPOINTS 20\n")
            file.writelines("&END KPOINT_SET\n")

    elif dimension == 3:
        set1, set2, set3, set4, set5, set6, set7, set8, set9, set10, set11, set12, set13 = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

        set1 = [-1 * edges[0], gamma, edges[0]]
        set2 = [-1 * edges[1], gamma, edges[1]]
        set3 = [-1 * edges[2], gamma, edges[2]]

        set4 = [-1 * edges[0] - 1 * edges[1], gamma, edges[0] + edges[1]]
        set5 = [-1 * edges[1] - 1 * edges[2], gamma, edges[1] + edges[2]]
        set6 = [-1 * edges[0] - 1 * edges[2], gamma, edges[0] + edges[2]]

        set7 = [-1 * edges[0] + edges[1], gamma, edges[0] - edges[1]]
        set8 = [-1 * edges[1] + edges[2], gamma, edges[1] - edges[2]]
        set9 = [-1 * edges[0] + edges[2], gamma, edges[0] - edges[2]]

        set10 = [-1 * edges[0] - 1 * edges[1] - 1 * edges[2], gamma, edges[0] + edges[1] + edges[2]]
        set11 = [edges[0] - 1 * edges[1] - 1 * edges[2], gamma, -1 * edges[0] + edges[1] + edges[2]]
        set12 = [-1 * edges[0] + edges[1] - 1 * edges[2], gamma, edges[0] - 1 * edges[1] + edges[2]]
        set13 = [-1 * edges[0] - 1 * edges[1] + edges[2], gamma, edges[0] + edges[1] - 1 * edges[2]]

        for set in [set1, set2, set3, set4, set5, set6, set7, set8, set9, set10, set11, set12, set13]:
            file.writelines("&KPOINT_SET\n")
            file.writelines("  UNITS B_VECTOR\n")
            for point in set:
                file.writelines("  SPECIAL_POINT " + str(point[0]) + " " + str(point[1]) + " " + str(point[2]) + "\n")
            file.writelines("NPOINTS 20\n")
            file.writelines("&END KPOINT_SET\n")

    file.close()
    return None


def band_index_bandstruc_check(mol, band_indexing_results, nh, nl, bsfilename, path="./"):
    import re

    file = open(path + bsfilename)
    content = file.read()
    set_separator = "\n# Set"
    sets_linewise = [part.rstrip().split("\n") for part in content.split(set_separator)]
    sets = content.split(set_separator)
    pattern = r"(\d+)\s+k-points.*?(\d+)\s+bands"
    point_separator = "\n#  Point"
    special_k, kpoints_on_path, band_energies_on_path, occupations_on_path = [], [], [], []
    for num_set in range(len(sets)):
        points_in_line, num_of_bands = (
            int(re.search(pattern, sets_linewise[num_set][0]).group(1)),
            int(re.search(pattern, sets_linewise[num_set][0]).group(2)),
        )
        special_k.append(
            [np.array([sets_linewise[num_set][j].split()[i] for i in [4, 5, 6]]).astype(float) for j in [1, 2, 3]]
        )
        del sets_linewise[num_set][:4]
        set_reduced = sets[num_set].split(point_separator, 1)[1]
        points = [part.split("\n") for part in set_reduced.split(point_separator)]
        kpath = []
        this_line_energies, this_line_occupations = (
            np.zeros((points_in_line, num_of_bands)),
            np.zeros((points_in_line, num_of_bands)),
        )
        for point in range(points_in_line):
            kpath.append([np.array(points[point][0].split()[i]).astype(float) for i in [3, 4, 5]])
            del points[point][:2]
            for band in range(num_of_bands):
                this_line_energies[point, band], this_line_occupations[point, band] = (
                    float(points[point][band].split()[1]),
                    float(points[point][band].split()[2]),
                )
        kpoints_on_path.append(kpath)
        band_energies_on_path.append(this_line_energies)
        occupations_on_path.append(this_line_occupations)
    file.close()

    cell_vectors = mol.cellvectors
    periodicity = mol.periodicity
    primitive_cell_vectors = [cell_vectors[i] / periodicity[i] for i in range(3)]
    reciprocal_lattice = np.zeros((3, 3))
    reciprocal_lattice[0, :], reciprocal_lattice[1, :], reciprocal_lattice[2, :] = get_reciprocal_lattice_vectors(
        *primitive_cell_vectors
    )
    conversion_factor = ConversionFactors["a.u.->A"]
    ksh = mol.Electronics.KS_Hamiltonian_alpha

    os.makedirs(path + "Testing_band_indexing", exist_ok=True)
    for band_ind, band in enumerate(band_indexing_results):
        os.makedirs(path + "Testing_band_indexing/" + str(band_ind), exist_ok=True)
        for path_index, kpath in enumerate(special_k):
            direction_vector = (kpath[0] * conversion_factor @ reciprocal_lattice) - (
                kpath[-1] * conversion_factor @ reciprocal_lattice
            )
            occ_matrix = occupations_on_path[path_index]
            vb_index = (
                np.where(occupations_on_path[path_index][0, :] < 0.95)[0][0] - 1
            )  # Assuming the Fermi level is in the gap and no partial occupations!
            e_fermi = np.max(band_energies_on_path[path_index][:, vb_index])
            k_dist = [0]
            k_dist.extend(
                np.cumsum(
                    np.linalg.norm(
                        np.diff(((kpoints_on_path[path_index] @ reciprocal_lattice) * conversion_factor), axis=0),
                        axis=1,
                    )
                )
            )
            k_dist = np.array(k_dist)
            if nh > vb_index:
                nh = vb_index + 1
            chosen_indices = [vb_index - (nh - 1), vb_index + (nl + 1)]
            chosen_dft_bands = band_energies_on_path[path_index][:, chosen_indices[0] : chosen_indices[1]]  # -e_fermi
            plt.figure()

            for dft_band_num in range(chosen_dft_bands.shape[1]):
                plt.plot(k_dist, chosen_dft_bands[:, dft_band_num], color="Blue")

            sampled_kpoints = list(band.keys())
            for kpoint in sampled_kpoints:
                if (
                    np.round(
                        np.cross(
                            direction_vector, np.array(kpoint) - (kpath[0] * conversion_factor @ reciprocal_lattice)
                        ),
                        5,
                    )
                    == np.array([0.0, 0.0, 0.0])
                ).all():
                    dist = np.linalg.norm(
                        np.array(kpoint) - np.array(kpath[0] * conversion_factor @ reciprocal_lattice)
                    )
                    state = band[kpoint]
                    ene = np.real(np.conjugate(state.T) @ ksh @ state) * 27.211  # - e_fermi
                    plt.scatter(dist, ene)

            plt.axhline(y=0, color="gray", linestyle="--", linewidth=0.75)
            plt.xticks(
                [0, k_dist[int(np.floor(float(k_dist.shape[0] * 0.5)))], k_dist[int(k_dist.shape[0] - 1)]], kpath
            )
            figname = path + "Testing_band_indexing/" + str(band_ind) + "/" + str(path_index) + ".png"
            plt.savefig(figname, dpi=600)
            plt.close()

    return None


def isolate_prim_gam_states(mol, nh, nl):
    # Isolating the states that correspond to the primitive BZ gamma point
    p_gamma_ind = []
    periodic_bool = np.array(mol.periodicity) > 1
    periodic = periodic_bool.astype(int)
    prim_gam_sym, _ = separating_symmetries(mol)

    estate_dict = mol.Electronics.indexmap["alpha"]

    finalind = next(iter(estate_dict))
    occ_ind = -1 * np.arange(0, -1 * finalind + 1)
    prim_homo_energy, prim_lumo_energy = 0, 0
    occ_prim_gam_states = []
    if nh > 0:
        energies = []
        for ind in occ_ind:
            if estate_dict[ind][0] in prim_gam_sym:
                occ_prim_gam_states.append(estate_dict[ind])
                energies.append(
                    mol.Electronics.real_eigenstates["alpha"][estate_dict[ind][0]][int(estate_dict[ind][1])].energy
                )
                if len(occ_prim_gam_states) == nh:
                    break

        prim_homo_energy = np.max(np.array(energies))

    unocc_prim_gam_states = []
    if nl > 0:
        energies = []
        for ind in range(1, int(len(estate_dict) - np.shape(occ_ind)[0])):
            if estate_dict[ind][0] in prim_gam_sym:
                unocc_prim_gam_states.append(estate_dict[ind])
                energies.append(
                    mol.Electronics.real_eigenstates["alpha"][estate_dict[ind][0]][int(estate_dict[ind][1])].energy
                )
                if len(unocc_prim_gam_states) == nl:
                    break
        prim_lumo_energy = np.min(np.array(energies))

    unocc_prim_gam_states.reverse()
    prim_gam_states = unocc_prim_gam_states + occ_prim_gam_states

    return prim_gam_states, prim_homo_energy, prim_lumo_energy


def debug_pk_mk(mol):
    import pickle

    over_mat1 = np.load("OLM_cell_per1.npy")
    over_mat2 = np.load("OLM_cell_per2.npy")
    ksh = mol.Electronics.KS_Hamiltonian_alpha

    with open("k_resolved_blochs.pickle", "rb") as f:
        k_blochs = pickle.load(f)

    point1 = tuple([-0.2, 0.0, 0.0])
    point2 = tuple([-0.2, 0.2, 0.0])
    print(point1, point2)
    for k_state1 in k_blochs[point1]:
        for k_state2 in k_blochs[point2]:
            energy1 = np.real(np.conjugate(k_state1) @ ksh @ k_state1.T)
            energy2 = np.real(np.conjugate(k_state2) @ ksh @ k_state2.T)
            if -5.5 > energy1 * 27.211 > -10.5 and -5.5 > energy2 * 27.211 > -10.5:
                print("---------------------------------")
                print(energy1 * 27.211, energy2 * 27.211)
                print(
                    np.conjugate(k_state1.T) @ over_mat2 @ k_state2,
                    np.abs(np.conjugate(k_state1.T) @ over_mat2 @ k_state2),
                )

    return None


debug_q_path1 = [
    [np.array((0.0, 0.0, 0.0)), np.array((0.2, 0.0, 0.0)), 0, +1],
    [np.array((0.0, 0.0, 0.0)), np.array((-0.2, 0.0, 0.0)), 0, -1],
    [np.array((0.2, 0.0, 0.0)), np.array((0.4, 0.0, 0.0)), 0, 1],
    [np.array((-0.2, 0.0, 0.0)), np.array((-0.4, 0.0, 0.0)), 0, -1],
]
debug_q_path2 = [
    [np.array((0.0, 0.0, 0.0)), np.array((0.0, 0.2, 0.0)), 1, +1],
    [np.array((0.0, 0.0, 0.0)), np.array((0.0, -0.2, 0.0)), 1, -1],
    [np.array((0.0, 0.2, 0.0)), np.array((0.0, 0.4, 0.0)), 1, +1],
    [np.array((0.0, -0.2, 0.0)), np.array((0.0, -0.4, 0.0)), 1, -1],
]
debug_q_path3 = [
    [np.array((0.2, 0.0, 0.0)), np.array((0.2, 0.2, 0.0)), 1, +1],
    [np.array((0.2, 0.2, 0.0)), np.array((0.4, 0.2, 0.0)), 0, +1],
    [np.array((0.4, 0.2, 0.0)), np.array((0.4, 0.4, 0.0)), 1, +1],
    [np.array((-0.2, 0.0, 0.0)), np.array((-0.2, -0.2, 0.0)), 1, -1],
    [np.array((-0.2, -0.2, 0.0)), np.array((-0.4, -0.2, 0.0)), 0, -1],
    [np.array((-0.4, -0.2, 0.0)), np.array((-0.4, -0.4, 0.0)), 1, -1],
]
debug_q_path4 = [
    [np.array((0.2, 0.0, 0.0)), np.array((0.2, -0.2, 0.0)), 1, -1],
    [np.array((0.2, -0.2, 0.0)), np.array((0.4, -0.2, 0.0)), 0, +1],
    [np.array((0.4, -0.2, 0.0)), np.array((0.4, -0.4, 0.0)), 1, -1],
    [np.array((-0.2, 0.0, 0.0)), np.array((-0.2, 0.2, 0.0)), 1, 1],
    [np.array((-0.2, 0.2, 0.0)), np.array((-0.4, 0.2, 0.0)), 0, -1],
    [np.array((-0.4, 0.2, 0.0)), np.array((-0.4, 0.4, 0.0)), 1, 1],
]
debug_q_path5 = [
    [np.array((0.2, 0.2, 0.0)), np.array((0.2, 0.4, 0.0)), 1, +1],
    [np.array((-0.2, -0.2, 0.0)), np.array((-0.2, -0.4, 0.0)), 1, -1],
    [np.array((0.2, -0.2, 0.0)), np.array((0.2, -0.4, 0.0)), 1, -1],
    [np.array((-0.2, 0.2, 0.0)), np.array((-0.2, 0.4, 0.0)), 1, +1],
]
full_debug_q_path = debug_q_path1 + debug_q_path2 + debug_q_path3 + debug_q_path4 + debug_q_path5


def band_index(mol, nh, nl, name_of_bloch_file="general", name_of_k_res_bloch="k_resolved_blochs.pickle", path="./"):
    # checking if all the k-points of the primitive brillouin zone that are expected to be there are there
    q_points, cart_to_direct_dict, direct_to_cart_dict, unit_vectors = get_q_points(mol)

    # with open(path + "k_resolved_blochs.pickle", "rb") as f:
    #    k_resolv_blochs = pickle.load(f)
    # sampled_qpoints = list(k_resolv_blochs.keys())
    # sampled_qpoints.append(tuple([0.0, 0.0 ,0.0]))

    # if np.all([np.any(np.all(np.isclose(np.array(this_point), np.array(sampled_qpoints), atol=1e-7), axis=1)) for this_point in direct_q_points]):
    #    print('Sampled k-points match the predcted k-points.')

    prim_gam_states, _, _ = isolate_prim_gam_states(mol, nh, nl)
    periodic_bool = np.array(mol.periodicity) > 1
    periodic = periodic_bool.astype(int)
    basis = mol.Electronics.Basis
    xyz_filepath = Read.get_xyz_filename(path)
    atoms = Read.read_atomic_coordinates(xyz_filepath)

    unit_vectors = [np.array(vector) for vector in unit_vectors]
    q_points.sort(key=lambda x: np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2))
    q_arrays = [np.array(point) for point in q_points]
    q_path = simple_qpath(q_arrays, unit_vectors=unit_vectors)
    # q_path = full_debug_q_path
    # for step in q_path:
    #    print(cart_to_direct_dict[tuple(np.round(step[0],6))],cart_to_direct_dict[tuple(np.round(step[1],6))])
    olm = mol.Electronics.OLM
    ksh = mol.Electronics.KS_Hamiltonian_alpha

    # cellvectors = Geometry.getNeibouringCellVectors(cell=mol.cellvectors, m=periodic[0], n=periodic[1], l=periodic[2]) # assuming the supercell is large enough so that just the first neighbouring cells are enough; otherwise a convergence check would be needed!
    cellvectors = Geometry.getNeibouringCellVectors(cell=mol.cellvectors, m=3, n=3, l=1)
    all_connected_bands = []
    bloch_states_full = testing_bloch(
        mol, path=path, name_of_file=name_of_bloch_file
    )  # bloch_form(mol, path=path, name_of_file=name_of_bloch_file)
    determine_k_point(mol, bloch_states_full)
    available_mask = np.ones(bloch_states_full.shape[1], dtype=bool)

    phi_q = []
    cel_periodic_overlap_calc(mol, path=path)
    for d in [0, 1, 2]:
        if periodic_bool[d]:
            phase = "OLM_cell_per" + str(d + 1) + ".npy"
            phi_q.append(phase)
        else:
            phi_q.append(0)

    prim_blochs = []
    for state in prim_gam_states:
        prim_blochs.append(mol.Electronics.real_eigenstates["alpha"][state[0]][int(state[1])].a)
        ene = mol.Electronics.real_eigenstates["alpha"][state[0]][int(state[1])].energy
        print("prim_gam_energies: ", ene, ene * 27.211)

    for state in prim_blochs:
        band = {}
        band[tuple(q_arrays[0])] = state
        calculated_q_points = []
        for q_index in range(0, len(q_path)):
            previous_qpoint, current_qpoint, del_q, direc = (
                q_path[q_index][0],
                q_path[q_index][1],
                int(q_path[q_index][2]),
                int(q_path[q_index][3]),
            )
            # previous_qpoint, current_qpoint, del_q, direc =  direct_to_cart_dict[tuple(q_path[q_index][0])], direct_to_cart_dict[tuple(q_path[q_index][1])], int(q_path[q_index][2]), int(q_path[q_index][3])

            if direc == +1:
                phase = np.load(path + phi_q[del_q])
            elif direc == -1:
                phase = np.conjugate(np.load(path + phi_q[del_q]).T)
            else:
                print("Some problem with the q_path calculation!", direc)

            calculated_q_points.append(previous_qpoint)
            previous_state = band[tuple(previous_qpoint)]

            if not np.any(np.all(np.isclose(np.array(calculated_q_points), current_qpoint, atol=1e-7), axis=1)):
                available_indices = np.where(available_mask)[0]
                overlaps = np.round(
                    np.abs(np.conjugate(bloch_states_full[:, available_indices].T) @ phase @ previous_state), 6
                )
                sortedover = np.sort(overlaps)

                # Extract the local index from the overlap array (overlaps is small)
                indices_max_local = np.where(overlaps == np.max(overlaps))[0]
                index_max_local = indices_max_local[0]

                # Find the global index and the state from the full array
                index_max_global = available_indices[index_max_local]
                max_state = bloch_states_full[:, index_max_global]

                # 4. Update the mask
                available_mask[index_max_global] = False

                band[tuple(current_qpoint)] = max_state
                print(
                    np.round(cart_to_direct_dict[tuple(np.round(previous_qpoint, 6))], 3),
                    np.round(cart_to_direct_dict[tuple(np.round(current_qpoint, 6))], 3),
                    sortedover[-5:],
                    np.real(np.conjugate(previous_state.T) @ ksh @ previous_state) * 27.211,
                    np.real(np.conjugate(max_state.T) @ ksh @ max_state) * 27.211,
                )
                calculated_q_points.append(current_qpoint)

        all_connected_bands.append(band)

    return all_connected_bands


def old_band_index(mol, nh, nl, name_of_bloch_file="general", path="./"):
    # Isolating the states that correspond to the primitive BZ gamma point
    p_gamma_ind = []
    periodic_bool = np.array(mol.periodicity) > 1
    periodic = periodic_bool.astype(int)
    prim_gam_sym, non_prim_gam_sym = separating_symmetries(mol)

    estate_dict = mol.Electronics.indexmap["alpha"]
    for number in range(-5, 5):
        this_state = estate_dict[number]
        this_state_en = mol.Electronics.real_eigenstates["alpha"][this_state[0]][int(this_state[1])].energy
        print("energy check: ", number, this_state_en)

    finalind = next(iter(estate_dict))
    occ_ind = -1 * np.arange(0, -1 * finalind + 1)

    occ_prim_gam_states = []
    if nh > 0:
        for ind in occ_ind:
            if estate_dict[ind][0] in prim_gam_sym:
                occ_prim_gam_states.append(estate_dict[ind])
                if len(occ_prim_gam_states) == nh:
                    break

    unocc_prim_gam_states = []
    if nl > 0:
        for ind in range(0, int(len(estate_dict) - np.shape(occ_ind)[0])):
            if estate_dict[ind][0] in prim_gam_sym:
                unocc_prim_gam_states.append(estate_dict[ind])
                if len(unocc_prim_gam_states) == nl:
                    break

    unocc_prim_gam_states.reverse()
    prim_gam_states = unocc_prim_gam_states + occ_prim_gam_states

    basis = mol.Electronics.Basis
    xyz_filepath = Read.get_xyz_filename(path)
    atoms = Read.read_atomic_coordinates(xyz_filepath)
    q_points, direct_q_points, unit_vectors = get_q_points(mol)
    unit_vectors = [np.array(vector) for vector in unit_vectors]
    q_points.sort(key=lambda x: np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2))
    q_arrays = [np.array(point) for point in q_points]

    olm = mol.Electronics.OLM

    cellvectors = Geometry.getNeibouringCellVectors(
        cell=mol.cellvectors, m=periodic[0], n=periodic[1], l=periodic[2]
    )  # assuming the supercell is large enough so that just the first neighbouring cells are enough; otherwise a convergence check would be needed!

    all_connected_bands = []
    bloch_states_full = bloch_form(mol, path=path, name_of_file=name_of_bloch_file)
    available_mask = np.ones(bloch_states_full.shape[1], dtype=bool)

    prim_blochs = []
    for state in prim_gam_states:
        prim_blochs.append(mol.Electronics.real_eigenstates["alpha"][state[0]][int(state[1])].a)
        print(mol.Electronics.real_eigenstates["alpha"][state[0]][int(state[1])].energy)

    for state in prim_blochs:
        band = {}
        band[tuple(q_arrays[0])] = state
        calculated_q_points = []
        for q_index in range(1, len(q_arrays)):
            previous_qpoint, current_qpoint = q_arrays[0], q_arrays[q_index]
            phase = AtomicBasis.get_phase_operators(
                atoms, basis, q_vector=current_qpoint, cell_vectors=cellvectors, cutoff_radius=25
            )
            calculated_q_points.append(previous_qpoint)
            previous_state = band[tuple(previous_qpoint)]
            third = time.time()
            if not np.any(np.all(np.isclose(np.array(calculated_q_points), current_qpoint, atol=1e-7), axis=1)):
                available_indices = np.where(available_mask)[0]
                overlaps = np.round(
                    np.abs(np.conjugate(bloch_states_full[:, available_indices].T) @ phase @ previous_state), 6
                )
                sortedover = np.sort(overlaps)
                print(previous_qpoint, current_qpoint, sortedover[-5:])
                # Extract the local index from the overlap array (overlaps is small)
                indices_max_local = np.where(overlaps == np.max(overlaps))[0]
                index_max_local = indices_max_local[0]

                # Find the global index and the state from the full array
                index_max_global = available_indices[index_max_local]
                max_state = bloch_states_full[:, index_max_global]

                # 4. Update the mask
                available_mask[index_max_global] = False

                # import gc
                # gc.collect()

                band[tuple(current_qpoint)] = max_state
                calculated_q_points.append(current_qpoint)

        all_connected_bands.append(band)

    return all_connected_bands


def wannierise(mol, band_index_results, frags, path="./"):
    import scipy.linalg as sp

    file1 = open(path + "MolOrb")
    content = file1.readlines()
    olm = mol.Electronics.OLM
    num_of_bands = len(band_index_results)
    k_points = list(band_index_results[0].keys())
    num_k_points = len(k_points)
    bloch_state_size = olm.shape[0]
    supercell_size = mol.periodicity[0] * mol.periodicity[1] * mol.periodicity[2]

    prim_blochs = []
    gamma = tuple([0.0, 0.0, 0.0])
    for band in band_index_results:
        state = band[gamma]
        if np.max(abs(np.imag(state))) > 10**-6:
            print("Caution! Primitive Gamma states are imaginary!")
        prim_blochs.append(np.real(state))

    trial_orbs = fragmentize(prim_blochs, frags, content, bloch_state_size)

    Wannier_Orbs = np.zeros((trial_orbs.shape[1], bloch_state_size))
    for point in k_points:
        blochs_on_this_point = np.zeros((num_of_bands, bloch_state_size), dtype="complex128")
        for ind, band in enumerate(band_index_results):
            blochs_on_this_point[ind, :] = band[point]

        projection_mat = np.conjugate(blochs_on_this_point) @ olm @ trial_orbs
        u, s, vt = sp.svd(projection_mat)
        delta = np.zeros((np.shape(u)[0], np.shape(vt)[0]))
        np.fill_diagonal(delta, 1)
        Uk = u @ delta @ vt
        exp = 1.0
        Wannier_Orbs = Wannier_Orbs + (exp * Uk.T @ blochs_on_this_point)

    if np.max(abs(np.imag(Wannier_Orbs))) > 10**-6:
        print(
            "Warning, the Wannier orbitals are imaginary!. Max. imaginary component:",
            np.max(abs(np.imag(Wannier_Orbs))),
        )

    Wannier_Orbs = np.real(Wannier_Orbs / np.sqrt(supercell_size))

    print("Orthonormality of the Wannier orbitals:")
    print(np.max(abs((Wannier_Orbs @ olm @ Wannier_Orbs.T) - np.eye(trial_orbs.shape[1]))))

    return Wannier_Orbs


def wan_real_plot(mol, mode, ind=[0], Wan_npy="Wannier_orbitals_occ.npy", path="./", N1=100, N2=100, N3=100):
    if mode.casefold() == "q":
        supercell_size = mol.periodicity[0] * mol.periodicity[1] * mol.periodicity[2]
        Wans = np.load(path + Wan_npy)
        num_basis = Wans.shape[1]
        X = np.linspace(0, num_basis, num=num_basis)
        pos = np.linspace(0, num_basis, num=supercell_size + 1, endpoint=True)
        for orb in ind:
            plt.plot(X, Wans[orb, :])
        plt.xticks(pos)
        plt.grid()
        plt.show()
    elif mode.casefold() == "a":
        periodic_bool = np.array(mol.periodicity) > 1
        periodic = periodic_bool.astype(int)
        cellvectors = Geometry.getNeibouringCellVectors(
            cell=mol.cellvectors, m=periodic[0], n=periodic[1], l=periodic[2]
        )  # assuming the supercell is large enough so that just the first neighbouring cells are enough; otherwise a convergence check would be needed!
        data, gridx, gridy, gridz, atoms = TDDFT.WFNsOnGrid(
            ids=ind,
            N1=N1,
            N2=N2,
            N3=N3,
            cell_vectors=cellvectors,
            wannier_printflag=True,
            Wan_file=Wan_npy,
            saveflag=False,
            parentfolder=path,
        )
        for orb_index, orb in enumerate(ind):
            Write.write_cube_file(
                gridx,
                gridy,
                gridz,
                data[orb_index, :, :, :],
                atoms,
                filename="w" + str(orb) + ".cube",
                parentfolder=path,
            )

    else:
        print("Please specify either q or a in mode for the quick&dirty mode or accurate mode respectively")
    return None


###
# def wan_interpolate_bandstruc(mol, special_k_points, Wan_file='Wannier_orbitals8lumobands.npy', path='./'):
#    periodic_bool = np.array(mol.periodicity) > 1
#    periodic = periodic_bool.astype(int)
#    ksh = mol.Electronics.KS_Hamiltonian_alpha
#    olm = mol.Electronics.OLM
#    Wan_orbs = np.load(path+Wan_file)
#    onsite = Wan_orbs @ ksh @ Wan_orbs.T
#
#    #first nearest neighbour interaction
#    transfer_ints = []
#    for d in [0, 1, 2]:
#        if periodic_bool[d]:
#            T_op = mol.Electronics.electronic_symmetry.Symmetry_Generators['t' + str(d + 1)]
#            tr = Wan_orbs @ ksh @ T_op @ Wan_orbs.T
#            transfer_ints.append(tr)
#        else:
#            transfer_ints.append(np.zeros(onsite.shape[0]))
#
#    # 2nd nearest neighbour interaction
#    transfer_ints_2nd_nearest = []
#    for d1, d2 in [[0, 1], [1, 2], [0, 2]]:
#        if periodic_bool[d1] and periodic_bool[d2]:
#            T_op1 = mol.Electronics.electronic_symmetry.Symmetry_Generators['t' + str(d1 + 1)]
#            T_op2 = mol.Electronics.electronic_symmetry.Symmetry_Generators['t' + str(d2 + 1)]
#            tr = Wan_orbs @ ksh @ T_op2 @ T_op1 @ Wan_orbs.T
#            transfer_ints_2nd_nearest.append(tr)
#            tr = Wan_orbs @ ksh @ T_op2.T @ T_op1 @ Wan_orbs.T
#            transfer_ints_2nd_nearest.append(tr)
#        else:
#            transfer_ints_2nd_nearest.append(np.zeros(onsite.shape[0]))
#            transfer_ints_2nd_nearest.append(np.zeros(onsite.shape[0]))
#
#    # 3rd nearest neighbour interaction
#    transfer_ints_3rd_nearest = []
#    for d in [0, 1, 2]:
#        if periodic_bool[d]:
#            T_op = mol.Electronics.electronic_symmetry.Symmetry_Generators['t' + str(d + 1)]
#            tr = Wan_orbs @ ksh @ T_op @ T_op @ Wan_orbs.T
#            transfer_ints_3rd_nearest.append(tr)
#        else:
#            transfer_ints_3rd_nearest.append(np.zeros(onsite.shape[0]))
#
#    # 4rth nearest neighbour interaction
#    transfer_ints_4rth_nearest = []
#    for d in [0, 1, 2]:
#        if periodic_bool[d]:
#            T_op = mol.Electronics.electronic_symmetry.Symmetry_Generators['t' + str(d + 1)]
#            tr = Wan_orbs @ ksh @ T_op @ T_op @ Wan_orbs.T
#            transfer_ints_3rd_nearest.append(tr)
#        else:
#            transfer_ints_3rd_nearest.append(np.zeros(onsite.shape[0]))
#
#    T_opa = mol.Electronics.electronic_symmetry.Symmetry_Generators['t1']
#    T_opb = mol.Electronics.electronic_symmetry.Symmetry_Generators['t2']
#
#    #tr_ab = Wan_orbs @ ksh @ T_opb @ T_opa @ Wan_orbs.T
#
#    #tr_amb = Wan_orbs @ ksh @ T_opb.T @ T_opa @ Wan_orbs.T
#    print(Wan_orbs @ ksh @ T_opb @ T_opa.T @ Wan_orbs.T)
#    #tr_2a = Wan_orbs @ ksh @ T_opa @ T_opa @ Wan_orbs.T
#    #tr_2b = Wan_orbs @ ksh @ T_opb @ T_opb @ Wan_orbs.T
#
#    tr_2apb = Wan_orbs @ ksh @ T_opb @ T_opa @ T_opa @ Wan_orbs.T
#    tr_2amb = Wan_orbs @ ksh @ T_opb.T @ T_opa @ T_opa @ Wan_orbs.T
#    tr_2bpa = Wan_orbs @ ksh @ T_opa @ T_opb @ T_opb @ Wan_orbs.T
#    tr_2bma = Wan_orbs @ ksh @ T_opa.T @ T_opb @ T_opb @ Wan_orbs.T
#
#    tr_2ap2b = Wan_orbs @ ksh @ T_opb @ T_opb @ T_opa @ T_opa @ Wan_orbs.T
#    tr_2am2b = Wan_orbs @ ksh @ T_opb.T @ T_opb.T @ T_opa @ T_opa @ Wan_orbs.T
#
#
#
#    conversion_factor = ConversionFactors["a.u.->A"]
#    prim_unit_cellvectors = [ mol.cellvectors[d]/(mol.periodicity[d]*conversion_factor) for d in [0,1,2] ]
#
#    reciprocal_lattice = np.zeros((3, 3))
#    reciprocal_lattice[0, :], reciprocal_lattice[1, :], reciprocal_lattice[2, :] = get_reciprocal_lattice_vectors(*prim_unit_cellvectors)
#    # Following needs to be generalised for more dimensions..
#    ka = np.linspace( -0.5, 0.5, num=40, endpoint=True)
#    kb = np.linspace(-0.0, 0.0, num=40, endpoint=True)
#    kc = np.linspace(-0.0, 0.0, num=40, endpoint=True)
#    k_points = []
#    for i in range(len(ka)):
#        k_points.append(np.array([ka[i],kb[i],kc[i]]) @ reciprocal_lattice)
#
#    TB_bands = np.zeros((Wan_orbs.shape[0], int(len(k_points))))
#    q = 0
#    for k in k_points:
#        TBH = np.array(onsite.copy(), dtype="complex128")
#        k = np.array(k)
#        TBH += np.exp((1.0j) * k @ prim_unit_cellvectors[0]) * transfer_ints[0] + np.exp((1.0j) * k @ prim_unit_cellvectors[1]) * transfer_ints[1] + np.exp((1.0j) * k @ prim_unit_cellvectors[2]) * transfer_ints[2]
#        TBH += np.exp((-1.0j) * k @ prim_unit_cellvectors[0]) * transfer_ints[0].T + np.exp((-1.0j) * k @ prim_unit_cellvectors[1]) * transfer_ints[1].T + np.exp((-1.0j) * k @ prim_unit_cellvectors[2]) * transfer_ints[2].T
#
#        #2nd nearest neighbour
#        TBH += np.exp((1.0j) * k @ (prim_unit_cellvectors[0] + prim_unit_cellvectors[1])) * transfer_ints_2nd_nearest[0] + np.exp((1.0j) * k @ (prim_unit_cellvectors[0] - prim_unit_cellvectors[1])) * transfer_ints_2nd_nearest[1]
#        TBH += np.exp((1.0j) * k @ (prim_unit_cellvectors[1] + prim_unit_cellvectors[2])) * transfer_ints_2nd_nearest[2] + np.exp((1.0j) * k @ (prim_unit_cellvectors[1] - prim_unit_cellvectors[2])) * transfer_ints_2nd_nearest[3]
#        TBH += np.exp((1.0j) * k @ (prim_unit_cellvectors[0] + prim_unit_cellvectors[2])) * transfer_ints_2nd_nearest[4] + np.exp((1.0j) * k @ (prim_unit_cellvectors[0] - prim_unit_cellvectors[2])) * transfer_ints_2nd_nearest[5]
#        TBH += np.exp((-1.0j) * k @ (prim_unit_cellvectors[0] + prim_unit_cellvectors[1])) * transfer_ints_2nd_nearest[0].T + np.exp((-1.0j) * k @ (prim_unit_cellvectors[0] - prim_unit_cellvectors[1])) * transfer_ints_2nd_nearest[1].T
#        TBH += np.exp((-1.0j) * k @ (prim_unit_cellvectors[1] + prim_unit_cellvectors[2])) * transfer_ints_2nd_nearest[2].T + np.exp((-1.0j) * k @ (prim_unit_cellvectors[1] - prim_unit_cellvectors[2])) * transfer_ints_2nd_nearest[3].T
#        TBH += np.exp((-1.0j) * k @ (prim_unit_cellvectors[0] + prim_unit_cellvectors[2])) * transfer_ints_2nd_nearest[4].T + np.exp((-1.0j) * k @ (prim_unit_cellvectors[0] - prim_unit_cellvectors[2])) * transfer_ints_2nd_nearest[5].T
#
#        #TBH += np.exp((1.0j) * k @ (prim_unit_cellvectors[0]+prim_unit_cellvectors[1])) * tr_ab + np.exp((1.0j) * k @ (prim_unit_cellvectors[0]-prim_unit_cellvectors[1])) * tr_amb
#        #TBH += np.exp((-1.0j) * k @ (prim_unit_cellvectors[0] + prim_unit_cellvectors[1])) * tr_ab.T + np.exp((-1.0j) * k @ (prim_unit_cellvectors[0] - prim_unit_cellvectors[1])) * tr_amb.T
#
#        # 3rd nearest neighbour
#        TBH += np.exp((1.0j) * k @ (2 * prim_unit_cellvectors[0])) * transfer_ints_3rd_nearest[0] + np.exp((1.0j) * k @ (2 * prim_unit_cellvectors[1])) * transfer_ints_3rd_nearest[1] + np.exp((1.0j) * k @ (2 * prim_unit_cellvectors[2])) * transfer_ints_3rd_nearest[2]
#        TBH += np.exp((-1.0j) * k @ (2 * prim_unit_cellvectors[0])) * transfer_ints_3rd_nearest[0].T + np.exp((-1.0j) * k @ (2 * prim_unit_cellvectors[1])) * transfer_ints_3rd_nearest[1].T + np.exp((-1.0j) * k @ (2 * prim_unit_cellvectors[2])) * transfer_ints_3rd_nearest[2].T
#
#        #TBH += np.exp((1.0j) * k @ (2*prim_unit_cellvectors[0])) * tr_2a + np.exp((1.0j) * k @ (2*prim_unit_cellvectors[1])) * tr_2b
#        #TBH += np.exp((-1.0j) * k @ (2 * prim_unit_cellvectors[0])) * tr_2a.T + np.exp((-1.0j) * k @ (2 * prim_unit_cellvectors[1])) * tr_2b.T
#
#        TBH += np.exp((1.0j) * k @ ( (2*prim_unit_cellvectors[0]) + prim_unit_cellvectors[1])) * tr_2apb + np.exp((1.0j) * k @ ((2*prim_unit_cellvectors[1])+prim_unit_cellvectors[0])) * tr_2bpa
#        TBH += np.exp((1.0j) * k @ ((2 * prim_unit_cellvectors[0]) - prim_unit_cellvectors[1])) * tr_2amb + np.exp((1.0j) * k @ ((2 * prim_unit_cellvectors[1]) - prim_unit_cellvectors[0])) * tr_2bma
#        TBH += np.exp((-1.0j) * k @ ((2 * prim_unit_cellvectors[0]) + prim_unit_cellvectors[1])) * tr_2apb.T + np.exp((-1.0j) * k @ ((2 * prim_unit_cellvectors[1]) + prim_unit_cellvectors[0])) * tr_2bpa.T
#        TBH += np.exp((-1.0j) * k @ ((2 * prim_unit_cellvectors[0]) - prim_unit_cellvectors[1])) * tr_2amb.T + np.exp((-1.0j) * k @ ((2 * prim_unit_cellvectors[1]) - prim_unit_cellvectors[0])) * tr_2bma.T
#
#        TBH += np.exp((1.0j) * k @ (2 * (prim_unit_cellvectors[0] + prim_unit_cellvectors[1]))) * tr_2ap2b + np.exp((1.0j) * k @ (2 * (prim_unit_cellvectors[0] - prim_unit_cellvectors[1]))) * tr_2am2b
#        TBH += np.exp((-1.0j) * k @ (2 * (prim_unit_cellvectors[0] + prim_unit_cellvectors[1]))) * tr_2ap2b.T + np.exp((-1.0j) * k @ (2 * (prim_unit_cellvectors[0] - prim_unit_cellvectors[1]))) * tr_2am2b.T
#        eigvals = np.linalg.eigvalsh(TBH)
#        TB_bands[:, q] = eigvals
#        q += 1
#
#    TB_bands = 27.211 * TB_bands
#
#    k_dist = [0]
#    k_dist.extend(np.cumsum(np.linalg.norm(np.diff((np.array(k_points)), axis=0),axis=1)))
#    k_distance = np.array(k_dist)
#
#    #for band in range(Wan_orbs.shape[0]):
#    #    plt.plot(k_distance,TB_bands[band,:])
#
#    #plt.show()
#
#    # ------------------------------------------------------------------------------------------------------------------------------
#    bsfilename = 'graphPBE.bs'
#    nh = 2
#    nl = 8
#    import re
#    file = open(path + bsfilename, 'r')
#    content = file.read()
#    set_separator = f'\n# Set'
#    sets_linewise = [part.rstrip().split('\n') for part in content.split(set_separator)]
#    sets = content.split(set_separator)
#    pattern = r"(\d+)\s+k-points.*?(\d+)\s+bands"
#    point_separator = f'\n#  Point'
#    special_k, kpoints_on_path, band_energies_on_path, occupations_on_path = [], [], [], []
#    for num_set in range(len(sets)):
#        points_in_line, num_of_bands = int(re.search(pattern, sets_linewise[num_set][0]).group(1)), int(
#            re.search(pattern, sets_linewise[num_set][0]).group(2))
#        special_k.append(
#            [np.array([sets_linewise[num_set][j].split()[i] for i in [4, 5, 6]]).astype(float) for j in [1, 2, 3]])
#        del sets_linewise[num_set][:4]
#        set_reduced = sets[num_set].split(point_separator, 1)[1]
#        points = [part.split('\n') for part in set_reduced.split(point_separator)]
#        kpath = []
#        this_line_energies, this_line_occupations = np.zeros((points_in_line, num_of_bands)), np.zeros(
#            (points_in_line, num_of_bands))
#        for point in range(points_in_line):
#            kpath.append([np.array(points[point][0].split()[i]).astype(float) for i in [3, 4, 5]])
#            del points[point][:2]
#            for band in range(num_of_bands):
#                this_line_energies[point, band], this_line_occupations[point, band] = float(
#                    points[point][band].split()[1]), float(points[point][band].split()[2])
#        kpoints_on_path.append(kpath)
#        band_energies_on_path.append(this_line_energies)
#        occupations_on_path.append(this_line_occupations)
#    file.close()
#
#    cell_vectors = mol.cellvectors
#    periodicity = mol.periodicity
#    primitive_cell_vectors = [cell_vectors[i] / periodicity[i] for i in range(3)]
#    reciprocal_lattice = np.zeros((3, 3))
#    reciprocal_lattice[0, :], reciprocal_lattice[1, :], reciprocal_lattice[2, :] = get_reciprocal_lattice_vectors(
#        *primitive_cell_vectors)
#    conversion_factor = ConversionFactors["a.u.->A"]
#    ksh = mol.Electronics.KS_Hamiltonian_alpha
#
#    os.makedirs(path + 'Testing_wantb', exist_ok=True)
#    for path_index, kpath in enumerate(special_k):
#
#        direction_vector = (kpath[0] * conversion_factor @ reciprocal_lattice) - (
#                kpath[-1] * conversion_factor @ reciprocal_lattice)
#        occ_matrix = occupations_on_path[path_index]
#        vb_index = np.where(occupations_on_path[path_index][0, :] < 0.95)[0][0] - 1  # Assuming the Fermi level is in the gap and no partial occupations!
#        e_fermi = np.max(band_energies_on_path[path_index][:, vb_index])
#        k_dist = [0]
#        k_dist.extend(np.cumsum(np.linalg.norm(np.diff(((kpoints_on_path[path_index] @ reciprocal_lattice) * conversion_factor), axis=0), axis=1)))
#        k_dist = np.array(k_dist)
#        if nh > vb_index:
#            nh = vb_index + 1
#        chosen_indices = [vb_index - (nh - 1), vb_index + (nl + 1)]
#        chosen_dft_bands = band_energies_on_path[path_index][:, chosen_indices[0]:chosen_indices[1]]  # -e_fermi
#        plt.figure()
#
#        for dft_band_num in range(chosen_dft_bands.shape[1]):
#            plt.plot(k_dist, chosen_dft_bands[:, dft_band_num], color='Blue')
#        for band in range(Wan_orbs.shape[0]):
#            plt.scatter(k_distance, TB_bands[band, :], color='red',s=10)
#
#        plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.75)
#        plt.xticks([0, k_dist[int(np.floor(float(k_dist.shape[0] * 0.5)))], k_dist[int(k_dist.shape[0] - 1)]],kpath)
#        figname = path + 'Testing_wantb/' + str(path_index) + '.png'
#        plt.savefig(figname, dpi=600)
#        plt.close()
#
#    return None
#


def general_wan_interpolate_bands(mol, special_k_points, Wan_file="Wannier_orbitals8lumobands.npy", path="./"):
    periodic_bool = np.array(mol.periodicity) > 1
    periodic = periodic_bool.astype(int)
    a, b, c = mol.periodicity[0], mol.periodicity[1], mol.periodicity[2]
    grida = np.linspace(-np.floor(a / 2), np.floor(a / 2), a, endpoint=True)
    gridb = np.linspace(-np.floor(b / 2), np.floor(b / 2), b, endpoint=True)
    gridc = np.linspace(-np.floor(c / 2), np.floor(c / 2), c, endpoint=True)
    conversion_factor = ConversionFactors["a.u.->A"]
    prim_unit_cellvectors = [mol.cellvectors[d] / (mol.periodicity[d] * conversion_factor) for d in [0, 1, 2]]
    inverse_prim_cell = np.linalg.inv(prim_unit_cellvectors)

    points = []
    for av in grida:
        for bv in gridb:
            for cv in gridc:
                points.append(np.array([av, bv, cv]) @ prim_unit_cellvectors)

    points.sort(key=lambda x: np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2))
    points_dist = [np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2) for x in points]

    # decide upto where to take interaction into account
    upto_neighbour = 5
    neighbours = [points[index] @ inverse_prim_cell for index in list(group_equal_indices(points_dist)[0])]
    for upto in range(1, upto_neighbour + 1):
        for index in list(group_equal_indices(points_dist)[upto]):
            neighbours.append(np.round(points[index] @ inverse_prim_cell, 0))

    ksh = mol.Electronics.KS_Hamiltonian_alpha
    olm = mol.Electronics.OLM
    Wan_orbs = np.load(path + Wan_file)

    transfer_ints = []
    for cell_vector in neighbours:
        T_ops = []
        for direc in [0, 1, 2]:
            if periodic_bool[direc]:
                Transop = mol.Electronics.electronic_symmetry.Symmetry_Generators["t" + str(direc + 1)]
                if cell_vector[direc] == 0.0:
                    T_ops.append(np.identity(olm.shape[0]))
                elif cell_vector[direc] < 0.0:
                    T_ops.append(np.linalg.matrix_power(Transop.T, abs(int(cell_vector[direc]))))
                else:
                    T_ops.append(np.linalg.matrix_power(Transop, abs(int(cell_vector[direc]))))
            else:
                T_ops.append(np.identity(olm.shape[0]))

        tr = Wan_orbs @ ksh @ T_ops[2] @ T_ops[0] @ T_ops[1] @ Wan_orbs.T
        transfer_ints.append(tr)
    reciprocal_lattice = np.zeros((3, 3))
    reciprocal_lattice[0, :], reciprocal_lattice[1, :], reciprocal_lattice[2, :] = get_reciprocal_lattice_vectors(
        *prim_unit_cellvectors
    )

    ka = np.linspace(-0.5, 0.5, num=40, endpoint=True)
    kb = np.linspace(0.5, -0.5, num=40, endpoint=True)
    kc = np.linspace(-0.0, 0.0, num=40, endpoint=True)
    k_points = []
    for i in range(len(ka)):
        k_points.append(np.array([ka[i], kb[i], kc[i]]) @ reciprocal_lattice)

    TB_bands = np.zeros((Wan_orbs.shape[0], int(len(k_points))), dtype="complex128")
    q = 0
    for k in k_points:
        TBH = np.zeros((Wan_orbs.shape[0], Wan_orbs.shape[0]), dtype="complex128")
        k = np.array(k)
        for cell_index in range(len(neighbours)):
            TBH += np.exp((1.0j) * k @ (neighbours[cell_index] @ prim_unit_cellvectors)) * transfer_ints[cell_index]
        eigvals = np.linalg.eigvalsh(TBH)
        TB_bands[:, q] = eigvals
        q += 1

    if np.max(np.imag(TB_bands)) < 10**-10:
        TB_bands = np.real(TB_bands)
    TB_bands = 27.211 * TB_bands

    k_dist = [0]
    k_dist.extend(np.cumsum(np.linalg.norm(np.diff((np.array(k_points)), axis=0), axis=1)))
    k_distance = np.array(k_dist)

    # ------------------------------------------------------------------------------------------------------------------------------
    bsfilename = "graphPBE.bs"
    nh = 5
    nl = 3
    import re

    file = open(path + bsfilename)
    content = file.read()
    set_separator = "\n# Set"
    sets_linewise = [part.rstrip().split("\n") for part in content.split(set_separator)]
    sets = content.split(set_separator)
    pattern = r"(\d+)\s+k-points.*?(\d+)\s+bands"
    point_separator = "\n#  Point"
    special_k, kpoints_on_path, band_energies_on_path, occupations_on_path = [], [], [], []
    for num_set in range(len(sets)):
        points_in_line, num_of_bands = (
            int(re.search(pattern, sets_linewise[num_set][0]).group(1)),
            int(re.search(pattern, sets_linewise[num_set][0]).group(2)),
        )
        special_k.append(
            [np.array([sets_linewise[num_set][j].split()[i] for i in [4, 5, 6]]).astype(float) for j in [1, 2, 3]]
        )
        del sets_linewise[num_set][:4]
        set_reduced = sets[num_set].split(point_separator, 1)[1]
        points = [part.split("\n") for part in set_reduced.split(point_separator)]
        kpath = []
        this_line_energies, this_line_occupations = (
            np.zeros((points_in_line, num_of_bands)),
            np.zeros((points_in_line, num_of_bands)),
        )
        for point in range(points_in_line):
            kpath.append([np.array(points[point][0].split()[i]).astype(float) for i in [3, 4, 5]])
            del points[point][:2]
            for band in range(num_of_bands):
                this_line_energies[point, band], this_line_occupations[point, band] = (
                    float(points[point][band].split()[1]),
                    float(points[point][band].split()[2]),
                )
        kpoints_on_path.append(kpath)
        band_energies_on_path.append(this_line_energies)
        occupations_on_path.append(this_line_occupations)
    file.close()

    cell_vectors = mol.cellvectors
    periodicity = mol.periodicity
    primitive_cell_vectors = [cell_vectors[i] / periodicity[i] for i in range(3)]
    reciprocal_lattice = np.zeros((3, 3))
    reciprocal_lattice[0, :], reciprocal_lattice[1, :], reciprocal_lattice[2, :] = get_reciprocal_lattice_vectors(
        *primitive_cell_vectors
    )
    conversion_factor = ConversionFactors["a.u.->A"]
    ksh = mol.Electronics.KS_Hamiltonian_alpha

    os.makedirs(path + "Testing_wantb", exist_ok=True)
    for path_index, kpath in enumerate(special_k):
        direction_vector = (kpath[0] * conversion_factor @ reciprocal_lattice) - (
            kpath[-1] * conversion_factor @ reciprocal_lattice
        )
        occ_matrix = occupations_on_path[path_index]
        vb_index = (
            np.where(occupations_on_path[path_index][0, :] < 0.95)[0][0] - 1
        )  # Assuming the Fermi level is in the gap and no partial occupations!
        e_fermi = np.max(band_energies_on_path[path_index][:, vb_index])
        k_dist = [0]
        k_dist.extend(
            np.cumsum(
                np.linalg.norm(
                    np.diff(((kpoints_on_path[path_index] @ reciprocal_lattice) * conversion_factor), axis=0), axis=1
                )
            )
        )
        k_dist = np.array(k_dist)
        if nh > vb_index:
            nh = vb_index + 1
        chosen_indices = [vb_index - (nh - 1), vb_index + (nl + 1)]
        chosen_dft_bands = band_energies_on_path[path_index][:, chosen_indices[0] : chosen_indices[1]]  # -e_fermi
        plt.figure()

        for dft_band_num in range(chosen_dft_bands.shape[1]):
            plt.plot(k_dist, chosen_dft_bands[:, dft_band_num], color="Blue")
        for band in range(Wan_orbs.shape[0]):
            plt.scatter(k_distance, TB_bands[band, :], color="red", s=10)

        plt.axhline(y=0, color="gray", linestyle="--", linewidth=0.75)
        plt.xticks([0, k_dist[int(np.floor(float(k_dist.shape[0] * 0.5)))], k_dist[int(k_dist.shape[0] - 1)]], kpath)
        figname = path + "Testing_wantb/" + str(path_index) + ".png"
        plt.savefig(figname, dpi=600)
        plt.close()
    return None


def fragmentize(orbs, frags, content, num_basis_sup):
    indices = []
    for f in range(len(frags)):
        j = 0
        frag_index = np.zeros(num_basis_sup)
        for i in range(8, int(num_basis_sup) + 1):
            line1 = content[i]
            if len(line1.split()) > 2:
                if len(line1.split()) == 9 or len(line1.split()) > 9:
                    # putting 8 or 9 here will change depending on your MolOrb file
                    # Needs to be generalised
                    if int(line1.split()[2]) in frags[f]:
                        frag_index[j] = 1
                elif len(line1.split()) == 8:
                    if int(line1.split()[1]) in frags[f]:
                        frag_index[j] = 1
                j += 1

        indices.append(frag_index)

    trialorbs = np.zeros((int(np.shape(indices[0])[0]), len(indices)))
    for o in range(len(indices)):
        trialorbs[:, o] = np.multiply(orbs[o], indices[o])
    return trialorbs


def simple_qpath(points, unit_vectors, origin=np.array([0.0, 0.0, 0.0])):
    covered_points = []
    final_sequence = []
    covered_points.append(origin)
    for vector_index in range(len(unit_vectors)):
        for direction in [-1, 1]:
            new_point = origin + (direction * unit_vectors[vector_index])
            if np.any(np.all(np.isclose(np.array(points), new_point, atol=1e-7), axis=1)) and not np.any(
                np.all(np.isclose(np.array(covered_points), new_point, atol=1e-7), axis=1)
            ):
                covered_points.append(new_point)
                final_sequence.append([origin, new_point, vector_index, direction])

    if len(covered_points) < len(points):
        for new_origin in covered_points:
            for vector_index in range(len(unit_vectors)):
                for direction in [-1, 1]:
                    new_point = new_origin + (direction * unit_vectors[vector_index])
                    if np.any(np.all(np.isclose(np.array(points), new_point, atol=1e-7), axis=1)) and not np.any(
                        np.all(np.isclose(np.array(covered_points), new_point, atol=1e-7), axis=1)
                    ):
                        covered_points.append(new_point)
                        final_sequence.append([new_origin, new_point, vector_index, direction])

                        if int(len(covered_points)) == int(len(points)):
                            if np.all(
                                [
                                    np.any(
                                        np.all(np.isclose(np.array(this_point), np.array(points), atol=1e-7), axis=1)
                                    )
                                    for this_point in covered_points
                                ]
                            ):
                                return final_sequence

                        elif len(covered_points) > len(points):
                            print("SOMETHING IS WRONG!")

    elif len(covered_points) == len(points):
        if np.all(
            [
                np.any(np.all(np.isclose(np.array(this_point), np.array(points), atol=1e-7), axis=1))
                for this_point in covered_points
            ]
        ):
            return final_sequence


# Example check for q_path:
# list_of_points = [np.array([-0.5,0.0,0.0]),np.array([-0.25,0.0,0.0]), np.array([0.0,0.0,0.0]),np.array([0.25,0.0,0.0]),np.array([-0.5,0.4,0.0]),np.array([-0.25,0.4,0.0]), np.array([0.0,0.4,0.0]),np.array([0.25,0.4,0.0])]
# unit_vectors = [np.array([0.25,0.0,0.0]),np.array([0.0,0.4,0.0])]
# print(list_of_points)
# c,_=simple_qpath(list_of_points,unit_vectors)
# print(c)


def get_reciprocal_lattice_vectors(a1, a2, a3):
    """
    Calculate reciprocal lattice vectors from direct lattice vectors.

    Parameters:
    -----------
    a1, a2, a3 : array-like, shape (3,)
        Direct lattice vectors in real space

    Returns:
    --------
    b1, b2, b3 : numpy.ndarray, shape (3,)
        Reciprocal lattice vectors

    Notes:
    ------
    The reciprocal lattice vectors are defined by:
    b1 = 2π * (a2 × a3) / (a1 · (a2 × a3))
    b2 = 2π * (a3 × a1) / (a1 · (a2 × a3))
    b3 = 2π * (a1 × a2) / (a1 · (a2 × a3))

    The factor of 2π ensures that aᵢ · bⱼ = 2π δᵢⱼ
    """
    # Convert to numpy arrays
    a1 = np.array(a1, dtype=float)
    a2 = np.array(a2, dtype=float)
    a3 = np.array(a3, dtype=float)

    # Calculate the volume of the unit cell (scalar triple product)
    volume = np.dot(a1, np.cross(a2, a3))

    if np.abs(volume) < 1e-12:
        raise ValueError("Lattice vectors are coplanar (volume = 0). Cannot form 3D lattice.")

    # Calculate reciprocal lattice vectors
    b1 = 2 * np.pi * np.cross(a2, a3) / volume
    b2 = 2 * np.pi * np.cross(a3, a1) / volume
    b3 = 2 * np.pi * np.cross(a1, a2) / volume

    return b1, b2, b3


def incl_kpoints(N):
    klist = []
    for kx in range(-int(np.floor(N / 2)), int(np.ceil(N / 2))):
        klist.append(kx / N)
    return klist


def get_q_points(mol, return_format="combined"):
    """
    Generate k-point grid in reciprocal space for periodic systems.

    Parameters:
    -----------
    mol : molecule object
        Must have attributes: cellvectors, periodicity
    return_format : str, optional
        - 'combined': returns all k-points as single list (default)
        - 'separate': returns separate lists for each direction

    Returns:
    --------
    Depends on return_format:
    - 'combined': list of k-point vectors [k1+k2+k3 combinations]
    - 'separate': tuple (qs1, qs2, qs3) of individual direction k-points
    """

    cell_vectors = mol.cellvectors
    periodicity = mol.periodicity

    # Calculate primitive cell vectors
    primitive_cell_vectors = [cell_vectors[i] / periodicity[i] for i in range(3)]

    # Get reciprocal lattice vectors
    b1, b2, b3 = get_reciprocal_lattice_vectors(*primitive_cell_vectors)

    # Generate q_points in fractional coordinates (more general alternative)
    k1s = incl_kpoints(periodicity[0])
    k2s = incl_kpoints(periodicity[1])
    k3s = incl_kpoints(periodicity[2])

    # Convert to reciprocal space coordinates with proper units
    conversion_factor = ConversionFactors["a.u.->A"]
    b = [b1, b2, b3]

    periodic_bool = np.array(mol.periodicity) > 1
    unit_vecs = []
    for i in [0, 1, 2]:
        unit_vecs.append(tuple(np.round((b[i] * conversion_factor * periodic_bool[i]) / periodicity[i], 8)))

    # Individual direction k-points
    qs1 = [list(k1s[i] * b1 * conversion_factor) for i in range(len(k1s))]
    qs2 = [list(k2s[i] * b2 * conversion_factor) for i in range(len(k2s))]
    qs3 = [list(k3s[i] * b3 * conversion_factor) for i in range(len(k3s))]

    if return_format == "separate":
        return qs1, qs2, qs3

    elif return_format == "combined":
        # Return all combinations of k-points (original behavior + generalizations)
        direct_to_cart_dict, cart_to_direct_dict = {}, {}
        q_points = []
        for k1 in k1s:
            for k2 in k2s:
                for k3 in k3s:
                    direct_q_vector = [k1, k2, k3]
                    q_vector = np.round((k1 * b1 + k2 * b2 + k3 * b3) * conversion_factor, 8)
                    cart_to_direct_dict[tuple(np.round(q_vector, 6))] = tuple(direct_q_vector)
                    direct_to_cart_dict[tuple(direct_q_vector)] = tuple(q_vector)
                    q_points.append(list(q_vector))
        return q_points, cart_to_direct_dict, direct_to_cart_dict, unit_vecs

    else:
        raise ValueError(
            f"Unknown return_format: {return_format}. Choose from 'combined', 'separate', 'grid', or 'mesh'."
        )


def group_equal_indices(values: list, start=0, end=-2) -> list[list[int]]:
    """
    Groups the indices of a list of values that are equal.

    Args:
        values: The input list of values.

    Returns:
        A list of lists, where each inner list contains the indices of
        elements with the same value in the original list.
    """
    # Choose the relevant eigenvalues based on the start and end variables
    from collections import defaultdict

    values = list(np.round(values[start : end + 1], 1))
    # Use a defaultdict to automatically handle new keys.
    # The default value for a new key will be an empty list.
    index_map = defaultdict(list)
    # Iterate through the list with both index and value.
    for index, value in enumerate(values):
        # Append the current index to the list for the corresponding value.
        index_map[value].append(index)

    indices = list(index_map.values())
    for i in range(len(indices)):
        indices[i] = np.array(indices[i]) + start

    # Return the indices
    return indices
