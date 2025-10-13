from . import Read
from . import AtomicBasis
from . import Symmetry
from . import Util
from . import Geometry
from . import Electronic_Structure
from . import TDDFT
from . import Write
from .PhysConst import ConversionFactors
import numpy as np
import matplotlib.pyplot as plt
import os
import time

def atoms_within_unit_cell(path='./'):
    # Parabola (Symmetry part) has issues processing the atoms that are outside the unit cell.
    # This function maps/'wraps' the atoms that are outside to their periodic copies that fall within the unit cell

    from ase import Atoms
    from ase.io import read, write
    xyzfile_name = Read.get_xyz_filename(path)
    xyzfile = read(xyzfile_name)
    xyzfile.set_pbc([True, True, True])
    xyzfile.wrap()
    write(xyzfile_name[:-4] + '_wrapped.xyz', images=xyzfile, format='extxyz')

    return None
def bloch_form(mol, path= './', name_of_file='general'):

    if 'Bloch_states_'+name_of_file+'.npy' in os.listdir(path):
        print('Found pre-calculated bloch states!')
        bloch_eigenstates = np.load(path+'Bloch_states_'+name_of_file+'.npy')
    else:
        ksh = mol.electronic_structure.KS_Hamiltonian_alpha
        olm = mol.electronic_structure.OLM
        prim_gam_sym, non_prim_gam_sym = separating_symmetries(mol)
        bloch_eigenstates = []

        for sym_sector in non_prim_gam_sym:
            sym_states = []
            for state in mol.electronic_structure.ElectronicEigenstates["alpha"][sym_sector]:
                sym_states.append(state.a)
            sym_states = np.array(sym_states)

            periodic_bool = np.array(mol.periodicity) > 1
            full = sym_states @ ksh @ sym_states.T
            for d in [0, 1, 2]:
                if periodic_bool[d]:
                    T_op = mol.electronic_structure.Electronic_Symmetry.Symmetry_Generators['t' + str(d + 1)]
                    full += (sym_states @ olm @ T_op @ sym_states.T)
            val, vec = np.linalg.eig(full)
            bloch = sym_states.T @ vec

            for i in range(bloch.shape[1]):
                bloch_eigenstates.append(bloch[:, i])

        bloch_eigenstates = np.array(bloch_eigenstates)
        np.save(path+'Bloch_states_'+name_of_file+'.npy', bloch_eigenstates)

    return bloch_eigenstates.T

def separating_symmetries(mol):
    import re
    # Figuring out the translational symmetries in the structure:
    periodic_bool = np.array(mol.periodicity) > 1
    periodic = periodic_bool.astype(int)

    # Isolating Symmetry sectors that correspond to the primitive gamma point
    symsecs = list(mol.electronic_structure.Electronic_Symmetry.SymSectors.keys())

    patternT = re.compile(r't(\d+)=([-+]?\d+)')
    prim_gam_sym = []
    for sym in symsecs:
        if 'Id=1' in sym:
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


def cel_periodic_overlap_calc(mol, path='./'):
    import re
    pattern = re.compile(r'^OLM_cell_per[123]\.npy$')
    already_stored_flag = False
    for file in os.listdir(path):
        if pattern.match(file):
            already_stored_flag = True

    if already_stored_flag:
        print('Found already calculated and stored phases!')
        return None
    else:
        periodic_bool = np.array(mol.periodicity) > 1
        periodic = periodic_bool.astype(int)
        basis = mol.electronic_structure.Basis
        xyz_filepath = Read.get_xyz_filename(path)
        atoms = Read.read_atomic_coordinates(xyz_filepath)
        q_points, unit_vectors = get_q_points(mol)
        cellvectors = Geometry.getNeibouringCellVectors(cell=mol.cellvectors, m=periodic[0], n=periodic[1], l=periodic[2])
        # assuming the supercell is large enough so that just the first neighbouring cells are enough; otherwise a convergence check would be needed!
        for d in [0, 1, 2]:
            if periodic_bool[d]:
                print('Calculating ', str(d+1), ' direction.')
                phase = AtomicBasis.get_phase_operators(atoms, basis, q_vector=unit_vectors[d], cell_vectors=cellvectors)
                np.save(path + 'OLM_cell_per' + str(d + 1) + '.npy', phase)
        return None

def recommended_kpath_bandstruc(mol, path='./'):
    # K-path for bandstructure plot to check the working of band indexing
    # Written in CP2K input format for ease
    periodic_bool = np.array(mol.periodicity) > 1
    dimension = np.sum(periodic_bool.astype(int))
    edges = [np.array([0.500, 0.000, 0.000]), np.array([0.000, 0.500, 0.000]), np.array([0.000, 0.000, 0.500])]
    gamma = np.array([0.000, 0.000, 0.000])
    file = open(path+'kpoint_set.txt', 'w')
    if dimension == 1:
        line = []
        line.append(-1 * edges[np.where(periodic_bool)[0][0]])
        line.append(gamma)
        line.append(edges[np.where(periodic_bool)[0][0]])
        file.writelines("&KPOINT_SET\n")
        file.writelines("  UNITS B_VECTOR\n")
        for point in line:
            file.writelines('SPECIAL_POINT ' + str(point[0]) + ' ' + str(point[1]) + ' ' + str(point[2]) + '\n')
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
                file.writelines('  SPECIAL_POINT ' + str(point[0]) + ' ' + str(point[1]) + ' ' + str(point[2]) + '\n')
            file.writelines("NPOINTS 20\n")
            file.writelines("&END KPOINT_SET\n")

    elif dimension == 3:
        set1, set2, set3, set4, set5, set6, set7, set8, set9, set10, set11, set12, set13 = [], [], [], [], [], [], [], [], [], [], [], [], []

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
        set11 = [edges[0] - 1 * edges[1] - 1 * edges[2], gamma, - 1 * edges[0] + edges[1] + edges[2]]
        set12 = [-1 * edges[0] + edges[1] - 1 * edges[2], gamma, edges[0] - 1 * edges[1] + edges[2]]
        set13 = [-1 * edges[0] - 1 * edges[1] + edges[2], gamma, edges[0] + edges[1] - 1 * edges[2]]

        for set in [set1, set2, set3, set4, set5, set6, set7, set8, set9, set10, set11, set12, set13]:
            file.writelines("&KPOINT_SET\n")
            file.writelines("  UNITS B_VECTOR\n")
            for point in set:
                file.writelines('  SPECIAL_POINT ' + str(point[0]) + ' ' + str(point[1]) + ' ' + str(point[2]) + '\n')
            file.writelines("NPOINTS 20\n")
            file.writelines("&END KPOINT_SET\n")

    file.close()
    return None

def band_index_bandstruc_check(mol, band_indexing_results, nh, nl, bsfilename,path='./'):
    import re
    file = open(path+bsfilename,'r')
    content = file.read()
    set_separator = f'\n# Set'
    sets_linewise = [part.rstrip().split('\n') for part in content.split(set_separator)]
    sets = content.split(set_separator)
    pattern = r"(\d+)\s+k-points.*?(\d+)\s+bands"
    point_separator = f'\n#  Point'
    special_k, kpoints_on_path, band_energies_on_path, occupations_on_path = [], [], [], []
    for num_set in range(len(sets)):
        points_in_line, num_of_bands = int(re.search(pattern, sets_linewise[num_set][0]).group(1)), int(re.search(pattern, sets_linewise[num_set][0]).group(2))
        special_k.append([np.array([sets_linewise[num_set][j].split()[i] for i in [4,5,6]]).astype(float) for j in [1,2,3]])
        del sets_linewise[num_set][:4]
        set_reduced = sets[num_set].split(point_separator,1)[1]
        points = [part.split('\n') for part in set_reduced.split(point_separator)]
        kpath=[]
        this_line_energies, this_line_occupations = np.zeros((points_in_line, num_of_bands)), np.zeros((points_in_line, num_of_bands))
        for point in range(points_in_line):
            kpath.append([np.array(points[point][0].split()[i]).astype(float) for i in [3,4,5]])
            del points[point][:2]
            for band in range(num_of_bands):
                this_line_energies[point,band], this_line_occupations[point,band] = float(points[point][band].split()[1]), float(points[point][band].split()[2])
        kpoints_on_path.append(kpath)
        band_energies_on_path.append(this_line_energies)
        occupations_on_path.append(this_line_occupations)
    file.close()

    cell_vectors = mol.cellvectors
    periodicity = mol.periodicity
    primitive_cell_vectors = [cell_vectors[i] / periodicity[i] for i in range(3)]
    reciprocal_lattice = np.zeros((3, 3))
    reciprocal_lattice[0,:], reciprocal_lattice[1,:], reciprocal_lattice[2,:] = get_reciprocal_lattice_vectors(*primitive_cell_vectors)
    conversion_factor = ConversionFactors["a.u.->A"]
    ksh = mol.electronic_structure.KS_Hamiltonian_alpha

    os.makedirs(path+'Testing_band_indexing',exist_ok=True)
    for band_ind, band in enumerate(band_indexing_results):
        os.makedirs(path + 'Testing_band_indexing/' + str(band_ind), exist_ok=True)
        for path_index,kpath in enumerate(special_k):

            direction_vector = (kpath[0]*conversion_factor @reciprocal_lattice ) - (kpath[-1]*conversion_factor @ reciprocal_lattice )
            occ_matrix = occupations_on_path[path_index]
            vb_index = np.where(occupations_on_path[path_index][0, :] < 0.95)[0][0] -1  # Assuming the Fermi level is in the gap and no partial occupations!
            e_fermi = np.max(band_energies_on_path[path_index][:, vb_index])
            k_dist = [0]
            k_dist.extend(np.cumsum(np.linalg.norm(np.diff( ((kpoints_on_path[path_index] @ reciprocal_lattice) *conversion_factor) ,axis=0), axis=1)))
            k_dist = np.array(k_dist)
            chosen_indices = [vb_index - (nh-1), vb_index + (nl+1)]
            chosen_dft_bands = band_energies_on_path[path_index][:,chosen_indices[0]:chosen_indices[1]]-e_fermi
            plt.figure()

            for dft_band_num in range(chosen_dft_bands.shape[1]):
                plt.plot(k_dist, chosen_dft_bands[:, dft_band_num], color='Blue')

            sampled_kpoints = list(band.keys())
            for kpoint in sampled_kpoints:
                if (np.round(np.cross(direction_vector,np.array(kpoint) - (kpath[0]*conversion_factor @reciprocal_lattice)),5) == np.array([0.,0.,0.])).all():
                    dist = np.linalg.norm(np.array(kpoint) - np.array(kpath[0]*conversion_factor @reciprocal_lattice))
                    state = band[kpoint]
                    ene = (np.real(np.conjugate(state.T) @ ksh @ state)*27.211) - e_fermi
                    plt.scatter(dist, ene)

            plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.75)
            plt.xticks([0,k_dist[int(np.floor(float(k_dist.shape[0]*0.5)))],k_dist[int(k_dist.shape[0]-1)]],kpath)
            figname = path+'Testing_band_indexing/'+ str(band_ind) +'/' +str(path_index)+'.png'
            plt.savefig(figname, dpi=600)
            plt.close()

    return None

def band_index(mol, nh, nl, name_of_bloch_file='general' ,path='./'):

    p_gamma_ind = []
    periodic_bool = np.array(mol.periodicity) > 1
    periodic = periodic_bool.astype(int)
    prim_gam_sym, non_prim_gam_sym = separating_symmetries(mol)

    # Isolating the states that correspond to the primitive BZ gamma point

    estate_dict = mol.electronic_structure.indexmap['alpha']
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


    basis = mol.electronic_structure.Basis
    xyz_filepath = Read.get_xyz_filename(path)
    atoms = Read.read_atomic_coordinates(xyz_filepath)
    q_points, unit_vectors = get_q_points(mol)
    unit_vectors = [np.array(vector) for vector in unit_vectors]
    q_points.sort(key=lambda x: np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2))
    q_arrays = [np.array(point) for point in q_points]
    q_path= simple_qpath(q_arrays, unit_vectors=unit_vectors)
    olm = mol.electronic_structure.OLM

    cellvectors = Geometry.getNeibouringCellVectors(cell=mol.cellvectors, m=periodic[0], n=periodic[1], l=periodic[2]) # assuming the supercell is large enough so that just the first neighbouring cells are enough; otherwise a convergence check would be needed!

    all_connected_bands = []
    bloch_states_full = bloch_form(mol, path=path, name_of_file=name_of_bloch_file)
    available_mask = np.ones(bloch_states_full.shape[1], dtype=bool)

    phi_q = []
    cel_periodic_overlap_calc(mol, path=path)
    for d in [0, 1, 2]:
        if periodic_bool[d]:
            phase = 'OLM_cell_per' + str(d + 1) + '.npy'
            phi_q.append(phase)
        else:
            phi_q.append(0)

    prim_blochs = []
    for state in prim_gam_states:
        prim_blochs.append(mol.electronic_structure.ElectronicEigenstates["alpha"][state[0]][int(state[1])].a)

    for state in prim_blochs:
        band = {}
        band[tuple(q_arrays[0])] = state
        calculated_q_points = []
        for q_index in range(0, len(q_path)):
            previous_qpoint, current_qpoint, del_q, direc = q_path[q_index][0], q_path[q_index][1], int(q_path[q_index][2]), int(q_path[q_index][3])

            if direc == +1:
                phase = np.load(path + phi_q[del_q])
            elif direc == -1:
                phase = np.conjugate(np.load(path + phi_q[del_q]).T)
            else:
                print('Some problem with the q_path calculation!', direc)

            calculated_q_points.append(previous_qpoint)
            previous_state = band[tuple(previous_qpoint)]
            third = time.time()
            if not np.any(np.all(np.isclose(np.array(calculated_q_points), current_qpoint, atol=1e-7), axis=1)):
                available_indices = np.where(available_mask)[0]
                overlaps = np.round(np.abs(np.conjugate(bloch_states_full[:, available_indices].T) @ phase @ previous_state), 6)

                # Extract the local index from the overlap array (overlaps is small)
                indices_max_local = np.where(overlaps == np.max(overlaps))[0]
                index_max_local = indices_max_local[0]

                # Find the global index and the state from the full array
                index_max_global = available_indices[index_max_local]
                max_state = bloch_states_full[:, index_max_global]

                # 4. Update the mask
                available_mask[index_max_global] = False

                #import gc
                #gc.collect()

                band[tuple(current_qpoint)] = max_state
                calculated_q_points.append(current_qpoint)

        all_connected_bands.append(band)

    return all_connected_bands

def wannierise(mol,band_index_results,frags,path='./'):
    import scipy.linalg as sp

    file1 = open(path + 'MolOrb')
    content = file1.readlines()
    olm = mol.electronic_structure.OLM
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
            print('Caution! Primitive Gamma states are imaginary!')
        prim_blochs.append(np.real(state))

    trial_orbs = fragmentize(prim_blochs,frags,content,bloch_state_size)

    Wannier_Orbs = np.zeros((trial_orbs.shape[1],bloch_state_size))
    for point in k_points:
        blochs_on_this_point = np.zeros((num_of_bands,bloch_state_size),dtype="complex128")
        for ind,band in enumerate(band_index_results):
            blochs_on_this_point[ind,:] = band[point]

        projection_mat = np.conjugate(blochs_on_this_point) @ olm @ trial_orbs
        u, s, vt = sp.svd(projection_mat)
        delta = np.zeros((np.shape(u)[0], np.shape(vt)[0]))
        np.fill_diagonal(delta, 1)
        Uk = u @ delta @ vt
        exp = 1.0
        Wannier_Orbs = Wannier_Orbs + ( exp * Uk.T @ blochs_on_this_point )

    if np.max(abs(np.imag(Wannier_Orbs))) > 10**-6:
        print('Warning, the Wannier orbitals are imaginary!')

    Wannier_Orbs = np.real(Wannier_Orbs / np.sqrt(supercell_size))

    print('Orthonormality of the Wannier orbitals:')
    print( np.max(abs( (Wannier_Orbs @ olm @ Wannier_Orbs.T) - np.eye(trial_orbs.shape[1]) )))

    return Wannier_Orbs

def wan_real_plot(mol, mode, ind=[0], Wan_npy='Wannier_orbitals.npy', path='./', N1=100, N2=100, N3=100):
    if mode.casefold() == 'q':
        supercell_size = mol.periodicity[0] * mol.periodicity[1] * mol.periodicity[2]
        Wans = np.load(path+Wan_npy)
        num_basis = Wans.shape[1]
        X = np.linspace(0, num_basis, num=num_basis)
        pos = np.linspace(0, num_basis, num=supercell_size+1, endpoint=True)
        for orb in ind:
            plt.plot(X,Wans[orb,:])
        plt.xticks(pos)
        plt.grid()
        plt.show()
    elif mode.casefold() == 'a':
        periodic_bool = np.array(mol.periodicity) > 1
        periodic = periodic_bool.astype(int)
        cellvectors = Geometry.getNeibouringCellVectors(cell=mol.cellvectors, m=periodic[0], n=periodic[1], l=periodic[2])  # assuming the supercell is large enough so that just the first neighbouring cells are enough; otherwise a convergence check would be needed!
        data , gridx, gridy, gridz, atoms = TDDFT.WFNsOnGrid(ids=ind, N1=N1, N2=N2, N3=N3, cell_vectors=cellvectors, wannier_printflag=True, saveflag=False, parentfolder=path)
        for orb_index,orb in enumerate(ind):
            Write.write_cube_file(gridx, gridy, gridz, data[orb_index,:,:,:], atoms, filename='w'+str(orb)+'.cube', parentfolder=path)

    else:
        print('Please specify either q or a in mode for the quick&dirty mode or accurate mode respectively')
    return None

def wan_interpolate_bandstruc(mol, special_k_points, Wan_file='Wannier_orbitals.npy', path='./'):
    periodic_bool = np.array(mol.periodicity) > 1
    periodic = periodic_bool.astype(int)
    ksh = mol.electronic_structure.KS_Hamiltonian_alpha
    olm = mol.electronic_structure.OLM
    Wan_orbs = np.load(path+Wan_file)
    onsite = Wan_orbs @ ksh @ Wan_orbs.T
    transfer_ints = []
    for d in [0, 1, 2]:
        if periodic_bool[d]:
            T_op = mol.electronic_structure.Electronic_Symmetry.Symmetry_Generators['t' + str(d + 1)]
            tr = Wan_orbs @ ksh @ T_op @ Wan_orbs.T
            transfer_ints.append(tr)
        else:
            transfer_ints.append(np.zeros(onsite.shape[0]))

    prim_unit_cellvectors = [ mol.cellvectors[d]/mol.periodicity[d] for d in [0,1,2] ]

    # Following needs to be generalised for more dimensions..
    ka = np.linspace(0, np.pi / prim_unit_cellvectors[0][0], num=40, endpoint=True)
    k_points = []
    for i in range(len(ka)):
        k_points.append((ka[i], 0.0, 0.0))

    TB_bands = np.zeros((Wan_orbs.shape[0], int(len(k_points))))
    q = 0
    for k in k_points:
        TBH = np.array(onsite.copy(), dtype="complex128")
        k = np.array(k)
        TBH += np.exp((1.0j) * k @ prim_unit_cellvectors[0]) * transfer_ints[0] + np.exp((1.0j) * k @ prim_unit_cellvectors[1]) * transfer_ints[1] + np.exp((1.0j) * k @ prim_unit_cellvectors[2]) * transfer_ints[2]
        TBH += np.exp((-1.0j) * k @ prim_unit_cellvectors[0]) * transfer_ints[0].T + np.exp((-1.0j) * k @ prim_unit_cellvectors[1]) * transfer_ints[1].T + np.exp((-1.0j) * k @ prim_unit_cellvectors[2]) * transfer_ints[2].T
        eigvals = np.linalg.eigvalsh(TBH)
        TB_bands[:, q] = eigvals
        q += 1

    TB_bands = 27.211 * TB_bands

    k_dist = [0]
    k_dist.extend(np.cumsum(np.linalg.norm(np.diff((np.array(k_points)), axis=0),axis=1)))
    k_dist = np.array(k_dist)

    for band in range(Wan_orbs.shape[0]):
        plt.plot(k_dist,TB_bands[band,:])

    plt.show()
    return None
def fragmentize(orbs,frags,content,num_basis_sup):
    indices = []
    for f in range(len(frags)):
        j = 0
        frag_index = np.zeros(num_basis_sup)
        for i in range(8, int(num_basis_sup)+1):
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
        trialorbs[:,o] = np.multiply(orbs[o], indices[o])
    return trialorbs

def simple_qpath(points, unit_vectors, origin=np.array([0.0,0.0,0.0])):
    covered_points = []
    final_sequence = []
    covered_points.append(origin)
    for direction in [-1,1]:
        for vector_index in range(len(unit_vectors)):
            new_point = origin + (direction * unit_vectors[vector_index])
            if np.any(np.all(np.isclose(np.array(points), new_point, atol=1e-7), axis=1)) and not np.any(np.all(np.isclose(np.array(covered_points), new_point, atol=1e-7), axis=1)):
                covered_points.append(new_point)
                final_sequence.append([origin,new_point,vector_index,direction])

    if len(covered_points) < len(points):
        for new_origin in covered_points:
            for direction in [-1, 1]:
                for vector_index in range(len(unit_vectors)):
                    new_point = new_origin + (direction * unit_vectors[vector_index])
                    if np.any(np.all(np.isclose(np.array(points), new_point, atol=1e-7), axis=1)) and not np.any(np.all(np.isclose(np.array(covered_points), new_point, atol=1e-7), axis=1)):
                        covered_points.append(new_point)
                        final_sequence.append([new_origin, new_point, vector_index, direction])

                        if int(len(covered_points)) == int(len(points)):
                            if np.all([np.any(np.all(np.isclose(np.array(this_point), np.array(points), atol=1e-7), axis=1)) for this_point in covered_points]):
                                return final_sequence

                        elif len(covered_points) > len(points):
                            print('SOMETHING IS WRONG!')

    elif len(covered_points) == len(points):
        if np.all([np.any(np.all(np.isclose(np.array(this_point), np.array(points), atol=1e-7), axis=1)) for this_point in covered_points]):
            return final_sequence

# Example check for q_path:
#list_of_points = [np.array([-0.5,0.0,0.0]),np.array([-0.25,0.0,0.0]), np.array([0.0,0.0,0.0]),np.array([0.25,0.0,0.0]),np.array([-0.5,0.4,0.0]),np.array([-0.25,0.4,0.0]), np.array([0.0,0.4,0.0]),np.array([0.25,0.4,0.0])]
#unit_vectors = [np.array([0.25,0.0,0.0]),np.array([0.0,0.4,0.0])]
#print(list_of_points)
#c,_=simple_qpath(list_of_points,unit_vectors)
#print(c)


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

    if return_format == 'separate':
        return qs1, qs2, qs3

    elif return_format == 'combined':
        # Return all combinations of k-points (original behavior + generalizations)
        q_points = []
        for k1 in k1s:
            for k2 in k2s:
                for k3 in k3s:
                    q_vector = np.round((k1 * b1 + k2 * b2 + k3 * b3) * conversion_factor, 8)
                    q_points.append(list(q_vector))
        return q_points, unit_vecs

    else:
        raise ValueError(f"Unknown return_format: {return_format}. "
                         "Choose from 'combined', 'separate', 'grid', or 'mesh'.")
