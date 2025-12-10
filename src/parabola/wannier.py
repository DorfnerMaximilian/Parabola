from . import Read
from . import electronics
from . import atomic_basis
from .PhysConst import ConversionFactors
from . import Geometry
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sp
import re
import os
import math
import time
from collections import defaultdict
from . import TDDFT
from . import Write


def separating_symmetries(mol):
    # Separates the symmetry labels that would correspond to the gamma point of the primitive brillouin zone

    # Figuring out the translational symmetries in the structure:
    periodic_bool = np.array(mol.periodicity) > 1
    periodic = periodic_bool.astype(int)

    # Isolating Symmetry sectors that correspond to the primitive gamma point
    symsecs = list(mol.Electronics.electronic_symmetry.SymSectors.keys())

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

def recommended_kpath_bandstruc(mol, write_flag=True, path='./'):
    # K-path for bandstructure plot to check the working of band indexing or Wannier interpolatd (Tight-binding model)
    # Written in CP2K input format.

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
        full_path = [line]

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
        full_path =[set1, set2, set3, set4]

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

        set4new = [np.array([0.000, -0.500, 0.250]),gamma,np.array([0.000, 0.500, -0.250])]
        set5new = [np.array([0.000, -0.500, -0.250]), gamma, np.array([0.000, 0.500, 0.250])]
        set6new = [np.array([0.000, -0.500, -0.500]), gamma, np.array([0.000, 0.500, 0.500])]
        set7new = [np.array([-0.500, 0.000, 0.2500]), gamma, -1*np.array([-0.500, 0.000, 0.2500])]
        set8new = [np.array([-0.500, 0.000, -0.2500]), gamma, -1*np.array([-0.500, 0.000, -0.2500])]
        set9new = [np.array([-0.500, 0.000, -0.500]), gamma, -1*np.array([-0.500, 0.000, -0.500])]
        set10new = [np.array([-0.500, -0.500, 0.00]), gamma, -1*np.array([-0.500, -0.500, 0.00])]
        set11new = [np.array([-0.500, -0.500, 0.2500]), gamma, -1*np.array([-0.500, -0.500, 0.2500])]
        set12new = [np.array([-0.500, -0.500, -0.2500]), gamma, -1*np.array([-0.500, -0.500, -0.2500])]
        set13new = [np.array([-0.500, -0.500, -0.500]), gamma, -1 * np.array([-0.500, -0.500, -0.500])]
        full_path = [set1, set2, set3, set4new, set5new, set6new, set7new, set8new, set9new, set10new, set11new, set12new, set13new]

    if write_flag:
        for set in full_path:
            file.writelines("&KPOINT_SET\n")
            file.writelines("  UNITS B_VECTOR\n")
            for point in set:
                file.writelines('  SPECIAL_POINT ' + str(point[0]) + ' ' + str(point[1]) + ' ' + str(point[2]) + '\n')
            file.writelines("NPOINTS 20\n")
            file.writelines("&END KPOINT_SET\n")

    file.close()
    return full_path

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

    # Generate q_points in fractional coordinates
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
        direct_to_cart_dict,cart_to_direct_dict = {}, {}
        q_points  = []
        for k1 in k1s:
            for k2 in k2s:
                for k3 in k3s:
                    direct_q_vector = [k1,k2,k3]
                    q_vector = np.round((k1 * b1 + k2 * b2 + k3 * b3) * conversion_factor, 8)
                    cart_to_direct_dict[tuple(np.round(q_vector,6))]  = tuple(direct_q_vector)
                    direct_to_cart_dict[tuple(direct_q_vector)] = tuple(q_vector)
                    q_points.append(list(q_vector))
        return q_points, cart_to_direct_dict, direct_to_cart_dict, unit_vecs

    else:
        raise ValueError(f"Unknown return_format: {return_format}. "
                         "Choose from 'combined', 'separate', 'grid', or 'mesh'.")


"""
def testing_bloch(mol, path= './', name_of_file='general'):
    if 'Bloch_states_'+name_of_file+'.npy' in os.listdir(path):
        print('Found pre-calculated bloch states!')
        bloch_eigenstates_full = np.load(path+'Bloch_states_'+name_of_file+'.npy')

    else:
        ksh = mol.Electronics.KS_Hamiltonian_alpha
        olm = mol.Electronics.OLM
        olmm12 = np.real(mol.Electronics.inverse_sqrt_OLM)
        periodic_bool = np.array(mol.periodicity) > 1
        prim_gam_sym, non_prim_gam_sym = separating_symmetries(mol)
        periodic_bool = np.array(mol.periodicity) > 1
        ksh_orth = olmm12 @ ksh @ olmm12
        for sym_sector in non_prim_gam_sym:
            sym_states_A = mol.Electronics.real_eigenstates["alpha"][sym_sector].T
            print(sym_sector)
            #diagmat = sym_states_A @ ksh_orth @ sym_states_A.T
            # is really diagonal! Nice!
            #plt.matshow(np.real(diagmat))
            #plt.colorbar()
            #plt.show()

            full_matrix = sym_states_A @ ksh_orth @ sym_states_A.T
            #print('hermitian/symmetric?', np.max(abs(full_matrix - full_matrix.T)))
            T1 = mol.Electronics.electronic_symmetry.Symmetry_Generators['t1']
            T2 = mol.Electronics.electronic_symmetry.Symmetry_Generators['t2']

            #print('commute check', np.max(abs(olm @ T1 - T1 @ olm)), np.max(abs(T2 @ T1 - T1 @ T2)), np.max(abs(ksh_orth @ T1 - T1 @ ksh_orth)), np.max(abs(ksh_orth @ T2 - T2 @ ksh_orth)))
            # They commute! Nice!
            for d in [0, 1, 2]:
                if periodic_bool[d]:
                    T_op = mol.Electronics.electronic_symmetry.Symmetry_Generators['t' + str(d + 1)]
                    full_matrix += ( np.random.randn() * sym_states_A @  (T_op + T_op.T) @ sym_states_A.T)
            #full_matrix += sym_states_A @ (T1+T1.T) @ sym_states_A.T
            #full_matrix += sym_states_A @ (T2+T2.T) @ sym_states_A.T

            print('hermitian/symmetric still?', np.max(abs( full_matrix - full_matrix.T )))
            #print('unitary or Normal ?', np.max(abs( full_matrix @ full_matrix.T - np.eye(full_matrix.shape[0]) )) , np.max(abs( full_matrix @ full_matrix.T - full_matrix.T @ full_matrix )))

            val, vec = np.linalg.eigh(full_matrix)
            par_bloch_states = sym_states_A.T @ vec
            #print(par_bloch_states.shape, vec.shape)
            print('eigenvecs are orthonormal?', np.max(abs(vec.conj().T @ vec - np.eye(vec.shape[1]))), np.max(abs(par_bloch_states.conj().T @ par_bloch_states - np.eye(par_bloch_states.shape[1]))))
            print('eigenvecs form unitary matrix? ', np.max(abs(vec @ vec.conj().T - np.eye(vec.shape[0]))))

            #eig = par_bloch_states.conj().T @ T1 @ par_bloch_states
            #print(np.max(abs(np.imag(eig))))
            #plt.matshow(np.abs(eig))
            #plt.matshow(np.real(eig))
            #plt.colorbar()
            #plt.show()

            bloch_states = par_bloch_states.astype(np.complex128).copy()
            for d in [0, 1, 2]:
                if periodic_bool[d]:
                    T_op = mol.Electronics.electronic_symmetry.Symmetry_Generators['t' + str(d + 1)]
                    eig = np.real(bloch_states.conj().T @ T_op @ bloch_states)
                    blocks = electronics.detect_block_sizes(eig, tol=1e-2)
                    for block in blocks:
                        i, size = block
                        if size == 1:
                            pass
                        else:
                            this_block_bloch = par_bloch_states[:, i: i + size].conj().T @ T_op @ par_bloch_states[:, i: i + size]

                            val, vec = np.linalg.eig(this_block_bloch)
                            if np.max(np.abs(np.conjugate(vec).T @ vec - np.eye(vec.shape[0]))) > 1e-5:
                                print('this block normal?', np.max(abs(this_block_bloch @ this_block_bloch.T - this_block_bloch.T @ this_block_bloch)), this_block_bloch.shape, size)
                                print('direction', d+1)
                                print('bloch transformation not unitary',np.max(np.abs(np.conjugate(vec).T @ vec - np.eye(vec.shape[0]))))#, par_bloch_states[:, i: i + size].conj().T @ ksh_orth @ par_bloch_states[:, i: i + size] * 27.211)
                                print(np.round(vec,4),np.round(this_block_bloch,4))
                            bloch_states[:, i: i + size] = par_bloch_states[:, i: i + size] @ vec

            eig = bloch_states.conj().T @ T1 @ bloch_states
            eig_diag = np.diagonal(eig)
            print('off_diag:', np.max(abs( eig - np.diag(eig_diag))))
            plt.matshow(np.abs(eig - np.diag(eig_diag)))
            plt.colorbar()
            plt.show()
    return None
"""
def bloch_from_scratch(mol, path='./'):
    if 'Bloch_states.npy' in os.listdir(path):
        print('Found pre-calculated bloch states!')
        bloch_states = np.load(path+'Bloch_states.npy')

    else:
        ksh = mol.Electronics.KS_Hamiltonian_alpha
        olm = mol.Electronics.OLM
        sm12 = np.real(mol.Electronics.inverse_sqrt_OLM)
        ksh_orth = sm12 @ ksh @ sm12
        periodic_bool = np.array(mol.periodicity) > 1

        Trans_ops = []
        for d in [0, 1, 2]:
            if periodic_bool[d]:
                Trans_ops.append(mol.Electronics.electronic_symmetry.Symmetry_Generators[f't{d + 1}'])
            else:
                Trans_ops.append(None)

        try:
            symmH = np.load(path + 'symmetricH.npy')
        except FileNotFoundError:
            print("Generating Symmetric Hamiltonian...")
            symmH = ksh_orth.copy()

            for d, counts in enumerate(mol.periodicity):
                if not periodic_bool[d] or Trans_ops[d] is None:
                    continue

                T_base = Trans_ops[d]
                H_accum = np.zeros_like(symmH)

                T_base_dag = T_base.conj().T

                current_rotated_H = symmH.copy()
                H_accum += current_rotated_H

                for k in range(1, counts):
                    current_rotated_H = T_base_dag @ current_rotated_H @ T_base
                    H_accum += current_rotated_H

                symmH = H_accum / counts

            #symmH = 0.5 * (symmH + symmH.conj().T)  # making it hermitian by force
            np.save(path + 'symmetricH.npy', symmH)

        #T1 = mol.Electronics.electronic_symmetry.Symmetry_Generators['t1']
        #T2 = mol.Electronics.electronic_symmetry.Symmetry_Generators['t2']
        #T3 = np.eye(T1.shape[0])
        #print('ksh commutes? before', np.max(abs(ksh @ T1 - T1 @ ksh)), np.max(abs(ksh @ T2 - T2 @ ksh)), np.max(abs(ksh @ T3 - T3 @ ksh)))
        #print('ksh commutes? after', np.max(abs(symmH @ T1 - T1 @ symmH)),  np.max(abs(symmH @ T2 - T2 @ symmH)), np.max(abs(symmH @ T3 - T3 @ symmH)))
        #print('symmH hermitian?', np.max(abs( symmH - symmH.T )))

        vals, vecs = np.linalg.eigh(symmH)
        print('eigenvecs are orthonormal?', np.max(abs(vecs.conj().T @ vecs - np.eye(vecs.shape[1]))))

        bloch_states = vecs.astype(np.complex128).copy()
        for d in [0, 1, 2]:
            if periodic_bool[d]:
                T_op = mol.Electronics.electronic_symmetry.Symmetry_Generators['t' + str(d + 1)]
                trans_phase = bloch_states.conj().T @ T_op @ bloch_states
                blocks = electronics.detect_block_sizes(np.abs(trans_phase), tol=2e-4)
                for block in blocks:
                    i, size = block
                    if size == 1:
                        pass
                    else:
                        if not size == 2:
                            print(size)

                        this_block_bloch = trans_phase[i: i + size, i: i + size]
                        val, vec = np.linalg.eig(this_block_bloch)
                        if np.max(np.abs(np.conjugate(vec).T @ vec - np.eye(vec.shape[0]))) > 1e-5:
                            #print('this block normal?', np.max(abs(this_block_bloch @ this_block_bloch.T - this_block_bloch.T @ this_block_bloch)), this_block_bloch.shape, size)
                            print('bloch transformation not unitary', np.max(np.abs(np.conjugate(vec).T @ vec - np.eye(vec.shape[0]))))  # , par_bloch_states[:, i: i + size].conj().T @ ksh_orth @ par_bloch_states[:, i: i + size] * 27.211)
                            print(np.round(vec, 5))
                            print(np.round(this_block_bloch, 5))
                        bloch_states[:, i: i + size] = bloch_states[:, i: i + size] @ vec

        #T1 = mol.Electronics.electronic_symmetry.Symmetry_Generators['t3']
        #T2 = mol.Electronics.electronic_symmetry.Symmetry_Generators['t2']
        #eig1 = bloch_states.conj().T @ T1 @ bloch_states
        #eig_diag1 = np.diagonal(eig1)
        #eig2 = bloch_states.conj().T @ T2 @ bloch_states
        #eig_diag2 = np.diagonal(eig2)
        #print('off_diag:', np.max(abs(eig1 - np.diag(eig_diag1))), np.max(abs(eig2 - np.diag(eig_diag2))))

        bloch_states = (sm12 @ bloch_states)   # Now converted into non-orthogonal GTO basis
        print('non-orth orthogonal or not?', np.max(abs( bloch_states.conj().T @ olm @ bloch_states  - np.eye(olm.shape[0]))))
        np.save(path+'Bloch_states.npy', bloch_states)

    return bloch_states

def separate_k(mol, bloch_states, threshold=2e-4):
    # Can also be used to check if the bloch states have been made correctly!

    periodic_bool = np.array(mol.periodicity) > 1
    olm = np.real(mol.Electronics.OLM)

    eigs = []
    for d in [0, 1, 2]:
        if periodic_bool[d]:
            T_op = mol.Electronics.electronic_symmetry.Symmetry_Generators['t' + str(d + 1)]
            eig = bloch_states.conj().T @ olm @ T_op @ bloch_states
            eig_diag = np.diagonal(eig)
            eigs.append(eig_diag)

            if np.max(np.abs(eig - np.diag(eig_diag))) > threshold:
                print('Bloch states have not been correctly made!!')
                plt.matshow(np.abs(eig-np.diag(eig_diag)))
                plt.colorbar()
                plt.show()
        else:
            eigs.append((1.0+0.0j)*np.ones(bloch_states.shape[1]))

    full_phase = [[eigs[0][i],eigs[1][i],eigs[2][i]] for i in range(bloch_states.shape[1])]
    bloch_k_points = []
    for eigval in full_phase:
        k = []
        for direc in eigval:
            k.append(np.round((math.atan2(direc.imag, direc.real) / (2 * np.pi)), 6))
        k=np.array(k)
        k[np.isclose(k, 0.5, 1e-5)] = -0.5
        k[np.isclose(k, 0.0, 1e-5)] = 0.0
        bloch_k_points.append(k)

    k_resolved_bloch_dist = {}
    for state_ind in range(bloch_states.shape[1]):
        state = bloch_states[:,state_ind]
        key = tuple(bloch_k_points[state_ind])
        if key not in k_resolved_bloch_dist:
            k_resolved_bloch_dist[key] = []
        k_resolved_bloch_dist[key].append(state)

    print(len(list(k_resolved_bloch_dist.keys())))
    for key in list(k_resolved_bloch_dist.keys()):
        print(key,len(k_resolved_bloch_dist[key]))
    return k_resolved_bloch_dist

def simple_qpath(points, unit_vectors, origin=np.array([0.0,0.0,0.0])):
    covered_points = []
    final_sequence = []
    covered_points.append(origin)
    for vector_index in range(len(unit_vectors)):
        for direction in [-1,1]:
            new_point = origin + (direction * unit_vectors[vector_index])
            if np.any(np.all(np.isclose(np.array(points), new_point, atol=1e-7), axis=1)) and not np.any(np.all(np.isclose(np.array(covered_points), new_point, atol=1e-7), axis=1)):
                covered_points.append(new_point)
                final_sequence.append([origin,new_point,vector_index,direction])

    if len(covered_points) < len(points):
        for new_origin in covered_points:
            for vector_index in range(len(unit_vectors)):
                for direction in [-1, 1]:

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

def compare_to_dft_bandstruc(mol, nh, nl, mode, bsfilename, path='./', *, band_index_results=None, k_distance=None, TB_bands=None):
    # Plots the CP2K bandstructure  by reading the .bs file obtained from CP2K bandstrucutre calc upon the k-path provided by the function recommended_kpath_bandstruc.
    # along with the energies of the bloch states chosen by band indexing or the Wannier interpolated (Tight binding) bands.
    # To plot bloch energies (for checking band indexing) choose 'b' or 'B' in mode.
    # To plot Wannier interpolated bands, choose 'w' or 'W' in mode.
    # Here, nh and nl correspond to the number of occupied or unoccupied bands to be plotted from the CP2K .bs file, respectively.

    print(path,bsfilename)
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

    if mode.casefold() == 'b':
        ksh = mol.Electronics.KS_Hamiltonian_alpha
        os.makedirs(path+'Testing_band_indexing',exist_ok=True)
        for band_ind, band in enumerate(band_index_results):
            os.makedirs(path + 'Testing_band_indexing/' + str(band_ind), exist_ok=True)
            for path_index,kpath in enumerate(special_k):

                direction_vector = ( kpath[0]*conversion_factor @reciprocal_lattice ) - ( kpath[-1]*conversion_factor @ reciprocal_lattice )
                occ_matrix = occupations_on_path[path_index]
                vb_index = np.where(occupations_on_path[path_index][0, :] < 0.95)[0][0] -1  # Assuming the Fermi level is in the gap and no partial occupations!
                e_fermi = np.max(band_energies_on_path[path_index][:, vb_index])
                k_dist = [0]
                k_dist.extend(np.cumsum(np.linalg.norm(np.diff( ((kpoints_on_path[path_index] @ reciprocal_lattice) *conversion_factor) ,axis=0), axis=1)))
                k_dist = np.array(k_dist)
                if nh > vb_index:
                    nh = vb_index+1
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

    elif mode.casefold() == 'w':
        os.makedirs(path + 'Testing_wantb', exist_ok=True)
        for path_index, kpath in enumerate(special_k):

            direction_vector = (kpath[0] * conversion_factor @ reciprocal_lattice) - (kpath[-1] * conversion_factor @ reciprocal_lattice)
            occ_matrix = occupations_on_path[path_index]
            vb_index = np.where(occupations_on_path[path_index][0, :] < 0.95)[0][0] - 1  # Assuming the Fermi level is in the gap and no partial occupations!
            e_fermi = np.max(band_energies_on_path[path_index][:, vb_index])
            k_dist = [0]
            k_dist.extend(np.cumsum(np.linalg.norm(np.diff(((kpoints_on_path[path_index] @ reciprocal_lattice) * conversion_factor), axis=0), axis=1)))
            k_dist = np.array(k_dist)
            if nh > vb_index:
                nh = vb_index + 1
            chosen_indices = [vb_index - (nh - 1), vb_index + (nl + 1)]
            chosen_dft_bands = band_energies_on_path[path_index][:, chosen_indices[0]:chosen_indices[1]] -e_fermi
            plt.figure()

            for dft_band_num in range(chosen_dft_bands.shape[1]):
                plt.plot(k_dist, chosen_dft_bands[:, dft_band_num], color='Blue')
            for band in range(TB_bands.shape[0]):
                plt.scatter(k_distance[:,path_index], TB_bands[band, :, path_index] - e_fermi, color='red', s=10)

            plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.75)
            plt.xticks([0, k_dist[int(np.floor(float(k_dist.shape[0] * 0.5)))], k_dist[int(k_dist.shape[0] - 1)]], kpath)
            figname = path + 'Testing_wantb/' + str(path_index) + '.png'
            plt.savefig(figname, dpi=600)
            plt.close()
        return None
    else:
        print('Please specify either b or w in mode for the bloch_energy plotting or Wannier_interpolated_band plotting respectively')
    return None

def isolate_prim_gam_states_scratch(mol, nh, nl, path='./'):

    Homoindex_sup = (mol.Electronics.num_e / 2 )
    supsize = mol.periodicity[0] * mol.periodicity[1] * mol.periodicity[2]
    Homoindex_prim = int((Homoindex_sup / supsize)) -1

    all_bloch_funcs = bloch_from_scratch(mol, path=path)
    starttime=time.time()
    k_resolv_dict = separate_k(mol, all_bloch_funcs)
    print('time taken to sepratek', time.time()-starttime)
    prim_gamma_states = k_resolv_dict[tuple([0.0,0.0,0.0])]

    occ_ind_list = list(np.arange(0,Homoindex_prim+1))
    occ_ind_list.reverse()
    all_prim_gam_states = []
    if nl > 0:
        for ind in range(Homoindex_prim+1, Homoindex_prim+1+nl):
            all_prim_gam_states.append(prim_gamma_states[int(ind)])

    all_prim_gam_states.reverse()

    if nh > 0:
        for ind in occ_ind_list[:nh]:
            all_prim_gam_states.append(prim_gamma_states[int(ind)])


    non_prim_gam_states = []
    for key in list(k_resolv_dict.keys()):
        if not key == tuple([0.0,0.0,0.0]):
            blochs_on_this_kpoint = k_resolv_dict[key]
            for index in range(len(blochs_on_this_kpoint)):
                non_prim_gam_states.append(blochs_on_this_kpoint[index])
    non_prim_gam_states = np.array(non_prim_gam_states)
    all_prim_gam_states = np.array(all_prim_gam_states)

    print(all_prim_gam_states.shape,non_prim_gam_states.shape)
    return all_prim_gam_states, non_prim_gam_states

def cell_periodic_overlap_calc(mol, periodic_copies, path='./'):
    # Calculates and stores the Overlap matrix for the cell periodic part of the Bloch function (in the Gaussian basis).
    # Here 1, 2 and 3 are the 'unit vectors/lattice vectors' for the k-point grid.

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
        basis = mol.Electronics.basis
        xyz_filepath = Read.get_xyz_filename(path)
        atoms = Read.read_atomic_coordinates(xyz_filepath)
        q_points, _ , _ ,unit_vectors = get_q_points(mol)
        cellvectors = Geometry.getNeibouringCellVectors(cell=mol.cellvectors, m=periodic_copies[0], n=periodic_copies[1], l=periodic_copies[2])

        for d in [0, 1, 2]:
            if periodic_bool[d]:
                print('Calculating cell periodic OLM for direction: ', str(d+1))
                time1 = time.time()
                phase = atomic_basis.get_phase_operators(atoms, basis, q_vector=unit_vectors[d], cell_vectors=cellvectors, cutoff_radius=25)
                np.save(path + 'OLM_cell_per' + str(d + 1) + '.npy', phase)
                print('time taken to make this matrix', time.time()-time1)
        return None

def fragmentize(orbs,frags,content,num_basis_sup):
    indices = []
    for f in range(len(frags)):
        frag_index = np.zeros(num_basis_sup)
        for i in range(int(num_basis_sup)):
            line1 = content[i]
            atom_index_match = int(line1.split()[2]) in frags[f][0]
            orbital_label_match = np.array([ requested_orbs in str(line1.split()[4]) for requested_orbs in frags[f][1] ]).any()
            if atom_index_match and orbital_label_match:
                frag_index[i] = 1
        indices.append(frag_index)

    trialorbs = np.zeros((int(np.shape(indices[0])[0]), len(indices)))
    for o in range(len(indices)):
        trialorbs[:,o] = np.multiply(orbs[o], indices[o])
    return trialorbs

def group_equal_indices(values: list,) -> list[list[int]]:
    # Groups the indices of a list of values that are equal.
    values = list(np.round(values, 1))
    # Use a defaultdict to automatically handle new keys.
    # The default value for a new key will be an empty list.
    index_map = defaultdict(list)
    # Iterate through the list with both index and value.
    for index, value in enumerate(values):
        # Append the current index to the list for the corresponding value.
        index_map[value].append(index)

    indices = list(index_map.values())
    for i in range(len(indices)):
        indices[i] = np.array(indices[i])

    # Return the indices
    return indices

def general_wan_interpolate_bands(mol, nh, nl, bsfilename, upto_neighbour, Wan_file='Wannier_orbitals.npy', path='./'):

    periodic_bool = np.array(mol.periodicity) > 1
    a, b, c = mol.periodicity[0], mol.periodicity[1], mol.periodicity[2]
    if a%2 == 0:
        grida = np.linspace(-np.floor(a / 2), np.floor(a / 2), a, endpoint=False)
    elif a == 1:
        grida = np.array([0])
    else:
        grida = np.linspace(-np.floor(a / 2), np.floor(a / 2), a, endpoint=True)

    if b%2 == 0:
        gridb = np.linspace(-np.floor(b / 2), np.floor(b / 2), b, endpoint=False)
    elif b == 1:
        gridb = np.array([0])
    else:
        gridb = np.linspace(-np.floor(b / 2), np.floor(b / 2), b, endpoint=True)

    if c%2 == 0:
        gridc = np.linspace(-np.floor(c / 2), np.floor(c / 2), c, endpoint=False)
    elif c == 1:
        gridc = np.array([0])
    else:
        gridc = np.linspace(-np.floor(c / 2), np.floor(c / 2), c, endpoint=True)


    print(gridc)
    conversion_factor = ConversionFactors["a.u.->A"]
    prim_unit_cellvectors = [mol.cellvectors[d] / (mol.periodicity[d] * conversion_factor) for d in [0, 1, 2]]
    inverse_prim_cell = np.linalg.inv(prim_unit_cellvectors)

    points = []
    for av in grida:
        for bv in gridb:
            for cv in gridc:
                print(np.array([av, bv, cv]))
                points.append(np.array([av, bv, cv]) @ prim_unit_cellvectors)

    points.sort(key=lambda x: np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2))
    points_dist = [np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2) for x in points]
    print(len(points))
    neighbours = [points[index] @ inverse_prim_cell for index in list(group_equal_indices(points_dist)[0])]
    for upto in range(1, upto_neighbour + 1):
        for index in list(group_equal_indices(points_dist)[upto]):
            neighbours.append(np.round(points[index] @ inverse_prim_cell, 0))
    print(len(neighbours), neighbours)
    ksh = mol.Electronics.KS_Hamiltonian_alpha
    olm = mol.Electronics.OLM
    sm12 = np.real(mol.Electronics.inverse_sqrt_OLM)
    ksh_orth = sm12 @ ksh @ sm12
    if 'OLMp12.npy' in os.listdir(path):
        print('Found Sp12!')
        sp12 = np.load(path + 'OLMp12.npy')
    else:
        sp12 = np.real(sp.fractional_matrix_power(olm, 0.5))
        np.save(path + 'OLMp12.npy', sp12)
    Wan_orbs = np.load(path + Wan_file)

    Wan_orbs_orth = []
    for orb in range(Wan_orbs.shape[0]):
        Wan_orbs_orth.append( sp12 @ Wan_orbs[orb,:])
    Wan_orbs_orth = np.array(Wan_orbs_orth)

    transfer_ints = []
    for cell_vector in neighbours:
        T_ops = []
        for direc in [0, 1, 2]:
            if periodic_bool[direc]:
                Transop = mol.Electronics.electronic_symmetry.Symmetry_Generators['t' + str(direc + 1)]
                if cell_vector[direc] == 0.0:
                    T_ops.append(np.identity(olm.shape[0]))
                elif cell_vector[direc] < 0.0:
                    T_ops.append(np.linalg.matrix_power(Transop.T,abs(int(cell_vector[direc]))))
                else:
                    T_ops.append(np.linalg.matrix_power(Transop,abs(int(cell_vector[direc]))))
            else:
                T_ops.append(np.identity(olm.shape[0]))

        tr = Wan_orbs_orth @ ksh_orth @ T_ops[2] @ T_ops[1] @ T_ops[0] @ Wan_orbs_orth.T
        transfer_ints.append(tr)

    reciprocal_lattice = np.zeros((3, 3))
    reciprocal_lattice[0, :], reciprocal_lattice[1, :], reciprocal_lattice[2, :] = get_reciprocal_lattice_vectors(*prim_unit_cellvectors)
    k_paths = recommended_kpath_bandstruc(mol,write_flag=False,path=path)
    num_points_on_path = 40
    TB_bands = np.zeros((Wan_orbs_orth.shape[0], int(num_points_on_path), int(len(k_paths))), dtype="complex128")
    k_distance = np.zeros((int(num_points_on_path), int(len(k_paths))))
    for path_index, k_path in enumerate(k_paths):
        start, end = k_path[0], k_path[-1]
        ka = np.linspace(start[0], end[0], num=num_points_on_path, endpoint=True)
        kb = np.linspace(start[1], end[1], num=num_points_on_path, endpoint=True)
        kc = np.linspace(start[2], end[2], num=num_points_on_path, endpoint=True)
        k_points_on_this_path = [ np.array([ka[i], kb[i], kc[i]]) @ reciprocal_lattice for i in range(num_points_on_path)]

        q = 0
        for k in k_points_on_this_path:
            TBH = np.zeros((Wan_orbs_orth.shape[0], Wan_orbs_orth.shape[0]), dtype="complex128")
            for cell_index in range(len(neighbours)):
                TBH += np.exp((1.0j) * k @ (neighbours[cell_index] @ prim_unit_cellvectors)) * transfer_ints[cell_index]
            eigvals = np.linalg.eigvalsh(TBH)
            TB_bands[:, q, path_index] = eigvals
            q += 1

        k_dist = [0]
        k_dist.extend(np.cumsum(np.linalg.norm(np.diff((np.array(k_points_on_this_path)), axis=0), axis=1)))
        k_distance[:,path_index] = np.array(k_dist)

    if np.max(np.imag(TB_bands)) < 10**-10:
        TB_bands = np.real(TB_bands)
    TB_bands = 27.211 * TB_bands

    compare_to_dft_bandstruc(mol, nh=nh, nl=nl, mode='w', bsfilename=bsfilename, path=path, k_distance= k_distance, TB_bands=TB_bands)

    return None

def wan_real_plot(mol, mode, periodic_copies, ind=[0], Wan_npy='Wannier_Orbitals.npy', path='./', N1=100, N2=100, N3=100):
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
        # Essentially a modified copy of TDDFT.WFNsOnGrid
        Wannier_funcs = np.load(path+Wan_npy)
        ksh = mol.Electronics.KS_Hamiltonian_alpha
        Atoms = Read.read_atomic_coordinates(Read.get_xyz_filename(path))
        Basis = atomic_basis.getBasis(path)
        # Convert to atomic units
        Cellvectors = Read.read_cell_vectors(path)
        cellvector1 = Cellvectors[0] * 1.88972613288564
        cellvector2 = Cellvectors[1] * 1.88972613288564
        cellvector3 = Cellvectors[2] * 1.88972613288564
        latticevecs = np.array([cellvector1, cellvector2, cellvector3]).T
        print('lat', latticevecs)
        neigh_cell_vectors = Geometry.getNeibouringCellVectors(cell=Cellvectors, m=periodic_copies[0], n=periodic_copies[1], l=periodic_copies[2])
        # get voxel volume in a.u.**3
        voxelvolume = np.dot(cellvector1, np.cross(cellvector2, cellvector3)) / (N1 * N2 * N3)
        # discretization
        grid1 = np.arange(N1) / N1
        grid2 = np.arange(N2) / N2
        grid3 = np.arange(N3) / N3
        data = []
        for id in ind:
            timehere = time.time()
            a = Wannier_funcs[id, :]
            if not a.shape[0] == ksh.shape[0]:
                print('Warning! the orientation of Wans might be wrong!')
            a *= TDDFT.getPhaseOfMO(a)
            f = atomic_basis.WFNonxyzGrid(grid1, grid2, grid3, a, latticevecs, Atoms, Basis, neigh_cell_vectors)
            print(voxelvolume * np.sum(np.sum(np.sum(f ** 2))))
            f /= np.sqrt(voxelvolume * np.sum(np.sum(np.sum(f ** 2))))
            data.append(f)
            print('time taken for this function:', time.time() - timehere, 'sec')

        data = np.array(data)
        #np.save(path + 'real_space_data_unoccupied_bands.npy', data)
        print('data', data.shape)

        for orb_index, orb in enumerate(ind):
            Write.write_cube_file(grid1, grid2, grid3, data[orb_index, :, :, :], cellvector1, cellvector2, cellvector3, Atoms, filename='w_occupied_' + str(orb) + '.cube', parentfolder=path)

    else:
        print('Please specify either q or a in mode for the quick&dirty mode or accurate mode respectively')
    return None


def band_index(mol, nh, nl, periodic_copies, path='./'):
    # Performs Band indexing (= choosing a set of bloch states that correspond to the same band).
    # nh and nl represent the number of primitive Gamma point states (= number of bands) to be considered.

    q_points, cart_to_direct_dict, direct_to_cart_dict, unit_vectors = get_q_points(mol)
    periodic_bool = np.array(mol.periodicity) > 1
    unit_vectors = [np.array(vector) for vector in unit_vectors]
    q_points.sort(key=lambda x: np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2))
    q_arrays = [np.array(point) for point in q_points]
    q_path= simple_qpath(q_arrays, unit_vectors=unit_vectors)

    for q_index in range(0, len(q_path)):
        previous_qpoint, current_qpoint, del_q, direc = q_path[q_index][0], q_path[q_index][1], int(q_path[q_index][2]), int(q_path[q_index][3])
        print(cart_to_direct_dict[tuple(np.round(previous_qpoint,6))], cart_to_direct_dict[tuple(np.round(current_qpoint,6))])

    all_connected_bands = []
    prim_gam_states, non_prim_gam_states = isolate_prim_gam_states_scratch(mol, nh, nl, path=path)
    available_mask = np.ones(non_prim_gam_states.shape[0], dtype=bool)
    starttime = time.time()

    phi_q = []
    cell_periodic_overlap_calc(mol,periodic_copies, path=path)
    for d in [0, 1, 2]:
        if periodic_bool[d]:
            phase = 'OLM_cell_per' + str(d + 1) + '.npy'
            phi_q.append(phase)
        else:
            phi_q.append(0)

    for state in prim_gam_states:
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
            if not np.any(np.all(np.isclose(np.array(calculated_q_points), current_qpoint, atol=1e-7), axis=1)):
                available_indices = np.where(available_mask)[0]
                overlaps = np.round(np.abs(np.conjugate(non_prim_gam_states[available_indices,:]) @ (phase @ previous_state)), 6)
                print(cart_to_direct_dict[tuple(np.round(previous_qpoint, 6))], cart_to_direct_dict[tuple(np.round(current_qpoint, 6))], np.sort(overlaps)[-5:])
                indices_max_local = np.where(overlaps == np.max(overlaps))[0]
                index_max_local = indices_max_local[0]
                index_max_global = available_indices[index_max_local]
                max_state = non_prim_gam_states[index_max_global,:]
                available_mask[index_max_global] = False
                band[tuple(current_qpoint)] = max_state
                calculated_q_points.append(current_qpoint)
        all_connected_bands.append(band)
    return all_connected_bands



def wannierise(mol,band_index_results,frags,path='./',Wannier_file_name='Wannier_Orbitals'):

    ksh = mol.Electronics.KS_Hamiltonian_alpha
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
            print('Caution! Primitive Gamma states are imaginary!')
        prim_blochs.append(np.real(state))

    file = open(path + 'MolOrb')
    content = file.readlines()
    pattern = re.compile(r'^(\s*)MO\|\s+\d+\s+\d+\s+[A-Z].*')
    filtered_content = [line.strip() for line in content if pattern.match(line)]
    trial_orbs = fragmentize(prim_blochs,frags,filtered_content,bloch_state_size)

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
        Wannier_Orbs = Wannier_Orbs + exp * Uk.T @ blochs_on_this_point

    if np.max(abs(np.imag(Wannier_Orbs))) > 10**-6:
        print('Warning, the Wannier orbitals are imaginary!. Max. imaginary component:', np.max(abs(np.imag(Wannier_Orbs))))

    Wannier_Orbs = np.real(Wannier_Orbs / np.sqrt(supercell_size))

    print('Orthonormality of the Wannier orbitals:')
    print( np.max(abs( (Wannier_Orbs @ olm @ Wannier_Orbs.T) - np.eye(trial_orbs.shape[1]) )))

    np.save(path+Wannier_file_name+'.npy',Wannier_Orbs)

    if 'OLMp12.npy' in os.listdir(path):
        sp12 = np.load(path+'OLMp12.npy')
    else:
        sp12 = np.real(sp.fractional_matrix_power(olm, 0.5))
        np.save(path+'OLMp12.npy',sp12)

    Wan_orbs_orth = []
    for orb in range(Wannier_Orbs.shape[0]):
        Wan_orbs_orth.append(sp12 @ Wannier_Orbs[orb, :])
    Wan_orbs_orth = np.array(Wan_orbs_orth)

    print('complex part', np.max(abs(np.imag(Wan_orbs_orth))), np.max(abs(np.imag(sp12))))
    print('ortho',np.max(abs((Wan_orbs_orth @ Wan_orbs_orth.T) - np.eye(trial_orbs.shape[1]))))
    T1 = mol.Electronics.electronic_symmetry.Symmetry_Generators['t2']
    print('orthog', np.max(abs((Wan_orbs_orth @ T1 @ T1 @ Wan_orbs_orth.T))))

    supercell_size = mol.periodicity[0] * mol.periodicity[1] * mol.periodicity[2]
    num_basis = Wan_orbs_orth.shape[1]
    X = np.linspace(0, num_basis, num=num_basis)
    pos = np.linspace(0, num_basis, num=supercell_size + 1, endpoint=True)
    plt.plot(X, Wan_orbs_orth[0, :], color='red')
    plt.plot(X, T1 @ Wan_orbs_orth[0, :], color='blue')
    plt.xticks(pos)
    plt.grid()
    plt.show()

    return Wannier_Orbs