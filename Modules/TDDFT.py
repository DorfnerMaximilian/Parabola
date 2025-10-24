import os
import numpy as np
import scipy as sci
from ctypes import c_char_p, cdll, POINTER, c_double, c_int
from copy import deepcopy
from . import Read
from .PhysConst import ConversionFactors
from . import Util
from . import AtomicBasis

pathtocpp_lib = "./CPP_Extension/bin/AtomicBasis.so"


# -------------------------------------------------------------------------
def getPhaseOfMO(MO):
    ## Definition of the phase convention
    ## input:   MO                 np.array(NumBasisfunctions)      Expansion coefficients of the MO in terms of Löwdin orthogonalized AOs
    ## output:  MOphases            list of integers            (list)
    ## Example: MOphases[mo_index] is the phase (in +/- 1) defined by the function below (convention)
    # The first non-vanishing element
    sumofvalues = np.sum(MO)
    # print(sumofvalues)
    if sumofvalues > 0:
        phase = 1.0
    else:
        phase = -1.0
    return phase


def getPhaseOfMOs(MOs):
    # getNumberOfBasisFkts
    dim = np.shape(MOs)[0]
    MOphases = np.zeros(dim)
    for it in range(np.shape(MOs)[1]):
        MOphases[it] = getPhaseOfMO(MOs[:, it])
    return MOphases


def getPhaseOfMOs_RealSpace(MOs, MOindices, parentfolder="./"):
    # getNumberOfBasisFkts
    dim = Util.getNumberofBasisFunctions(parentfolder)
    print(dim)
    MOphases = np.zeros(dim)
    HOMOid = Read.read_homo_index(parentfolder)
    Nx = np.shape(MOs[0, :, :, :])[0]
    Ny = np.shape(MOs[0, :, :, :])[1]
    Nz = np.shape(MOs[0, :, :, :])[2]
    MOs_Parabola = WFNsOnGrid(MOindices, Nx, Ny, Nz, False, parentfolder)
    for it, moindex in enumerate(MOindices):
        it0 = moindex + HOMOid
        overlap = getOverlapOnGrids(
            MOs_Parabola[it, :, :, :], MOs[it, :, :, :], parentfolder
        )
        print(overlap)
        if np.abs(overlap) > 0.8:
            MOphases[it0] = np.sign(overlap)
        else:
            ValueError("Overlap for Phase Determination too small!")
    return MOphases


def getMOsPhases(parentfolder="./"):
    ## Reads the Molecular Orbitals from a provided file
    ## input:   MOs                 np.array(NumBasisfunctions,Numbasisfunction)       Expansion coefficients of the MOs in terms of AO's (index 1 AO index, index2 MO index)
    ## (opt.)   filename            path to the MOs file        (string)
    ## output:  MOphases            list of integers            (list)
    ## Example: MOphases[mo_index] is the phase (in +/- 1) defined by the function below (convention)
    MOs, MOindices = Read.read_mos(parentfolder)
    _, _, S = Read.read_ks_matrices(parentfolder)
    S12 = sci.linalg.fractional_matrix_power(S, 0.5)
    a_orth = S12 @ MOs
    if len(np.shape(MOs)) == 2:
        MOphases = getPhaseOfMOs(a_orth)
    else:
        ValueError("Wrong shape of MOs!")
    return MOphases, MOindices


def getUniqueExcitedStates(minweight=1e-6, pathToExcitedStates="./", pathtoMO="./"):
    """
    Parses excited states from a TDDFPT calculation done in CP2K, determines the eta
    coefficients with respect to the MO basis (all positive phases), and normalizes
    each excited state.

    Parameters
    ----------
    minweight : float, optional
        Minimum amplitude to consider (default = 0.01).
    pathToExcitedStates : str, optional
        Path to the output file of the CP2K TDDFPT calculation.
    pathtoMO : str, optional
        Path to the molecular orbital (MO) files.

    Returns
    -------
    eta : ndarray
        Excited states represented as an NBasis x NBasis x NStates numpy array.
    Energies : ndarray
        Excited state energies (in eV).
    """
    eta_file = os.path.join(pathToExcitedStates, "eta.npy")
    energies_file = os.path.join(pathToExcitedStates, "ExcitedStateEnergies.npy")

    # Try to load precomputed results
    if os.path.exists(eta_file) and os.path.exists(energies_file):
        try:
            eta = np.load(eta_file)
            Energies = np.load(energies_file)
            print(f"Loaded eta and energies from '{pathToExcitedStates}'.")
            return eta, Energies
        except Exception as e:
            print(f"Warning: Failed to load precomputed data ({e}). Recomputing...")

    # Compute from scratch
    try:
        states = Read.read_excited_states(pathToExcitedStates, minweight)
    except Exception as e:
        raise RuntimeError(
            f"Failed to read excited states from '{pathToExcitedStates}': {e}"
        )

    try:
        dim = Util.getNumberofBasisFunctions(pathToExcitedStates)
    except Exception as e:
        raise RuntimeError(f"Failed to determine number of basis functions: {e}")

    try:
        MOsphases, _ = getMOsPhases(pathtoMO)
    except Exception as e:
        raise RuntimeError(f"Failed to get MO phases from '{pathtoMO}': {e}")

    try:
        homoid = Read.read_homo_index(pathToExcitedStates)
    except Exception as e:
        raise RuntimeError(f"Failed to get HOMO ID from '{pathToExcitedStates}': {e}")

    eta = np.zeros((dim, dim, len(states)), dtype=float)
    Energies = []

    for it, state in enumerate(states):
        try:
            excited_state_energy = state[0]
            Energies.append(excited_state_energy)

            for composition in state[1]:
                holeState = composition[0] + homoid
                particleState = composition[1] + homoid
                unique_weight = (
                    composition[2] * MOsphases[holeState] * MOsphases[particleState]
                )
                eta[particleState, holeState, it] = unique_weight
        except Exception as e:
            print(f"Warning: Skipping state {it} due to error: {e}")

    # Set global phase to be positive
    for it in range(len(states)):
        idx = np.argmax(np.abs(eta[:, :, it]))
        tup = np.unravel_index(idx, (dim, dim))
        eta[:, :, it] *= np.sign(eta[tup[0], tup[1], it])

    # Normalize states
    for it in range(len(states)):
        normalization = np.sqrt(np.tensordot(eta[:, :, it], eta[:, :, it], axes=2))
        if normalization == 0:
            print(f"Warning: State {it} has zero norm — skipping normalization.")
            continue
        eta[:, :, it] /= normalization
        print(f"State {it+1}/{len(states)} normalized. Norm = {normalization:.6f}")

    # Save results
    try:
        np.save(eta_file, eta)
        np.save(energies_file, np.array(Energies))
        print(f"Results saved to '{pathToExcitedStates}'.")
    except Exception as e:
        print(f"Warning: Failed to save results: {e}")

    return eta, np.array(Energies)


def getTransitionDipoleMomentsAnalytic(
    minweigth=0.01, pathtoMO="./", pathtoExcitedstates="./"
):
    """Function to generate a file, where the Dipolmatrixelements and the excited states are summarized
    input:   path              (string)                path to the folder, where the wavefunctions have been generated and where the .inp/outputfile of the
                                                       TDDFPT calculation lies
    output:                    (void)
    """

    energies, eta = getUniqueExcitedStates(minweigth, pathtoExcitedstates, pathtoMO)
    id_homo = Read.read_homo_index(pathtoExcitedstates)
    ### Generate the Overlap Contribution
    KSHamiltonian, _, OLM = Read.read_ks_matrices(pathtoMO)
    Sm12 = Util.LoewdinTransformation(OLM)
    S12 = sci.linalg.fractional_matrix_power(OLM, 0.5)
    KSHorth = np.dot(Sm12, np.dot(KSHamiltonian, Sm12))
    _, A = np.linalg.eigh(KSHorth)
    # Fix the Phase
    for it in range(np.shape(A)[1]):
        A[:, it] *= getPhaseOfMO(A[:, it])
    Atoms = Read.read_atomic_coordinates(pathtoExcitedstates)
    Basis = AtomicBasis.getBasis(pathtoExcitedstates)
    Rx = np.zeros(np.shape(Sm12))
    Ry = np.zeros(np.shape(Sm12))
    Rz = np.zeros(np.shape(Sm12))
    it = 0
    for itAtom in range(len(Atoms)):
        Atom_type1 = Atoms[itAtom][1]
        for itBasis1 in range(len(Basis[Atom_type1])):
            R1 = np.array(Atoms[itAtom][2:]) * ConversionFactors["A->a.u."]
            Rx[it][it] = R1[0]
            Ry[it][it] = R1[1]
            Rz[it][it] = R1[2]
            it += 1
    overlapcontribution_x = Sm12 @ Rx @ S12
    overlapcontribution_y = Sm12 @ Ry @ S12
    overlapcontribution_z = Sm12 @ Rz @ S12

    dx = np.zeros(np.shape(Sm12))
    dy = np.zeros(np.shape(Sm12))
    dz = np.zeros(np.shape(Sm12))
    it1 = 0
    it2 = 0
    for itAtom1 in range(len(Atoms)):
        Atom_type1 = Atoms[itAtom1][1]
        B1 = Basis[Atom_type1]
        for itBasis1 in range(len(Basis[Atom_type1])):
            R1 = (
                np.array(Atoms[itAtom1][2:]) * ConversionFactors["A->a.u."]
            )  # conversion from angstroem to atomic units
            state1 = B1[itBasis1]
            dalpha1 = state1[3:]
            lm1 = state1[2][1:]
            for itAtom2 in range(len(Atoms)):
                Atom_type2 = Atoms[itAtom2][1]
                B2 = Basis[Atom_type2]
                for itBasis2 in range(len(Basis[Atom_type2])):
                    # get the position of the Atoms
                    R2 = (
                        np.array(Atoms[itAtom2][2:]) * ConversionFactors["A->a.u."]
                    )  # conversion from angstroem to atomic units
                    state2 = B2[itBasis2]
                    dalpha2 = state2[3:]
                    lm2 = state2[2][1:]
                    dx[it1][it2] = AtomicBasis.getContribution(
                        R1, lm1, dalpha1, R2, lm2, dalpha2, 0
                    )
                    dy[it1][it2] = AtomicBasis.getContribution(
                        R1, lm1, dalpha1, R2, lm2, dalpha2, 1
                    )
                    dz[it1][it2] = AtomicBasis.getContribution(
                        R1, lm1, dalpha1, R2, lm2, dalpha2, 2
                    )
                    it2 += 1
            it1 += 1
            it2 = 0
    dx = Sm12 @ dx @ Sm12
    dy = Sm12 @ dy @ Sm12
    dz = Sm12 @ dz @ Sm12
    DipoleOperator_x = dx + overlapcontribution_x
    DipoleOperator_y = dy + overlapcontribution_y
    DipoleOperator_z = dz + overlapcontribution_z
    TransitionDipolevectors = []
    with open("ExcitedStatesAndDipoles.dat", "a") as file:
        file.write("Python Convention of state labeling!\n")
    for it in range(len(energies)):
        i_index, m_index = np.where(np.abs(eta[:, :, it] > minweigth))
        # generate the T matrix

        Energy = energies[it]
        dx = 0.0
        dy = 0.0
        dz = 0.0
        for id, i in enumerate(i_index):
            m = m_index[id]
            amplitude = eta[i, m, it]
            StateLabel1 = m
            StateLabel2 = i
            edx = -np.dot(A[:, StateLabel2], DipoleOperator_x @ A[:, StateLabel1])
            edy = -np.dot(A[:, StateLabel2], DipoleOperator_y @ A[:, StateLabel1])
            edz = -np.dot(A[:, StateLabel2], DipoleOperator_z @ A[:, StateLabel1])
            dx += amplitude * edx
            dy += amplitude * edy
            dz += amplitude * edz
        TransitionDipolevectors.append(np.array([dx, dy, dz]))
        with open("ExcitedStatesAndDipoles.dat", "a") as file:
            file.write("Excited State #:" + str(it + 1) + "\n")
            file.write("Energy [eV]:" + format(Energy, "12.6f") + "\n")
            file.write(
                "Dipolematrixelements (x,y,z) [eBohr]:"
                + format(dx, "12.6f")
                + format(dy, "12.6f")
                + format(dz, "12.6f")
                + "\n"
            )
            file.write(
                "Dipole strength**2:" + format((dx**2 + dy**2 + dz**2), "12.6f") + "\n"
            )
            file.write(
                "Oszillator strength: "
                + format(
                    Energy / (3 * conFactors["a.u.->eV"] / 2) * (dx**2 + dy**2 + dz**2),
                    "12.6f",
                )
                + "\n"
            )
            sorted_m_index = []
            sorted_i_index = []
            eta_prime = deepcopy(eta)
            for id in enumerate(i_index):
                imax, mmax = np.unravel_index(
                    np.argmax(np.abs(eta_prime[:, :, it])),
                    np.shape(eta_prime[:, :, it]),
                )
                sorted_i_index.append(imax)
                sorted_m_index.append(mmax)
                eta_prime[imax, mmax, it] = 0.0
            it1 = 0
            for id2, i2 in enumerate(sorted_i_index):
                m2 = sorted_m_index[id2]
                amplitude2 = eta[i2, m2, it]
                if it1 == 0:
                    file.write(
                        "Dominant state transition:"
                        + "HOMO-"
                        + str(int(np.abs(m2 - id_homo)))
                        + "->"
                        + "LUMO+"
                        + str(i2 - id_homo - 1)
                        + "\n"
                    )
                    file.write(
                        "Excited State is composed of the individual particle-hole excitations:\n"
                    )
                if np.abs(amplitude2) > 0.1:
                    file.write(
                        format(m2 - id_homo, "3.0f")
                        + " ->"
                        + format(i2 - id_homo, "3.0f")
                        + ":"
                        + format(amplitude2, "12.6f")
                        + "\n"
                    )
                it1 += 1
    np.save("Transitiondipolevectors", TransitionDipolevectors)


def WFNonGrid(id=0, N1=200, N2=200, N3=200, parentfolder="./"):
    """Function to represent the DFT eigenstate HOMO+id on a real space grid within the unit cell with Nx,Ny,Nz grid points
    input:   id:               (int)                   specifies the Orbital, id=0 is HOMO id=1 is LUMO id=-1 is HOMO-1 ect.
    (opt.)   parentfolder:     (str)                   path to the .inp file of the cp2k calculation to read in the cell dimensions
             Nx,Ny,Nz:         (int)                   Number of grid points in each direction
    output:  f                 (Nx x Ny x Nz np.array) Wavefunction coefficients, where first index is x, second y and third z
    """

    def ElementaryPolynomialFunction(r, Rs, csdec):
        diff = r - Rs
        dx = diff[0, :, :, :]
        dy = diff[1, :, :, :]
        dz = diff[2, :, :, :]
        res = np.zeros(np.shape(dx))
        for item in csdec:
            res += (
                (item[1]) * dx ** (item[0][0]) * dy ** (item[0][1]) * dz ** (item[0][2])
            )
        return res

    def getElementaryBasisFunction(r, Rs, alphas, ds, csdec):
        """Function to evaluate a elementary CP2K atom centered basis function at position r.
        input:   r:         (3x1 np.array)               position at which the wf is evaluated
                 Rs:        (3x1 np.array)               position of the atom to which this basis function is attached
                 alphas:    (np.array)                   numpy array of the exponents  of the gaussian (same order as ds)
                 ds:        (np.array)                   contraction coefficients of the elementary gaussian
                 csdec:     ()
        """
        diff = r - Rs
        res = np.zeros(np.shape(diff[0, :, :, :]))
        for it in range(len(alphas)):
            res += ds[it] * np.exp(
                -alphas[it]
                * (
                    diff[0, :, :, :] ** 2
                    + diff[1, :, :, :] ** 2
                    + diff[2, :, :, :] ** 2
                )
            )
        return ElementaryPolynomialFunction(r, Rs, csdec) * res

    def getWavefunction(x, y, z, Atoms, a, Basis, voxelvolume, v1, v2, v3):
        """Function to construct the wavefunction with coefficients stored in a and CP2K atom centered basis described by
        Atoms, Basis, cs on a orthonormal xyz grid. The lengthx,lenghty, lenghtz parameters are the dimensions of the minimal voxel.
        """
        cs = {}

        cs["s"] = [[[0, 0, 0], 0.5 / np.sqrt(np.pi)]]

        cs["py"] = [[[0, 1, 0], np.sqrt(3.0 / (4.0 * np.pi))]]
        cs["pz"] = [[[0, 0, 1], np.sqrt(3.0 / (4.0 * np.pi))]]
        cs["px"] = [[[1, 0, 0], np.sqrt(3.0 / (4.0 * np.pi))]]

        cs["d-2"] = [[[1, 1, 0], 0.5 * np.sqrt(15.0 / np.pi)]]
        cs["d-1"] = [[[0, 1, 1], 0.5 * np.sqrt(15.0 / np.pi)]]
        cs["d0"] = [
            [[2, 0, 0], -0.25 * np.sqrt(5.0 / np.pi)],
            [[0, 2, 0], -0.25 * np.sqrt(5.0 / np.pi)],
            [[0, 0, 2], 0.5 * np.sqrt(5.0 / np.pi)],
        ]
        cs["d+1"] = [[[1, 0, 1], 0.5 * np.sqrt(15.0 / np.pi)]]
        cs["d+2"] = [
            [[2, 0, 0], 0.25 * np.sqrt(15.0 / np.pi)],
            [[0, 2, 0], -0.25 * np.sqrt(15.0 / np.pi)],
        ]

        cs["f-3"] = [
            [[2, 1, 0], 0.75 * np.sqrt(35.0 / 2.0 / np.pi)],
            [[0, 3, 0], -0.25 * np.sqrt(35.0 / 2.0 / np.pi)],
        ]
        cs["f-2"] = [[[1, 1, 1], 0.5 * np.sqrt(105.0 / np.pi)]]
        cs["f-1"] = [
            [[0, 1, 2], np.sqrt(21.0 / 2.0 / np.pi)],
            [[2, 1, 0], -0.25 * np.sqrt(21.0 / 2.0 / np.pi)],
            [[0, 3, 0], -0.25 * np.sqrt(21.0 / 2.0 / np.pi)],
        ]
        cs["f0"] = [
            [[0, 0, 3], 0.5 * np.sqrt(7.0 / np.pi)],
            [[2, 0, 1], -0.75 * np.sqrt(7 / np.pi)],
            [[0, 2, 1], -0.75 * np.sqrt(7 / np.pi)],
        ]
        cs["f+1"] = [
            [[1, 0, 2], np.sqrt(21.0 / 2.0 / np.pi)],
            [[1, 2, 0], -0.25 * np.sqrt(21.0 / 2.0 / np.pi)],
            [[3, 0, 0], -0.25 * np.sqrt(21.0 / 2.0 / np.pi)],
        ]
        cs["f+2"] = [
            [[2, 0, 1], 0.25 * np.sqrt(105.0 / np.pi)],
            [[0, 2, 1], -0.25 * np.sqrt(105.0 / np.pi)],
        ]
        cs["f+3"] = [
            [[3, 0, 0], 0.25 * np.sqrt(35.0 / 2.0 / np.pi)],
            [[1, 2, 0], -0.75 * np.sqrt(35.0 / 2.0 / np.pi)],
        ]

        cs["g-4"] = [
            [[3, 1, 0], 0.75 * np.sqrt(35.0 / np.pi)],
            [[1, 3, 0], -0.75 * np.sqrt(35.0 / np.pi)],
        ]
        cs["g-3"] = [
            [[2, 1, 1], 9.0 * np.sqrt(35.0 / (2 * np.pi)) / 4.0],
            [[0, 3, 1], -0.75 * np.sqrt(35.0 / (2.0 * np.pi))],
        ]
        cs["g-2"] = [
            [[1, 1, 2], 18.0 * np.sqrt(5.0 / (np.pi)) / 4.0],
            [[3, 1, 0], -3.0 * np.sqrt(5.0 / (np.pi)) / 4.0],
            [[1, 3, 0], -3.0 * np.sqrt(5.0 / (np.pi)) / 4.0],
        ]
        cs["g-1"] = [
            [[0, 1, 3], 3.0 * np.sqrt(5.0 / (2 * np.pi))],
            [[2, 1, 1], -9.0 * np.sqrt(5.0 / (2 * np.pi)) / 4.0],
            [[0, 3, 1], -9.0 * np.sqrt(5.0 / (2 * np.pi)) / 4.0],
        ]
        cs["g0"] = [
            [[0, 0, 4], 3.0 * np.sqrt(1.0 / (np.pi)) / 2.0],
            [[4, 0, 0], 9.0 * np.sqrt(1.0 / (np.pi)) / 16.0],
            [[0, 4, 0], 9.0 * np.sqrt(1.0 / (np.pi)) / 16.0],
            [[2, 0, 2], -9.0 * np.sqrt(1.0 / np.pi) / 2.0],
            [[0, 2, 2], -9.0 * np.sqrt(1.0 / np.pi) / 2.0],
            [[2, 2, 0], 9.0 * np.sqrt(1.0 / np.pi) / 8.0],
        ]
        cs["g+1"] = [
            [[1, 0, 3], 3.0 * np.sqrt(5.0 / (2 * np.pi))],
            [[1, 2, 1], -9.0 * np.sqrt(5.0 / (2 * np.pi)) / 4.0],
            [[3, 0, 1], -9.0 * np.sqrt(5.0 / (2 * np.pi)) / 4.0],
        ]
        cs["g+2"] = [
            [[2, 0, 2], 18.0 * np.sqrt(5.0 / (np.pi)) / 8.0],
            [[0, 2, 2], -18.0 * np.sqrt(5.0 / (np.pi)) / 8.0],
            [[0, 4, 0], 3.0 * np.sqrt(5.0 / (np.pi)) / 8.0],
            [[4, 0, 0], -3.0 * np.sqrt(5.0 / (np.pi)) / 8.0],
        ]
        cs["g+3"] = [
            [[1, 2, 1], -9.0 * np.sqrt(35.0 / (2 * np.pi)) / 4.0],
            [[3, 0, 1], 0.75 * np.sqrt(35.0 / (2.0 * np.pi))],
        ]
        cs["g+4"] = [
            [[4, 0, 0], 3.0 * np.sqrt(35.0 / np.pi) / 16.0],
            [[2, 2, 0], -18.0 * np.sqrt(35.0 / np.pi) / 16.0],
            [[0, 4, 0], 3.0 * np.sqrt(35.0 / np.pi) / 16.0],
        ]

        r = np.array(
            [
                x * v1[0] + y * v2[0] + z * v3[0],
                x * v1[1] + y * v2[1] + z * v3[1],
                x * v1[2] + y * v2[2] + z * v3[2],
            ]
        )
        it = 0
        res = 0.0
        for atom in Atoms:
            Rs = np.array(
                [
                    1.88972613288564 * atom[2],
                    1.88972613288564 * atom[3],
                    1.88972613288564 * atom[4],
                ]
            )
            Rstensor = np.zeros(np.shape(r))
            Rstensor[0, :, :, :] = Rs[0]
            Rstensor[1, :, :, :] = Rs[1]
            Rstensor[2, :, :, :] = Rs[2]
            Atomicsymbol = atom[1]
            for Bf in Basis[Atomicsymbol]:
                Coefficients = np.array(Bf[3:])
                lm = Bf[2][1:]
                # l=Bf[2][1]
                alphas = np.array(
                    [Coefficients[it][0] for it in range(len(Coefficients))]
                )
                ds = np.array([Coefficients[it][1] for it in range(len(Coefficients))])
                csdec = cs[lm]
                value = getElementaryBasisFunction(r, Rstensor, alphas, ds, csdec)
                # check that normalized
                normfactor = np.sqrt(voxelvolume * np.sum(value**2))
                res += a[it] * value / normfactor
                it += 1
        return res

    Homoid = Read.read_homo_index(parentfolder)
    KSHamiltonian, OLM = Read.read_ks_matrices(parentfolder)
    Sm12 = Util.LoewdinTransformation(OLM)
    KSHorth = np.dot(Sm12, np.dot(KSHamiltonian, Sm12))
    _, A = np.linalg.eigh(KSHorth)
    a = Sm12 @ A[:, id + Homoid]
    a *= getPhaseOfMO(A[:, id + Homoid])

    Atoms = Read.read_atomic_coordinates(parentfolder)
    Basis = AtomicBasis.getBasis(parentfolder)
    Cellvectors = Read.read_cell_vectors(parentfolder)
    # Convert to atomic units
    cellvector1 = Cellvectors[0] * 1.88972613288564
    cellvector2 = Cellvectors[1] * 1.88972613288564
    cellvector3 = Cellvectors[2] * 1.88972613288564
    # get voxel volume in a.u.**3
    voxelvolume = np.dot(cellvector1, np.cross(cellvector2, cellvector3)) / (
        N1 * N2 * N3
    )
    # discretization
    v1 = cellvector1 / np.linalg.norm(cellvector1)
    v2 = cellvector2 / np.linalg.norm(cellvector2)
    v3 = cellvector3 / np.linalg.norm(cellvector3)
    length1 = np.linalg.norm(cellvector1) / N1
    length2 = np.linalg.norm(cellvector2) / N2
    length3 = np.linalg.norm(cellvector3) / N3
    grid1 = length1 * np.arange(N1)
    grid2 = length2 * np.arange(N2)
    grid3 = length3 * np.arange(N3)
    xx, yy, zz = np.meshgrid(grid1, grid2, grid3, indexing="ij")
    f = np.zeros((N1, N2, N3))
    f = getWavefunction(xx, yy, zz, Atoms, a, Basis, voxelvolume, v1, v2, v3)
    print(voxelvolume * np.sum(np.sum(np.sum(f**2))))
    f /= np.sqrt(voxelvolume * np.sum(np.sum(np.sum(f**2))))
    filename = str(id)
    np.save(parentfolder + "/" + filename, f)
    return f


def WFNsOnGrid(
    ids=[0],
    N1=200,
    N2=200,
    N3=200,
    cell_vectors=[0.0, 0.0, 0.0],
    saveflag=True,
    parentfolder="./",
):
    """Function to represent the DFT eigenstate HOMO+id on a real space grid within the unit cell with Nx,Ny,Nz grid points
    input:   id:               (int)                   specifies the Orbital, id=0 is HOMO id=1 is LUMO id=-1 is HOMO-1 ect.
    (opt.)   parentfolder:     (str)                   path to the .inp file of the cp2k calculation to read in the cell dimensions
             Nx,Ny,Nz:         (int)                   Number of grid points in each direction
    output:  f                 (Nx x Ny x Nz np.array) Wavefunction coefficients, where first index is x, second y and third z
    """
    Homoid = Read.read_homo_index(parentfolder)
    _, A, Sm12 = Util.Diagonalize_KS_Hamiltonian(parentfolder)
    data = []
    for id in ids:
        a = Sm12 @ A[:, id + Homoid]
        a *= getPhaseOfMO(A[:, id + Homoid])
        Atoms = Read.read_atomic_coordinates(parentfolder)
        Basis = AtomicBasis.getBasis(parentfolder)
        Cellvectors = Read.read_cell_vectors(parentfolder)
        # Convert to atomic units
        cellvector1 = Cellvectors[0] * 1.88972613288564
        cellvector2 = Cellvectors[1] * 1.88972613288564
        cellvector3 = Cellvectors[2] * 1.88972613288564
        # get voxel volume in a.u.**3
        voxelvolume = np.dot(cellvector1, np.cross(cellvector2, cellvector3)) / (
            N1 * N2 * N3
        )
        # discretization
        length1 = np.linalg.norm(cellvector1) / N1
        length2 = np.linalg.norm(cellvector2) / N2
        length3 = np.linalg.norm(cellvector3) / N3
        grid1 = length1 * np.arange(N1)
        grid2 = length2 * np.arange(N2)
        grid3 = length3 * np.arange(N3)
        f = AtomicBasis.WFNonxyzGrid(grid1, grid2, grid3, a, Atoms, Basis, cell_vectors)
        print(voxelvolume * np.sum(np.sum(np.sum(f**2))))
        f /= np.sqrt(voxelvolume * np.sum(np.sum(np.sum(f**2))))
        data.append(f)
        filename = str(id)
        if saveflag:
            np.save(parentfolder + "/" + filename, f)
    return np.array(data)


def LocalPotentialOnGrid(
    gridpoints, MatrixElements, cell_vectors=[0.0, 0.0, 0.0], parentfolder="./"
):
    """Function to represent the DFT eigenstate HOMO+id on a real space grid within the unit cell with Nx,Ny,Nz grid points
    input:   id:               (int)                   specifies the Orbital, id=0 is HOMO id=1 is LUMO id=-1 is HOMO-1 ect.
    (opt.)   parentfolder:     (str)                   path to the .inp file of the cp2k calculation to read in the cell dimensions
             Nx,Ny,Nz:         (int)                   Number of grid points in each direction
    output:  f                 (Nx x Ny x Nz np.array) Wavefunction coefficients, where first index is x, second y and third z
    """
    Atoms = Read.read_atomic_coordinates(parentfolder)
    Basis = AtomicBasis.getBasis(parentfolder)
    data = AtomicBasis.LocalPotentialonxyzGrid(
        gridpoints, MatrixElements, Atoms, Basis, cell_vectors
    )
    return np.array(data)


def getOverlapOnGrids(WFN1, WFN2, parentfolder="./"):
    N1 = np.shape(WFN1)[0]
    N2 = np.shape(WFN1)[1]
    N3 = np.shape(WFN1)[2]
    Cellvectors = Read.read_cell_vectors(parentfolder)
    # Convert to atomic units
    cellvector1 = Cellvectors[0] * 1.88972613288564
    cellvector2 = Cellvectors[1] * 1.88972613288564
    cellvector3 = Cellvectors[2] * 1.88972613288564
    # get voxel volume in a.u.**3
    voxelvolume = np.dot(cellvector1, np.cross(cellvector2, cellvector3)) / (
        N1 * N2 * N3
    )
    overlap = voxelvolume * np.sum(WFN1 * WFN2)
    return overlap


def getTransitionDipoleMomentsNumerical(
    minweigth=0.05, Nx=100, Ny=100, Nz=100, pathtoExcitedstates="./", pathtoMO="./"
):
    """Function to generate a file, where the Dipolmatrixelements and the excited states are summarized
    input:   path              (string)                path to the folder, where the wavefunctions have been generated and where the .inp/outputfile of the
                                                       TDDFPT calculation lies
    output:                    (void)
    """
    # Readin the excited states
    energies = []
    energies, eta = getUniqueExcitedStates(minweigth, pathtoExcitedstates, pathtoMO)
    id_homo = Read.read_homo_index(pathtoExcitedstates)
    TransitionDipolevectors = []
    with open("ExcitedStatesAndDipoles.dat", "a") as file:
        file.write("Python Convention of state labeling!\n")
    for it in range(len(energies)):
        i_index, m_index = np.where(np.abs(eta[:, :, it] > minweigth))
        Energy = energies[it]
        dx = 0.0
        dy = 0.0
        dz = 0.0
        for id, i in enumerate(i_index):
            m = m_index[id]
            amplitude = eta[i, m, it]
            StateLabel1 = m - id_homo
            StateLabel2 = i - id_homo
            try:
                State1 = np.load(str(StateLabel1) + ".npy")
            except:
                print("Have not found state #: " + str(StateLabel1) + "\n")
                print("Creating it from scratch. This may take a while!\n")
                State1 = WFNonGrid(StateLabel1, Nx, Ny, Nz, pathtoMO)
            try:
                State2 = np.load(str(StateLabel2) + ".npy")
            except:
                print("Have not found state #: " + str(StateLabel2) + "\n")
                print("Creating it from scratch. This may take a while!\n")
                State2 = WFNonGrid(StateLabel2, Nx, Ny, Nz, pathtoMO)
            edx, edy, edz = ComputeDipolmatrixElements(State2, State1, pathtoMO)
            dx += -amplitude * edx
            dy += -amplitude * edy
            dz += -amplitude * edz
        TransitionDipolevectors.append(np.array([dx, dy, dz]))
        with open("ExcitedStatesAndDipoles.dat", "a") as file:
            file.write("Excited State #:" + str(it + 1) + "\n")
            file.write("Energy [eV]:" + format(Energy, "12.6f") + "\n")
            file.write(
                "Dipolematrixelements (x,y,z) [eBohr]:"
                + format(dx, "12.6f")
                + format(dy, "12.6f")
                + format(dz, "12.6f")
                + "\n"
            )
            file.write(
                "Dipole strength**2:" + format((dx**2 + dy**2 + dz**2), "12.6f") + "\n"
            )
            file.write(
                "Oszillator strength: "
                + format(
                    Energy
                    / (3 * ConversionFactors["a.u.->eV"] / 2)
                    * (dx**2 + dy**2 + dz**2),
                    "12.6f",
                )
                + "\n"
            )
    np.save("Transitiondipolevectors", TransitionDipolevectors)
    return TransitionDipolevectors


def ComputeDipolmatrixElements(State1, State2, path="./"):
    """Function to compute the Dipolematrixelement between two wavefunctions State1 and State2 on a real space grid
    input:   State1/2:         (np.array)              Nx x Ny x Nz numpy array with the wave function coefficients at each gridpoint
    (opt.)   path:             (str)                   path to the .inp file of the cp2k calculation to read in the cell dimensions
    output:  d12               (float)                 the dipolematrixelement between the states unit e*Bohr=a.u. for x y and z direction
    """
    # Check that State1 and State2 have the same dimensions
    if np.shape(State1) != np.shape(State2):
        raise ValueError("Different Grid for the two states. Reconsider your input!")
    N1 = np.shape(State1)[0]
    N2 = np.shape(State1)[1]
    N3 = np.shape(State1)[2]
    Cellvectors = Read.getCellSize(path)
    # Convert to atomic units
    cellvector1 = Cellvectors[0] * 1.88972613288564
    cellvector2 = Cellvectors[1] * 1.88972613288564
    cellvector3 = Cellvectors[2] * 1.88972613288564
    # get voxel volume in a.u.**3
    voxelvolume = np.dot(cellvector1, np.cross(cellvector2, cellvector3)) / (
        N1 * N2 * N3
    )
    ##Set up real space grids
    # discretization
    length1 = np.linalg.norm(cellvector1) / N1
    length2 = np.linalg.norm(cellvector2) / N2
    length3 = np.linalg.norm(cellvector3) / N3
    grid1 = length1 * np.arange(N1)
    grid2 = length2 * np.arange(N2)
    grid3 = length3 * np.arange(N3)
    xx, yy, zz = np.meshgrid(grid1, grid2, grid3, indexing="ij")
    transitiondensity = State1 * State2
    dxint = np.sum(xx * transitiondensity) * voxelvolume
    dyint = np.sum(yy * transitiondensity) * voxelvolume
    dzint = np.sum(zz * transitiondensity) * voxelvolume
    return dxint, dyint, dzint
