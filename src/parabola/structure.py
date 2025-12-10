import os
import pickle

import numpy as np
import spglib

from . import Geometry, Read, Symmetry, electronics, vibrations


class Molecular_Structure:
    # Class-level flag to manage pickling recursion
    _is_unpickling = False

    # 1. The __new__ method intercepts creation.
    def __new__(cls, *args, **kwargs):
        """
        Check if a pickled instance exists. If so, return it.
        Otherwise, create a new instance and let it be initialized.

        Handles pickling by detecting when it's called internally by pickle.load().
        """
        # If the _is_unpickling flag is set, it means we are inside a pickle.load() call
        # and this __new__ is being called by pickle to create a blank instance.
        # In this case, we simply return a new, uninitialized instance.
        if cls._is_unpickling:
            return super().__new__(cls)

        # Normal constructor call path: 'name' must be provided
        name = kwargs.get("name")
        if name is None:
            # This handles cases where Molecular_Structure() is called without 'name',
            # which is not allowed for normal instantiation.
            raise TypeError("Molecular_Structure.__new__() missing 1 required keyword-only argument: 'name'")

        path = kwargs.get("path", "")
        abs_path = os.path.abspath(path)
        pickle_filename = os.path.join(abs_path, f"{name}.{cls.__name__.lower()}.pickle")

        if os.path.isfile(pickle_filename):
            print(f"🔄 : Found file. Loading from: {pickle_filename}")
            # Set the flag to prevent recursive calls to __new__ by pickle.load()
            cls._is_unpickling = True
            try:
                with open(pickle_filename, "rb") as f:
                    # Load and return the existing instance.
                    # __init__ will still be called on this loaded instance.
                    instance = pickle.load(f)
                return instance
            finally:
                # Always reset the flag, even if an error occurs during loading
                cls._is_unpickling = False
        else:
            # Create a new, empty instance using the parent class's __new__.
            instance = super().__new__(cls)
            # This new instance will be passed to __init__ for setup.
            return instance

    # 2. The __init__ method MUST have a guard.
    def __init__(
        self,
        *,
        name: str,
        path: str = "./",
        electronics_path: str | None = None,
        vibrational_path: str | None = None,
        save: bool = True,
    ):
        """
        Initializes or updates the attributes of the instance.

        If the instance is newly created, all core attributes are initialized.
        If the instance is loaded from a pickle, only electronics
        and Vibrations are updated based on the current
        `electronic_path` and `vibrational_path` arguments.
        """
        # Determine if this is a newly created instance or a loaded one.
        # A loaded instance will already have 'name' set by pickle.
        is_already_initialized = hasattr(self, "name")
        if not is_already_initialized:
            self.name = name
            self.path = os.path.abspath(path) if path == "." else os.path.abspath(path)
            print(f"⌬ : Creating new Molecular Structure for {name} in:")
            print(os.path.abspath(self.path))
            print("▲ : Processing Geometric Structural Information")
            xyz_filepath = Read.get_xyz_filename(path=path, verbose=True)
            self.xyz_path = xyz_filepath
            coordinates, masses, atomic_symbols = Read.read_coordinates_and_masses(xyz_filepath)
            self.coordinates = coordinates
            self.masses = masses
            self.geometric_center = np.zeros(3)
            self.geometrically_centered_coordinates = []
            self.geometric_principle_axis = []
            self.center_of_mass = np.zeros(3)
            self.mass_centered_coordinates = []
            self.mass_principle_axis = []
            self.atoms = atomic_symbols
            self.atomic_numbers = self._get_atomic_numbers()
            if Read.read_periodicity(path):
                self.periodicity = (1, 1, 1)
            else:
                self.periodicity = (0, 0, 0)
            self.unitcells = {}
            self.cellvectors = Read.read_cell_vectors(path)
            self.Molecular_Symmetry = Molecular_Symmetry(self)
            # Initialize paths and structures to None for a new instance
            self.electronics_path = None
            self.vibrations_path = None
            self.Electronics = None
            self.Vibrations = None
        else:
            print(f"✔️ : Object for '{self.name}' is already initialized. Checking for path updates.")
            # If loaded, ensure the base path reflects the current call's path if it's different.
            current_abs_path = os.path.abspath(path) if path == "." else os.path.abspath(path)
            if self.path != current_abs_path:
                print(f"🔄 : Base path updated from {self.path} to {current_abs_path}")
                self.path = current_abs_path

        # Logic for electronic structure:
        # If electronic_path is provided in the current call, update the structure.
        if electronics_path is not None:
            # Recompute only if the path has changed or if the structure is currently None.
            if self.electronics_path != electronics_path or self.Electronics is None:
                print("🔄 : Electronic Structure path provided/changed. Computing/updating electronic structure.")
                self.electronics_path = electronics_path
                self.Electronics = self.determine_electronics()
            else:
                print("✔️ : Electronic path is the same and structure exists. No recomputation needed.")

        # Logic for vibrational structure:
        # If vibrational_path is provided in the current call, update the structure.
        if vibrational_path is not None:
            # Recompute only if the path has changed or if the structure is currently None.
            if self.vibrations_path != vibrational_path or self.Vibrations is None:
                print("🔄 : Vibrational Structure path provided/changed. Computing/updating vibrational structure.")
                self.vibrations_path = vibrational_path
                self.Vibrations = self.determine_vibrations()
            else:
                print("✔️ : Vibrational Structure path is the same and structure exists. No recomputation needed.")
        if save and not is_already_initialized:
            self.save()

    def determine_electronics(self):
        """
        Lazy-load or compute the ElectronicStructure, only if a valid path is given.
        If no path is provided, raise an error (required data is missing).
        """
        print("\n" + "=" * 40)
        print("⚡ : ELECTRONIC STRUCTURE ")
        print("=" * 40)
        full_path = os.path.abspath(self.electronics_path)
        print("Vibrational Structure data taken from:")
        print(full_path)
        electronics_var = electronics.Electronics(self)
        return electronics_var

    def determine_vibrations(self):
        """
        Lazy-load or compute the VibrationalStructure, only if a valid path is given.
        If no path is provided, raise an error (required data is missing).
        """
        # Enhanced print statement for Vibrational Analysis
        print("\n" + "=" * 40)
        print("🎜 : VIBRATIONAL ANALYSIS  ")
        print("=" * 40)
        full_path = os.path.abspath(self.vibrations_path)
        print("Vibrational Structure data taken from:")
        print(full_path)
        return vibrations.Vibrations(self)

    def save(self, filename=None):
        if filename is None:
            pickle_suffix = f"{self.__class__.__name__.lower()}.pickle"
            filename = f"{self.name}.{pickle_suffix}"

        full_path = os.path.join(self.path, filename)
        print(f"💾 Saving to: {full_path}")

        with open(full_path, "wb") as f:
            pickle.dump(self, f)

    @property
    def info(self):
        """
        Nicely formatted molecular structure summary with attribute descriptions.
        """

        info = {
            "name": {
                "label": "Name",
                "description": "The name identifier of the molecular structure",
                "value": self.name,
                "unit": None,
            },
            "path": {
                "label": "Path",
                "description": "File or data source path for the molecular data",
                "value": self.path,
                "unit": None,
            },
            "atoms": {
                "label": "Number of atoms",
                "description": "Total number of atoms in the structure",
                "value": len(self.atoms),
                "unit": None,
            },
            "atomic_symbols": {
                "label": "Atomic Symbols",
                "description": "Chemical symbols of the constituent atoms",
                "value": ", ".join(self.atoms),
                "unit": None,
            },
            "coordinates": {
                "label": "Cartesian Coordinates",
                "description": "Original coordinates",
                "value": [
                    f"Atom {i + 1} [{self.atoms[i]}]: {np.round(coord, 6)}" for i, coord in enumerate(self.coordinates)
                ],
                "unit": "Angström",
            },
            "geom_centered": {
                "label": "Geometrically Centered Coordinates",
                "description": "Coordinates centered to geometric_center and represented in the frame described by geometric_principle_axis",
                "value": [
                    f"Atom {i + 1} [{self.atoms[i]}]: {np.round(coord, 6)}"
                    for i, coord in enumerate(self.geometrically_centered_coordinates)
                ],
                "unit": "Angström",
            },
            "cell_vectors": {
                "label": "Cell Vectors",
                "description": "Lattice or box vectors (in Angstroms)",
                "value": [f"Vector {i + 1}: {vec}" for i, vec in enumerate(self.cellvectors)],
                "unit": "Angström",
            },
            "periodicity": {
                "label": "Periodicity",
                "description": "How many copies are detected along the cellvectors. 0: open boundary conditions, 1:periodic boundary conditions",
                "value": self.periodicity,
                "unit": None,
            },
            "symmetry": {
                "label": "Symmetry Information",
                "description": "Symmetry generators or operations of the structure",
                "value": (
                    list(self.Molecular_Symmetry.Symmetry_Generators.keys())
                    if self.Molecular_Symmetry.Symmetry_Generators.keys()
                    else ["No symmetry information available."]
                ),
                "unit": None,
            },
        }

        print("=" * 40)
        print(" Molecular Structure Information")
        print("=" * 40)

        for key, entry in info.items():
            print(f"\n{entry['label']}:")
            print(f"  Description: {entry['description']}")
            if entry["unit"] is not None:
                print(f"  Unit: {entry['unit']}")
            value = entry["value"]

            if isinstance(value, list):
                for v in value:
                    print(f"  {v}")
            else:
                print(f"  {value}")

        print("\n" + "=" * 40)

    @property
    def n_atoms(self):
        return len(self.atoms)

    def _get_atomic_numbers(self):
        """
        Convert atomic symbols to atomic numbers using a predefined mapping.
        """
        periodic_table = {
            "H": 1,
            "He": 2,
            "Li": 3,
            "Be": 4,
            "B": 5,
            "C": 6,
            "N": 7,
            "O": 8,
            "F": 9,
            "Ne": 10,
            "Na": 11,
            "Mg": 12,
            "Al": 13,
            "Si": 14,
            "P": 15,
            "S": 16,
            "Cl": 17,
            "Ar": 18,
            "K": 19,
            "Ca": 20,
            "Sc": 21,
            "Ti": 22,
            "V": 23,
            "Cr": 24,
            "Mn": 25,
            "Fe": 26,
            "Co": 27,
            "Ni": 28,
            "Cu": 29,
            "Zn": 30,
            "Ga": 31,
            "Ge": 32,
            "As": 33,
            "Se": 34,
            "Br": 35,
            "Kr": 36,
            # Add more elements as needed
        }
        atomic_numbers = []
        for symbol in self.atoms:
            atomic_number = periodic_table.get(symbol)
            if atomic_number is None:
                raise ValueError(f"Unknown atomic symbol: {symbol}")
            atomic_numbers.append(atomic_number)
        return atomic_numbers


#### Define Symmetry Class ####
class Molecular_Symmetry(Symmetry.Symmetry):
    def __init__(self, molecular_structure):
        super().__init__()  # Initialize parent class
        self.determine_symmetry(molecular_structure=molecular_structure)

    def determine_symmetry(
        self,
        molecular_structure: "Molecular_Structure",
    ):
        tol_translation = 5 * 10 ** (-4)
        self._test_translation(molecular_structure=molecular_structure, tol_translation=tol_translation)
        primitive_indices = self._find_indices_in_primitive_cell(
            molecular_structure=molecular_structure, tol_translation=tol_translation
        )

        geometry_centered_coordinates, center = Geometry.ComputeCenterOfGeometryCoordinates(
            np.array(molecular_structure.coordinates)[primitive_indices]
        )
        molecular_structure.geometric_center = center
        if len(primitive_indices) > 1:
            v1, v2, v3 = Geometry.getPrincipleAxis(geometry_centered_coordinates, masses=None, tol=1e-3)
        else:
            v1 = np.array([1.0, 0.0, 0.0])
            v2 = np.array([0.0, 1.0, 0.0])
            v3 = np.array([0.0, 0.0, 1.0])
        molecular_structure.geometric_principle_axis = [v1, v2, v3]
        G_UC_CC = []
        for coor in geometry_centered_coordinates:
            x = np.dot(v1, coor)
            y = np.dot(v2, coor)
            z = np.dot(v3, coor)
            G_UC_CC.append(np.array([x, y, z]))
        molecular_structure.geometrically_centered_coordinates = G_UC_CC
        if len(primitive_indices) > 1:
            mass_centered_coordinates, center_of_mass = Geometry.ComputeCenterOfMassCoordinates(
                np.array(molecular_structure.coordinates)[primitive_indices],
                np.array(molecular_structure.masses)[primitive_indices],
            )
            molecular_structure.center_of_mass = center_of_mass
            v1, v2, v3 = Geometry.getPrincipleAxis(
                mass_centered_coordinates,
                masses=np.array(molecular_structure.masses)[primitive_indices],
                tol=1e-3,
            )
            molecular_structure.mass_principle_axis = [v1, v2, v3]
            G_UC_CC = []
            for coor in geometry_centered_coordinates:
                x = np.dot(v1, coor)
                y = np.dot(v2, coor)
                z = np.dot(v3, coor)
                G_UC_CC.append(np.array([x, y, z]))
            molecular_structure.mass_centered_coordinates = G_UC_CC
        else:
            molecular_structure.center_of_mass = center
            molecular_structure.mass_principle_axis = [v1, v2, v3]
            molecular_structure.mass_centered_coordinates = G_UC_CC
        if len(primitive_indices) > 1:
            self._test_inversion(molecular_structure, tol_inversion=5 * 10 ** (-3))
            self._test_rotation(molecular_structure, tol_rotation=5 * 10 ** (-3))
            self._test_mirror(molecular_structure, tol_mirror=5 * 10 ** (-2))
        if not self.Symmetry_Generators:
            self.Symmetry_Generators["Id"] = np.eye(len(molecular_structure.masses))

    def _test_translation(self, molecular_structure: "Molecular_Structure", tol_translation):
        if molecular_structure.periodicity == (1, 1, 1):
            coords = molecular_structure.coordinates
            cellvectors = molecular_structure.cellvectors
            atomic_symbols = molecular_structure.atoms
            supercell, primitive_indices, scaled_lattice = getPrimitiveUnitCell(
                cellvectors, coords, atomic_symbols, tolerance=tol_translation
            )
            molecular_structure.periodicity = supercell
            molecular_structure.primitive_indices = primitive_indices
            relative_cell_coordinates, _ = getCellCoordinates(
                scaled_lattice,
                coords,
                primitive_indices,
                tolerance=tol_translation,
            )
            Tx, Ty, Tz, xFlag, yFlag, zFlag = getTranslationOps(relative_cell_coordinates, supercell)
            if xFlag and np.sum(np.abs(Tx - np.eye(np.shape(Tx)[0]))) > 10 ** (-10):
                self.Symmetry_Generators["t1"] = Tx
            if yFlag and np.sum(np.abs(Ty - np.eye(np.shape(Ty)[0]))) > 10 ** (-10):
                self.Symmetry_Generators["t2"] = Ty
            if zFlag and np.sum(np.abs(Tz - np.eye(np.shape(Tz)[0]))) > 10 ** (-10):
                self.Symmetry_Generators["t3"] = Tz
            # Compute the unit cells
            p0, p1, p2 = molecular_structure.periodicity
            primitive_indices = np.array(primitive_indices)
            dim = Tx.shape[0]

            # Precompute one-hot identity matrix
            identity = np.eye(dim)

            # Precompute all matrix powers
            Tx_powers = [np.linalg.matrix_power(Tx, i) for i in range(p0)]
            Ty_powers = [np.linalg.matrix_power(Ty, i) for i in range(p1)]
            Tz_powers = [np.linalg.matrix_power(Tz, i) for i in range(p2)]
            # Loop through periodicity grid
            for n in range(p0):
                for m in range(p1):
                    for k in range(p2):
                        # Combined transformation matrix
                        M = Tx_powers[n] @ Ty_powers[m] @ Tz_powers[k]

                        # Transform selected primitive basis vectors
                        transformed = M @ identity[:, primitive_indices]

                        # Get index of max in each transformed vector
                        indices = np.argmax(transformed, axis=0)
                        molecular_structure.unitcells[(n, m, k)] = indices.tolist()
        else:
            molecular_structure.unitcells[(0, 0, 0)] = [it for it in range(len(molecular_structure.atoms))]

    def _test_inversion(self, molecular_structure: "Molecular_Structure", tol_inversion):
        atomic_symbols = molecular_structure.atoms
        primitive_indices = molecular_structure.unitcells[(0, 0, 0)]
        geometry_centered_coordinates = molecular_structure.geometrically_centered_coordinates
        has_symmetry, inversion_pairs = detect_inversion_symmetry(
            geometry_centered_coordinates,
            np.array(atomic_symbols)[primitive_indices],
            tol_inversion,
        )
        if has_symmetry:
            # Generate the Original Pairs
            pairs = {}
            for idx, inv_idx in inversion_pairs.items():
                for uc in molecular_structure.unitcells:
                    ucindex = molecular_structure.unitcells[uc]
                    pairs[ucindex[idx]] = ucindex[inv_idx]
                pairs[primitive_indices[idx]] = primitive_indices[inv_idx]
            # Add the remaining pairs on the diagonal
            nAtoms = len(atomic_symbols)
            PrimitiveInversion = get_Inversion_Symmetry_Generator(pairs, nAtoms)
            self.Symmetry_Generators["i"] = PrimitiveInversion

    def _test_rotation(self, molecular_structure: "Molecular_Structure", tol_rotation, nmax=10):
        geometrically_centered_coordinates = molecular_structure.geometrically_centered_coordinates
        atomic_symbols = molecular_structure.atoms
        primitive_indices = molecular_structure.unitcells[(0, 0, 0)]
        for axis in ["x", "y", "z"]:
            for n in range(nmax, 1, -1):
                has_symmetry, rotation_pairs = detect_rotational_symmetry(
                    geometrically_centered_coordinates,
                    np.array(atomic_symbols)[primitive_indices],
                    axis=axis,
                    n=n,
                    tolerance=tol_rotation,
                )
                if has_symmetry and n != 1 and n != nmax:
                    break
                elif has_symmetry and n == nmax:
                    has_symmetry_2, _ = detect_rotational_symmetry(
                        geometrically_centered_coordinates,
                        np.array(atomic_symbols)[primitive_indices],
                        axis=axis,
                        n=nmax + 1,
                        tolerance=tol_rotation,
                    )
                    if has_symmetry_2 and has_symmetry:
                        has_symmetry = False
                        break
            if has_symmetry and n != 1:
                # Generate the Original Pairs
                pairs = {}
                for idx, rot_idx in rotation_pairs.items():
                    for uc in molecular_structure.unitcells:
                        ucindex = molecular_structure.unitcells[uc]
                        pairs[ucindex[idx]] = ucindex[rot_idx]
                # Add the remaining pairs on the diagonal
                nAtoms = len(atomic_symbols)
                PrimitiveRotation = get_Inversion_Symmetry_Generator(pairs, nAtoms)
                self.Symmetry_Generators["C" + axis + "_" + str(n)] = PrimitiveRotation

    def _test_mirror(self, molecular_structure: "Molecular_Structure", tol_mirror):
        geometrically_centered_coordinates = molecular_structure.geometrically_centered_coordinates
        atomic_symbols = molecular_structure.atoms
        primitive_indices = molecular_structure.unitcells[(0, 0, 0)]
        for axis in ["x", "y", "z"]:
            has_symmetry, mirror_pairs = detect_mirror_symmetry(
                geometrically_centered_coordinates,
                np.array(atomic_symbols)[primitive_indices],
                axis=axis,
                tolerance=tol_mirror,
            )
            if has_symmetry:
                # Generate the Original Pairs
                pairs = {}
                for idx, rot_idx in mirror_pairs.items():
                    for uc in molecular_structure.unitcells:
                        ucindex = molecular_structure.unitcells[uc]
                        pairs[ucindex[idx]] = ucindex[rot_idx]
                nAtoms = len(atomic_symbols)
                PrimitiveMirror = get_Inversion_Symmetry_Generator(pairs, nAtoms)
                self.Symmetry_Generators["S" + axis] = PrimitiveMirror

    def _find_indices_in_primitive_cell(self, molecular_structure: "Molecular_Structure", tol_translation=1e-6):
        """
        This function checks if the given structure is a primitive cell and returns the indices of the atoms
        located in the primitive unit cell.
        The primitive_cellvectors are also stored/updated in the molecular_structure object.
        Returns
        -------
            primitive_indices: list
                list of indices of the atoms in the primitive unit cell
        """
        # use spglib to find the primitive lattice vectors
        primitive_cell = spglib.spglib.find_primitive(
            (
                molecular_structure.cellvectors,
                molecular_structure.coordinates @ np.linalg.inv(molecular_structure.cellvectors),
                molecular_structure.atomic_numbers,
            ),
            symprec=tol_translation,
        )  # returns (cellvectors_of_primitive_cell, positions_in_fractional_coord, atomic_numbers)

        # determine the multiplicity of the primitive cell in supercell
        # if multiplicity == np.prod(molecular_structure.periodicity),
        # then the found conventional cell is already primitive else, find the sublattice in the conventional cell
        # SC_cellvectors = transformation_matrix @ PC_cellvectors
        transformation_matrix = molecular_structure.cellvectors @ np.linalg.inv(primitive_cell[0])
        multiplicity_of_primitive = np.linalg.det(transformation_matrix)
        if np.sum(np.array(molecular_structure.periodicity))>0: #this is the periodic case
            if np.isclose(multiplicity_of_primitive, np.round(multiplicity_of_primitive), atol=1e-6):
                multiplicity_of_primitive = int(np.round(multiplicity_of_primitive))
                molecular_structure.primitive_cellvectors = molecular_structure.cellvectors / np.array(
                        molecular_structure.periodicity
                    )

                if multiplicity_of_primitive / np.prod(molecular_structure.periodicity) != 1:
                    print("ℹ️ : Sublattice detected in the conventional cell. Updating to primitive cell.")

                    # This translation vector is hardcoded for orthorhombic supercell to hexagonal unit cell conversion!
                    # However, it is generalized for orthorhomic supercells constructed from any multiplicity of the
                    # hexagonal primitive unit cell
                    translation_vector = np.array([1.0, 1.0, 0.0]) / (
                        multiplicity_of_primitive / np.prod(molecular_structure.periodicity)
                    )
                    primitive_indices = self._find_sublattice(
                        molecular_structure=molecular_structure, translation_vector=translation_vector
                    )

                    # This 30 degree rotation is hardcoded for orthorhombic supercell to hexagonal unit cell conversion!
                    rotation = np.array(
                        [[np.cos(np.pi / 6), -np.sin(np.pi / 6), 0], [np.sin(np.pi / 6), np.cos(np.pi / 6), 0], [0, 0, 1]]
                    )
                    molecular_structure.primitive_cellvectors = primitive_cell[0] @ rotation
                    molecular_structure.cellvectors = np.round(
                        (molecular_structure.primitive_cellvectors * np.array(molecular_structure.periodicity)[:, None]),
                        decimals=6,
                    )
                    molecular_structure.cellvectors[0] *= multiplicity_of_primitive / np.prod(
                        molecular_structure.periodicity
                    )

                    molecular_structure.coordinates = wrapping_coordinates(
                        molecular_structure.coordinates, molecular_structure.cellvectors
                    )

                    # run _test_translation again with updated cell vectors
                    molecular_structure.periodicity = (1, 1, 1)  # resetting before calling _test_translation again
                    self._test_translation(molecular_structure=molecular_structure, tol_translation=tol_translation)
                else:
                    # there is no sublattice, the conventional cell is already primitive
                    primitive_indices = molecular_structure.unitcells[(0, 0, 0)]
            else:
                raise ValueError("Could not determine multiplicity of the primitive cell.")
        else:# this is the non-periodic case
            primitive_indices = molecular_structure.unitcells[(0, 0, 0)]
        return primitive_indices

    def _find_sublattice(
        self,
        molecular_structure: "Molecular_Structure",
        translation_vector: np.ndarray = [0.5, 0.5, 0],
        tol_translation=1e-6,
    ):
        """
        Find if there exists a sublattice in the given "conventional" primitive unit cell.
        Implemented and tested for: Orthorhombic -> hexagonal symmetry

        Parameters
        ----------
            translation_vector: translation vector, default [0.5, 0.5, 0] for hexagonal to orthorhombic cell
            tol_translation: tolerance for translation symmetry detection, default 1e-6

        Returns
        -------
            primitive_indices: list
                list of indices of the atoms in the primitive unit cell
        """
        fractional_coordinates = to_fractional(
            molecular_structure.primitive_cellvectors,
            molecular_structure.coordinates,
        )
        relative_vectors = fractional_coordinates[None, :, :] - fractional_coordinates[:, None, :]
        translation_vector = np.abs(translation_vector)  # make sure translation vector is positive
        mask = (
            np.isclose(np.abs(relative_vectors[..., 0]), translation_vector[0], atol=tol_translation)
            & np.isclose(np.abs(relative_vectors[..., 1]), translation_vector[1], atol=tol_translation)
            & np.isclose(np.abs(relative_vectors[..., 2]), translation_vector[2], atol=tol_translation)
        )

        mapping_indices = np.argwhere(mask)

        # find the unique indices closest to (0,0,0) that map onto each other
        parent = np.arange(molecular_structure.n_atoms)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            ix = find(x)
            iy = find(y)
            if ix < iy:
                parent[iy] = ix
            else:
                parent[ix] = iy

        # Find connected components
        for i, j in mapping_indices:
            union(i, j)

        dists = np.linalg.norm(molecular_structure.coordinates, axis=1)

        primitive_indices = np.zeros(len(set(parent)), dtype=int)

        for i, unique_roots in enumerate(set(parent)):
            mask = parent == unique_roots
            idxs = np.where(mask)[0]
            primitive_indices[i] = idxs[np.argmin(dists[idxs])]

        return primitive_indices


#### Translation Symmetry Helper Functions ####
def to_fractional(lattice, positions):
    """
    Converts Cartesian coordinates to fractional coordinates based on the given lattice.

    Parameters:
    - lattice (array-like, shape (3, 3)):
        The lattice vectors defining the unit cell, where each row represents a lattice
        vector in Cartesian coordinates.
    - positions (array-like, shape (n, 3)):
        Cartesian coordinates of atoms or points to be converted into fractional coordinates.

    Returns:
    - fractional_positions (numpy array, shape (n, 3)):
        Fractional coordinates of the input positions, where each coordinate is expressed
        relative to the unit cell defined by the lattice vectors.

    Methodology:
    1. Compute the inverse of the lattice matrix using `np.linalg.inv`.
    2. For each position in `positions`, compute its fractional coordinates by multiplying
    the position vector with the inverse lattice matrix (`inv_lattice @ pos`).
    3. Return the resulting fractional coordinates as a NumPy array.

    Example:
    ```python
    lattice = np.array([[1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0]])
    positions = np.array([[0.5, 0.5, 0.5],
                        [1.0, 1.0, 1.0]])

    fractional_positions = to_fractional(lattice, positions)

    print("Fractional Coordinates:")
    print(fractional_positions)
    # Output:
    # [[0.5 0.5 0.5]
    #  [1.0 1.0 1.0]]
    ```

    Notes:
    - This function assumes the lattice is invertible.
    """
    inv_lattice = np.linalg.inv(lattice)
    fractional_positions = positions @ inv_lattice
    return np.array(fractional_positions)


def is_legitimate_scaled_cell(v1, v2, v3, coordinates, atomicsymbols, N1, N2, N3, tolerance=1e-7):
    """
    Checks if a scaled unit cell is legitimate by verifying that all atoms
    in the scaled lattice can be mapped into the primitive unit cell using
    periodic boundary conditions.

    Parameters:
    - v1, v2, v3 (array-like):
        The three lattice vectors defining the original crystal lattice.
    - coordinates (array-like, shape (n, 3)):
        Cartesian coordinates of the atomic positions in the original lattice.
    - atomicsymbols (list or array-like, shape (n,)):
        List of atomic symbols corresponding to the atoms in `coordinates`.
    - N1, N2, N3 (int):
        Scaling factors along the directions of lattice vectors `v1`, `v2`, and `v3`, respectively.
    - tolerance (float, optional, default=1e-5):
        Numerical tolerance used to compare atomic positions and determine equivalence.

    Returns:
    - is_legitimate (bool):
        True if the scaled unit cell is legitimate, False otherwise.
    - primitive_indices (array-like):
        Indices of the atoms that belong to the primitive unit cell.
    - scaled_lattice (array-like, shape (3, 3)):
        The scaled lattice vectors of the reduced unit cell.

    The function works as follows:
    1. Scales the lattice vectors based on the provided scaling factors.
    2. Converts Cartesian atomic coordinates to fractional coordinates in the scaled lattice.
    3. Identifies atoms within the primitive unit cell by their fractional coordinates.
    4. Generates translation vectors for periodic boundary conditions.
    5. Checks if all atoms in the scaled lattice can be mapped to the primitive unit cell,
    comparing positions and atomic symbols within the specified tolerance.
    """
    # Scale the lattice vectors
    scaled_lattice = np.row_stack((v1 / N1, v2 / N2, v3 / N3))
    # Convert atom positions to fractional coordinates in the original lattice
    fractional_positions = to_fractional(scaled_lattice, coordinates)
    # Scale and wrap fractional positions into [0,1) range for periodicity
    primitive_indices = np.where(
        (fractional_positions[:, 0] < 1 - tolerance)
        & (fractional_positions[:, 1] < 1 - tolerance)
        & (fractional_positions[:, 2] < 1 - tolerance)
    )[0]
    # Extract the candidate unit cell positions and symbols
    primitive_positions = fractional_positions[primitive_indices]
    primitive_symbols = np.array(atomicsymbols)[primitive_indices]
    # Generate all possible translation vectors within the periodic limits
    translations = [np.array([i, j, k]) for i in range(N1) for j in range(N2) for k in range(N3)]
    # Check each translated position to confirm it maps into the unit cell
    for pos, symbol in zip(fractional_positions, atomicsymbols):
        # Check if the translated positions of `pos` map into the unit cell
        found_match = False
        for translation in translations:
            translated_pos = pos - translation  # Wrap within [0,1)
            # Check if the translated position matches any point in the primitive cell
            if any(
                np.allclose(translated_pos, primitive_pos, atol=tolerance) and primitive_symbol == symbol
                for primitive_pos, primitive_symbol in zip(primitive_positions, primitive_symbols)
            ):
                found_match = True
                break  # Found a match, no need to check other translations

        # If no match found for this atom, the cell is not legitimate
        if not found_match:
            return False, primitive_indices, scaled_lattice

    return True, primitive_indices, scaled_lattice


def getPrimitiveUnitCell(cellvectors, coordinates, atomicsymbols, tolerance=1e-8):
    """
    Identifies the primitive unit cell of a crystal lattice by determining the
    smallest valid scaling factors along each lattice vector direction.

    Parameters:
    - cellvectors (array-like, shape (3, 3)):
        The three lattice vectors defining the crystal lattice.
    - coordinates (array-like, shape (n, 3)):
        Cartesian coordinates of the atomic positions in the lattice.
    - atomicsymbols (list or array-like, shape (n,)):
        List of atomic symbols corresponding to the atoms in `coordinates`.
    - tolerance (float, optional, default=1e-5):
        Numerical tolerance used to compare atomic positions and determine equivalence.
    - Nx, Ny, Nz (int, optional, default=5):
        Maximum multipliers for refining the divisors of the scaling factors
        along the x, y, and z lattice directions, respectively.

    Returns:
    - scaling_factors (tuple of 3 ints):
        The scaling factors `(Nx, Ny, Nz)` that define the primitive unit cell
        along the x, y, and z lattice directions.
    - primitive_indices (array-like):
        Indices of the atoms that belong to the primitive unit cell.
    - scaled_lattice (array-like, shape (3, 3)):
        The lattice vectors of the identified primitive unit cell.

    The function works as follows:
    1. Extracts the lattice vectors (`v1`, `v2`, `v3`) from the `cellvectors`.
    2. Iteratively tests divisors of the lattice vector lengths to find valid
    scaling factors using the `is_legitimate_scaled_cell` function:
    - Initial divisors are selected from small prime numbers (e.g., 2, 3, 5).
    - The scaling factors are adjusted to maximize the number of divisions
        that still produce a legitimate unit cell.
    3. Combines the refined scaling factors along the x, y, and z directions
    to identify the primitive unit cell.
    4. Returns the scaling factors, indices of atoms in the primitive cell,
    and the lattice vectors of the identified cell.
    """
    v1 = cellvectors[0]
    v2 = cellvectors[1]
    v3 = cellvectors[2]
    # primes=[2,3,5,6,7,8,9,10,11,13,17,1]
    primes = range(30, 0, -1)
    # x1 divisor
    for itx in primes:
        iscell, _, _ = is_legitimate_scaled_cell(v1, v2, v3, coordinates, atomicsymbols, itx, 1, 1, tolerance=tolerance)
        if iscell:
            break
    # x2 divisor
    for ity in primes:
        iscell, _, _ = is_legitimate_scaled_cell(v1, v2, v3, coordinates, atomicsymbols, 1, ity, 1, tolerance=tolerance)
        if iscell:
            break
    # x3 divisor
    for itz in primes:
        iscell, _, _ = is_legitimate_scaled_cell(v1, v2, v3, coordinates, atomicsymbols, 1, 1, itz, tolerance=tolerance)
        if iscell:
            break
    iscell, primitive_indices, scaled_lattice = is_legitimate_scaled_cell(
        v1, v2, v3, coordinates, atomicsymbols, itx, ity, itz, tolerance=tolerance
    )
    return (itx, ity, itz), primitive_indices, scaled_lattice


def getCellCoordinates(lattice, coordinates, primitive_indices, tolerance=1e-5):
    """
    Computes relative lattice translations and fractional coordinates.
     Parameters:

    - lattice (array-like, shape (3, 3)):

    The lattice vectors (columns) defining the crystal structure.

    - coordinates (array-like, shape (n, 3)):

    Cartesian coordinates of all atoms in the structure.

    - primitive_indices (list[int]):

    Indices of atoms belonging to the primitive unit cell.

    - supercell (tuple[int, int, int]):

    Supercell replication along each lattice vector (e.g., (2,2,2)).

    - tolerance (float):

    Numerical tolerance for position matching in fractional coordinates.


    Returns:

    - relative_cell_coordinates (ndarray, shape (n, 4)):

    Array of [tx, ty, tz, primitive_index] for each atom,

    where (tx, ty, tz) are integer lattice translations.

    - in_cell_coordinates (ndarray, shape (len(primitive_indices), 3)):

    Fractional coordinates of primitive atoms wrapped into [0,1).

    """
    # 1. Get all fractional coordinates
    # (Assuming to_fractional is defined externally as before)
    fractional_positions = to_fractional(lattice, coordinates)

    # 2. Get reference primitive coordinates
    # We do NOT apply floor() here. We compare raw values.
    primitive_frac_coords = fractional_positions[primitive_indices]

    # 3. Calculate differences between ALL pairs (Broadcasting)
    # Shape: (n_atoms, n_prim, 3)
    # raw_diff represents (Atom_Position - Primitive_Position)
    raw_diffs = fractional_positions[:, None, :] - primitive_frac_coords[None, :, :]

    # 4. Find the deviation from the nearest integer translation
    # If atom matches primitive, raw_diff should be like 0.0, 1.0, -1.0, 2.0, etc.
    # np.round(raw_diff) gives us the "Translation Integer candidate"
    closest_integer_translation = np.round(raw_diffs)

    # residual is the distance from that integer translation
    residual = raw_diffs - closest_integer_translation

    # 5. Check matches
    # We check if the residual is effectively zero
    is_match = np.all(np.abs(residual) < tolerance, axis=2)

    # 6. Error Handling (Uniqueness check)
    match_count = np.sum(is_match, axis=1)

    if np.any(match_count == 0):
        first_bad = np.flatnonzero(match_count == 0)[0]
        # --- DIAGNOSTIC BLOCK ---
        # Find the "best" failure to tell the user how far off they are
        # We look at the residual for the bad atom against all primitive atoms
        bad_atom_residuals = np.abs(residual[first_bad])  # Shape: (n_prim, 3)

        # Find the maximum deviation in (x,y,z) for each primitive candidate
        max_dev_per_prim = np.max(bad_atom_residuals, axis=1)

        # Find the primitive atom that came closest
        closest_prim_idx_local = np.argmin(max_dev_per_prim)
        closest_prim_global = primitive_indices[closest_prim_idx_local]
        min_deviation = max_dev_per_prim[closest_prim_idx_local]

        raise ValueError(
            f"Atom {first_bad} at {fractional_positions[first_bad]} could not be matched.\n"
            f"Closest candidate was Primitive Atom {closest_prim_global}.\n"
            f"Deviation: {min_deviation:.2e} (Tolerance: {tolerance}).\n"
            f"-> Suggestion: Increase tolerance to at least {min_deviation * 1.1:.1e}"
        )
        # ------------------------

    if np.any(match_count > 1):
        first_bad = np.flatnonzero(match_count > 1)[0]
        raise ValueError(f"Atom {first_bad} matches multiple primitive atoms. Decrease tolerance or check structure.")

    # 7. Extract Results
    # Get the index of the primitive atom (0 to n_prim-1)
    prim_indices_map = np.argmax(is_match, axis=1)

    # Retrieve the correct translation vector for that specific match
    # We pick the specific translation integer we found in step 4
    # Fancy indexing: [range(n), prim_indices_map] selects the matching column for each row
    n_atoms = len(fractional_positions)
    t_integers = closest_integer_translation[np.arange(n_atoms), prim_indices_map].astype(int)

    # 8. Format output
    relative_cell_coordinates = np.hstack((t_integers, prim_indices_map[:, None]))

    # For the return value of `in_cell_coordinates`, we usually want them wrapped [0, 1)
    # strictly for display/reference purposes.
    in_cell_coordinates = primitive_frac_coords - np.floor(primitive_frac_coords)

    return relative_cell_coordinates, in_cell_coordinates


def getTranslationOps(relative_cell_coordinates, supercell):
    """
    Computes the translation operators ( T_x ), ( T_y ), and ( T_z ) for a given set
    of relative cell coordinates in a supercell. These operators represent translations
    by one unit along the x, y, and z directions in the supercell.

    Parameters:
    - relative_cell_coordinates (list of arrays):
        A list where each element is an array of the form `[x, y, z, index]`, representing
        the relative coordinates `(x, y, z)` of an atom in the supercell and its
        corresponding index in the primitive cell.
    - supercell (array-like, shape (3,)):
        The dimensions of the supercell `[Nx, Ny, Nz]`, where ( Nx, Ny, Nz ) are the
        number of units along the x, y, and z directions, respectively.

    Returns:
    - Tx, Ty, Tz (numpy arrays, shape (n, n)):
        Translation matrices along the x, y, and z directions, respectively. Each matrix
        has shape `(n, n)` where ( n ) is the number of elements in `relative_cell_coordinates`.
        The element ( T_x[i, j] = 1 ) indicates that applying a translation along x to
        the atom at index ( j ) results in the atom at index ( i ). Similar logic applies
        to ( T_y ) and ( T_z ).

    Raises:
    - ValueError:
        If a translated position does not match any of the elements in `relative_cell_coordinates`.

    Methodology:
    1. Initialize zero matrices `Tx`, `Ty`, and `Tz` of size `(n, n)`.
    2. For each relative cell coordinate in `relative_cell_coordinates`:
       - Compute the translated coordinates by adding 1 to the x, y, and z components,
         wrapping around using `np.mod` for periodic boundary conditions.
       - Find the matching relative cell coordinate in the list for each translation
         (x, y, z) and update the corresponding entry in `Tx`, `Ty`, or `Tz`.
    3. If a match is not found for any translation, raise a `ValueError`.

    Example:
    ```python
    relative_cell_coordinates = [
        np.array([0, 0, 0, 0]),
        np.array([1, 0, 0, 0]),
        np.array([0, 1, 0, 0]),
        np.array([0, 0, 1, 0]),
    ]
    supercell = [2, 2, 2]

    Tx, Ty, Tz = getTranslationOps(relative_cell_coordinates, supercell)

    print("Tx:")
    print(Tx)
    print("Ty:")
    print(Ty)
    print("Tz:")
    print(Tz)
    ```
    """
    supercell = np.asarray(supercell, dtype=int)

    Tx = np.zeros((len(relative_cell_coordinates), len(relative_cell_coordinates)))
    Ty = np.zeros((len(relative_cell_coordinates), len(relative_cell_coordinates)))
    Tz = np.zeros((len(relative_cell_coordinates), len(relative_cell_coordinates)))
    xFlag = False
    yFlag = False
    zFlag = False
    for it1, rel_cell_coo1 in enumerate(relative_cell_coordinates):
        shiftedx = np.array(
            [
                np.mod(rel_cell_coo1[0] + 1, supercell[0]),
                rel_cell_coo1[1],
                rel_cell_coo1[2],
                rel_cell_coo1[3],
            ]
        )
        shiftedy = np.array(
            [
                rel_cell_coo1[0],
                np.mod(rel_cell_coo1[1] + 1, supercell[1]),
                rel_cell_coo1[2],
                rel_cell_coo1[3],
            ]
        )
        shiftedz = np.array(
            [
                rel_cell_coo1[0],
                rel_cell_coo1[1],
                np.mod(rel_cell_coo1[2] + 1, supercell[2]),
                rel_cell_coo1[3],
            ]
        )
        Txflag = False
        Tyflag = False
        Tzflag = False
        for it2, rel_cell_coo2 in enumerate(relative_cell_coordinates):
            if (shiftedx == rel_cell_coo2).all() and not Txflag:
                Tx[it2, it1] = 1.0
                Txflag = True
                xFlag = True
            elif (shiftedy == rel_cell_coo2).all() and not Tyflag:
                Ty[it2, it1] = 1.0
                Tyflag = True
                yFlag = True
            elif (shiftedz == rel_cell_coo2).all() and not Tzflag:
                Tz[it2, it1] = 1.0
                Tzflag = True
                zFlag = True
    return Tx, Ty, Tz, xFlag, yFlag, zFlag


#### Inversion Symmetry Helper Functions ####
def find_negated_vector(vectors, target_vector, tolerance=1e-5):
    """
    Determines if the negation of a target vector exists within a list of vectors.

    Args:
        vectors (list): List of numpy arrays representing 3D coordinates.
        target_vector (numpy array): Vector to check for its negation in vectors.
        tolerance (float): Tolerance for floating-point comparison.

    Returns:
        tuple:
            - bool: True if negation exists within tolerance, False otherwise.
            - int or None: Index of the negated vector if found, None otherwise.
    """
    for idx, vec in enumerate(vectors):
        if np.allclose(-target_vector, vec, atol=tolerance):
            return True, idx
    return False, None


def detect_inversion_symmetry(centered_coords, atomic_symbols, tolerance=1e-5):
    """
    Checks if a set of atomic coordinates has inversion symmetry by verifying if each coordinate's
    negation exists in the set with the same atomic symbol.

    Args:
        centered_coords (list): List of numpy arrays, each representing an atom's 3D coordinates.
        atomic_symbols (list): List of atomic symbols corresponding to each coordinate in centered_coords.
        tolerance (float): Tolerance for floating-point precision in symmetry detection.

    Returns:
        bool: True if inversion symmetry is detected, False otherwise.
        dict: Dictionary mapping each index to its inversion pair index if symmetry exists, empty if not.
    """
    inversion_pairs = {}

    for idx, coord in enumerate(centered_coords):
        # Check if the negated coordinates are in the list
        negation_found, neg_idx = find_negated_vector(centered_coords, coord, tolerance=tolerance)

        # Verify the atomic symbols match for symmetry
        if negation_found and atomic_symbols[idx] == atomic_symbols[neg_idx]:
            inversion_pairs[idx] = neg_idx
        else:
            return False, {}  # No symmetry if any atom lacks a valid inversion pair

    return True, inversion_pairs


def get_Inversion_Symmetry_Generator(pairs_pairs, n_atoms):
    """
    Generates a permutation matrix based on detected inversion pairs.

    Args:
        inversion_pairs (dict): Dictionary mapping each atom index to its inversion pair index.
        n_atoms (int): Total number of atoms in the system.

    Returns:
        numpy array: Permutation matrix implementing inversion symmetry.
    """
    perm_matrix = np.zeros((n_atoms, n_atoms), dtype=float)

    for idx, inv_idx in pairs_pairs.items():
        perm_matrix[idx, inv_idx] = 1.0
    return perm_matrix


#### Rotation Symmetry Helper Functions ####
def rotate_coords(coords, axis, angle_rad):
    """
    Rotates coordinates around a given axis by a given angle (in radians).
    """
    c, s = np.cos(angle_rad), np.sin(angle_rad)

    if axis == "x":
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == "y":
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif axis == "z":
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")

    return [R @ coord for coord in coords]


def detect_rotational_symmetry(centered_coords, atomic_symbols, axis="z", n=2, tolerance=1e-5):
    """
    Checks for Cn rotational symmetry about a specified axis.

    Args:
        centered_coords (list): List of numpy arrays, each representing an atom's 3D coordinates.
        atomic_symbols (list): List of atomic symbols corresponding to each coordinate.
        axis (str): Axis of rotation: 'x', 'y', or 'z'.
        n (int): Order of rotation (C_n means 360/n degrees).
        tolerance (float): Tolerance for matching atom positions.

    Returns:
        bool: True if Cn rotational symmetry is detected.
        dict: Mapping of original atom indices to rotated counterparts if symmetry exists.
    """
    angle_rad = 2 * np.pi / n
    rotated_coords = rotate_coords(centered_coords, axis, angle_rad)
    return find_rotated_match(rotated_coords, centered_coords, atomic_symbols, tolerance=tolerance)


def find_rotated_match(rotated_coords, original_coords, atomic_symbols, tolerance=1e-5):
    """
    Attempts to match rotated coordinates back to original ones by atom type and position.
    Returns True and a mapping if a complete match is found.
    """
    used_indices = set()
    rotation_pairs = {}

    for i, (rot_coord, symbol) in enumerate(zip(rotated_coords, atomic_symbols)):
        matched = False
        for j, (orig_coord, orig_symbol) in enumerate(zip(original_coords, atomic_symbols)):
            if j in used_indices:
                continue
            if orig_symbol != symbol:
                continue
            if np.linalg.norm(rot_coord - orig_coord) < tolerance:
                rotation_pairs[i] = j
                used_indices.add(j)
                matched = True
                break
        if not matched:
            return False, {}

    return True, rotation_pairs


#### Mirror Symmetry Helper Functions ####
def reflect_coords(coords, axis):
    """
    Reflects coordinates across the specified axis-aligned mirror plane.
    """
    if axis == "x":
        R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    elif axis == "y":
        R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    elif axis == "z":
        R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")

    return [R @ coord for coord in coords]


def detect_mirror_symmetry(centered_coords, atomic_symbols, axis="z", tolerance=1e-5):
    """
    Detects mirror symmetry across a specified axis-aligned plane.

    Args:
        centered_coords (list): List of numpy arrays of atomic coordinates.
        atomic_symbols (list): List of atomic symbols.
        axis (str): Mirror plane normal direction: 'x', 'y', or 'z'.
        tolerance (float): Distance tolerance for matching atoms.

    Returns:
        bool: True if mirror symmetry is detected.
        dict: Mapping of original atom indices to reflected counterparts if symmetry exists.
    """
    reflected_coords = reflect_coords(centered_coords, axis)
    return find_rotated_match(reflected_coords, centered_coords, atomic_symbols, tolerance=tolerance)


def wrapping_coordinates(coordinates, cellvectors):
    """
    Wraps atomic coordinates into the unit cell defined by the given cell vectors.

    Parameters:
    - coordinates (array-like, shape (n, 3)):
        Cartesian coordinates of atoms.
    - cellvectors (array-like, shape (3, 3)):
        Lattice vectors defining the unit cell.

    Returns:
    - wrapped_coordinates (numpy array, shape (n, 3)):
        Coordinates wrapped into the unit cell.
    """
    inv_cell = np.linalg.inv(cellvectors)
    fractional_coords = coordinates @ inv_cell
    wrapped_fractional_coords = fractional_coords % 1.0
    wrapped_coordinates = wrapped_fractional_coords @ cellvectors
    return np.round(wrapped_coordinates, decimals=6)
