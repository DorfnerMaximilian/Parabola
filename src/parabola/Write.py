import numpy as np
from . import PhysConst
import os


def write_xyz_file(
    atomic_symbols,
    coordinates,
    filename,
    comment=None,
    overwrite=False,
    append=False,
    success_comment="Successfully wrote XYZ file to",
):
    """
    Writes or appends atomic coordinates to a standard .xyz file.

    Args:
        atomic_symbols (list[str]): A list of atomic symbols (e.g., ['C', 'H', 'H']).
        coordinates (list[list[float]] or np.ndarray): An Nx3 list or NumPy array of atomic coordinates.
        filename (str): The name for the output file (e.g., 'molecule.xyz').
        comment (str, optional): A comment to be written on the second line of the frame.
                                 If None, a default comment with the atom count is used.
        overwrite (bool, optional): If True, overwrites the file if it exists. Has no effect if append is True.
                                    Defaults to False.
        append (bool, optional): If True, appends the coordinates as a new frame to the end of the file.
                                 If the file doesn't exist, it is created. Defaults to False.
    """

    # Validate that the number of symbols matches the number of coordinates
    num_atoms = len(atomic_symbols)
    if num_atoms != len(coordinates):
        raise ValueError(
            "The number of atomic symbols must match the number of coordinates."
        )

    # Determine the file writing mode ('w' for write/overwrite, 'a' for append)
    if append:
        file_mode = "a"
        # Adjust success message for append mode
        current_success_comment = "Successfully appended to"
        full_path = filename
    else:
        file_mode = "w"
        current_success_comment = success_comment
        full_path = filename
        # Check if the file exists and handle the overwrite logic
        if not overwrite and os.path.exists(full_path):
            base, ext = os.path.splitext(filename)
            # Ensure the extension is .xyz if not present
            new_filename = f"{base}_new{ext if ext else '.xyz'}"
            full_path = new_filename
            print(f"⚠️  File '{filename}' exists. Saving as '{full_path}'.")

    # Prepare the comment line for the XYZ file frame
    if comment is None:
        comment = f"Frame written by write_xyz_file"

    # Use a 'with' statement for safe and automatic file handling
    try:
        with open(full_path, file_mode) as xyz_file:
            # Write the header: number of atoms and a comment line
            xyz_file.write(f"{num_atoms}\n")
            xyz_file.write(f"{comment}\n")

            # Write the atomic coordinates
            for symbol, coord in zip(atomic_symbols, coordinates):
                # Format the line for clean, readable output
                x, y, z = coord
                xyz_file.write(f"{symbol:<4} {x:12.8f} {y:12.8f} {z:12.8f}\n")

        print(f"✅ {current_success_comment} '{full_path}'")

    except IOError as e:
        print(f"❌ Error writing file: {e}")


def write_cube_file(x, y, z, data, atoms, filename="test.cube", parentfolder="./"):
    """Function to write a .cube file that is readable by e.g. Jmol, the origin is assumed to be [0.0,0.0,0.0]
    input:   data:             (np.array)              Nx x Ny x Nz numpy array with the wave function coefficients at each gridpoint
    (opt.)   parentfolder:     (str)                   path to the .inp file of the cp2k calculation to read in the cell dimensions
    output:  f                 (np.array)              Nx x Ny x Nz numpy array where first index is x, second y and third is z
    """
    numofatoms = len(atoms)
    Nx = np.shape(data)[0]
    Ny = np.shape(data)[1]
    Nz = np.shape(data)[2]
    origin = [x[0], y[0], z[0]]
    # Calculate voxel spacings — assuming uniform grid spacing
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]
    with open(parentfolder + "/" + filename, "w") as file:
        file.write("Cube File generated with Parabola\n")
        file.write("OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n")
        file.write(
            format(numofatoms, "5.0f")
            + format(origin[0], "12.6f")
            + format(origin[1], "12.6f")
            + format(origin[2], "12.6f")
            + "\n"
        )
        # Number of voxels and axis vectors
        file.write(f"{Nx:5d} {dx:12.6f} {0.0:12.6f} {0.0:12.6f}\n")
        file.write(f"{Ny:5d} {0.0:12.6f} {dy:12.6f} {0.0:12.6f}\n")
        file.write(f"{Nz:5d} {0.0:12.6f} {0.0:12.6f} {dz:12.6f}\n")
        for atom in atoms:
            file.write(
                format(AtomSymbolToAtomnumber(atom[1]), "5.0f")
                + format(0.0, "12.6f")
                + format(ConversionFactors["A->a.u."] * atom[2], "12.6f")
                + format(ConversionFactors["A->a.u."] * atom[3], "12.6f")
                + format(conFactors["A->a.u."] * atom[4], "12.6f")
                + "\n"
            )
        for itx in range(Nx):
            for ity in range(Ny):
                for itz in range(Nz):
                    if (itx or ity or itz) and itz % 6 == 0:
                        file.write("\n")
                    file.write(" {0: .5E}".format(data[itx, ity, itz]))


def write_mol_file(
    normalmodeEnergies, normalmodes, normfactors, coordinates, atoms, parentfolder="./"
):
    """
    Function to generate a .mol file for use with e.g., Jmol.

    Parameters:
    - normalmodeEnergies (np.array): The normal mode energies as a numpy array.
    - normalmodes (np.array): Normalized Cartesian displacements, i.e., vectors proportional to the Cartesian displacements.
                              v = sqrt(M^(-1)) @ X, where X are the eigenvectors of the rescaled Hessian.
    - normfactors (np.array): The normalization factors n = sqrt(dot(v, v)).
    - parentfolder (str, optional): The parent folder where the .mol file will be created. Default is "./".

    Notes:
    - The function relies on the `ConversionFactors` class.
    - The function looks for a single .xyz file in the specified directory and reads the atomic coordinates from it.
    - The .mol file is generated with sections for frequencies, atomic coordinates, normalization factors, and vibrational modes.
    - The generated .mol file is named "Vibrations.mol" and is saved in the specified parent folder.
    """
    ##########################################
    # Get the number of atoms from the xyz file
    ##########################################
    numofatoms = len(coordinates)
    with open(parentfolder + "/Vibrations.mol", "w") as f:
        f.write("[Molden Format]\n")
        f.write("[FREQ]\n")
        for Frequency in normalmodeEnergies:
            f.write("   " + str(Frequency) + "\n")
        f.write("[FR-COORD]\n")
        for it, coord in enumerate(coordinates):
            f.write(
                atoms[it]
                + "   "
                + str(coord[0] * ConversionFactors["A->a.u."])
                + "   "
                + str(coord[1] * ConversionFactors["A->a.u."])
                + "   "
                + str(coord[2] * ConversionFactors["A->a.u."])
                + "\n"
            )
        f.write("[NORM-FACTORS]\n")
        for normfactor in normfactors:
            f.write(str(normfactor) + "\n")
        f.write("[FR-NORM-COORD]\n")
        modeiter = 1
        for mode in normalmodes:
            f.write("vibration      " + str(modeiter) + "\n")
            for s in range(numofatoms):
                f.write(
                    "   "
                    + str(round(mode[3 * s], 12))
                    + "   "
                    + str(round(mode[3 * s + 1], 12))
                    + "   "
                    + str(round(mode[3 * s + 2], 12))
                    + "\n"
                )
            modeiter += 1
