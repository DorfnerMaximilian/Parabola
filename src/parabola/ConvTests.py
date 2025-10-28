import os
import numpy as np
from . import Geometry
from . import Read
from . import AtomicBasis
import matplotlib.pyplot as plt

# For standard Latex fonts
plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
# get the environmental variable
pathtocp2k = os.environ["cp2kpath"]
pathtobinaries = pathtocp2k + "/exe/local/"


# -------------------------------------------------------------------------
def changeR_Cutoff(origin, source, RCutoff):
    inp_files = [f for f in os.listdir(origin) if f.endswith(".inp")]
    if len(inp_files) != 1:
        raise ValueError(
            "InputError: There should be only one inp file in the current directory"
        )
    filename = inp_files[0]
    with open(filename, "r") as f:
        with open(source, "w") as g:
            for line in f.readlines():
                if len(line.split()) >= 1:
                    if line.split()[0] == "CUTOFF_RADIUS":
                        line = "\tCUTOFF_RADIUS " + str(RCutoff) + "\n"
                g.write(line)


# -------------------------------------------------------------------------
def R_CutoffTest_inputs(
    RCutoffs, parentpath="./", binaryloc=pathtobinaries, binary="cp2k.popt"
):
    # get the Projectname:
    inp_files = [f for f in os.listdir(parentpath) if f.endswith(".inp")]
    if len(inp_files) != 1:
        raise ValueError(
            "InputError: There should be only one .inp file in the current directory"
        )
    inpfilename = inp_files[0]
    Projectname = "emptyString"
    with open(inpfilename, "r") as f:
        for lines in f:
            if len(lines.split()) > 0:
                if lines.split()[0] == "PROJECT":
                    Projectname = lines.split()[1]
    if Projectname == "emptyString":
        raise ValueError("InputError: Projectname not found!")
    # get xyzfile
    xyz_files = [f for f in os.listdir(parentpath) if f.endswith(".xyz")]
    if len(xyz_files) != 1:
        raise ValueError(
            "InputError: There should be exactly one .xyz file in the current directory"
        )
    xyzfilename = xyz_files[0]
    Restart_files = [f for f in os.listdir(parentpath) if f.endswith("-RESTART.wfn")]
    if len(Restart_files) != 1:
        raise ValueError(
            "InputError: There should be exactly one Restart file in the current directory"
        )
    Restart_filename = Restart_files[0]
    if Restart_filename != Projectname + "-RESTART.wfn":
        raise ValueError(
            "InputError: Project- and Restartfilename differ! Reconsider your input."
        )
    for R_Cutoff in RCutoffs:
        work_dir = "Cutoff_" + str(R_Cutoff) + "A"
        if not os.path.isdir(parentpath + work_dir):
            os.mkdir(parentpath + work_dir)
        else:
            filelist = [f for f in os.listdir(parentpath + work_dir)]
            for f in filelist:
                os.remove(parentpath + work_dir + "/" + f)
        changeR_Cutoff(parentpath, parentpath + work_dir + "/input_file.inp", R_Cutoff)
        os.system("cp " + parentpath + xyzfilename + " " + parentpath + work_dir)
        os.system("cp " + parentpath + Restart_filename + " " + parentpath + work_dir)
        Geometry.centerMolecule(parentpath + work_dir)
        os.system(
            "ln -s " + binaryloc + "/" + binary + " " + parentpath + work_dir + "/"
        )


# -------------------------------------------------------------------------
def changeRelCutoff(origin, source, RelCutoff):
    inp_files = [f for f in os.listdir(origin) if f.endswith(".inp")]
    if len(inp_files) != 1:
        raise ValueError(
            "InputError: There should be only one inp file in the current directory"
        )
    filename = inp_files[0]
    with open(filename, "r") as f:
        with open(source, "w") as g:
            for line in f.readlines():
                if len(line.split()) >= 1:
                    if line.split()[0] == "REL_CUTOFF":
                        line = "\tREL_CUTOFF " + str(RelCutoff) + "\n"
                g.write(line)


# -------------------------------------------------------------------------
def RelCutoffTest_inputs(
    RelCutoffs, parentpath="./", binaryloc=pathtobinaries, binary="cp2k.popt"
):
    # get the Projectname:
    inp_files = [f for f in os.listdir(parentpath) if f.endswith(".inp")]
    if len(inp_files) != 1:
        raise ValueError(
            "InputError: There should be only one .inp file in the current directory"
        )
    inpfilename = inp_files[0]
    Projectname = "emptyString"
    with open(inpfilename, "r") as f:
        for lines in f:
            if len(lines.split()) > 0:
                if lines.split()[0] == "PROJECT":
                    Projectname = lines.split()[1]
    if Projectname == "emptyString":
        raise ValueError("InputError: Projectname not found!")
    # get xyzfile
    xyz_files = [f for f in os.listdir(parentpath) if f.endswith(".xyz")]
    if len(xyz_files) != 1:
        raise ValueError(
            "InputError: There should be exactly one .xyz file in the current directory"
        )
    xyzfilename = xyz_files[0]
    Restart_files = [f for f in os.listdir(parentpath) if f.endswith("-RESTART.wfn")]
    if len(Restart_files) != 1:
        raise ValueError(
            "InputError: There should be exactly one Restart file in the current directory"
        )
    Restart_filename = Restart_files[0]
    if Restart_filename != Projectname + "-RESTART.wfn":
        raise ValueError(
            "InputError: Project- and Restartfilename differ! Reconsider your input."
        )
    for RelCutoff in RelCutoffs:
        work_dir = "Rel_Cutoff_" + str(RelCutoff) + "Ry"
        if not os.path.isdir(parentpath + work_dir):
            os.mkdir(parentpath + work_dir)
        else:
            filelist = [f for f in os.listdir(parentpath + work_dir)]
            for f in filelist:
                os.remove(parentpath + work_dir + "/" + f)
        changeRelCutoff(
            parentpath, parentpath + work_dir + "/input_file.inp", RelCutoff
        )
        os.system("cp " + parentpath + xyzfilename + " " + parentpath + work_dir)
        os.system("cp " + parentpath + Restart_filename + " " + parentpath + work_dir)
        Geometry.centerMolecule(parentpath + work_dir)
        os.system(
            "ln -s " + binaryloc + "/" + binary + " " + parentpath + work_dir + "/"
        )


# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
def changeCutoff(origin, source, Cutoff):
    inp_files = [f for f in os.listdir(origin) if f.endswith(".inp")]
    if len(inp_files) != 1:
        raise ValueError(
            "InputError: There should be only one inp file in the current directory"
        )
    filename = inp_files[0]
    with open(filename, "r") as f:
        with open(source, "w") as g:
            for line in f.readlines():
                if len(line.split()) >= 1:
                    if line.split()[0] == "CUTOFF":
                        line = "\tCUTOFF " + str(Cutoff) + "\n"
                g.write(line)


# -------------------------------------------------------------------------
def CutoffTest_inputs(
    Cutoffs, parentpath="./", binaryloc=pathtobinaries, binary="cp2k.popt"
):
    # get the Projectname:
    inp_files = [f for f in os.listdir(parentpath) if f.endswith(".inp")]
    if len(inp_files) != 1:
        raise ValueError(
            "InputError: There should be only one .inp file in the current directory"
        )
    inpfilename = inp_files[0]
    Projectname = "emptyString"
    with open(inpfilename, "r") as f:
        for lines in f:
            if len(lines.split()) > 0:
                if lines.split()[0] == "PROJECT":
                    Projectname = lines.split()[1]
    if Projectname == "emptyString":
        raise ValueError("InputError: Projectname not found!")
    # get xyzfile
    xyz_files = [f for f in os.listdir(parentpath) if f.endswith(".xyz")]
    if len(xyz_files) != 1:
        raise ValueError(
            "InputError: There should be exactly one .xyz file in the current directory"
        )
    xyzfilename = xyz_files[0]
    Restart_files = [f for f in os.listdir(parentpath) if f.endswith("-RESTART.wfn")]
    if len(Restart_files) != 1:
        raise ValueError(
            "InputError: There should be exactly one Restart file in the current directory"
        )
    Restart_filename = Restart_files[0]
    if Restart_filename != Projectname + "-RESTART.wfn":
        raise ValueError(
            "InputError: Project- and Restartfilename differ! Reconsider your input."
        )
    for Cutoff in Cutoffs:
        work_dir = "Cutoff_" + str(Cutoff) + "Ry"
        if not os.path.isdir(parentpath + work_dir):
            os.mkdir(parentpath + work_dir)
        else:
            filelist = [f for f in os.listdir(parentpath + work_dir)]
            for f in filelist:
                os.remove(parentpath + work_dir + "/" + f)
        changeCutoff(parentpath, parentpath + work_dir + "/input_file.inp", Cutoff)
        os.system("cp " + parentpath + xyzfilename + " " + parentpath + work_dir)
        os.system("cp " + parentpath + Restart_filename + " " + parentpath + work_dir)
        Geometry.centerMolecule(parentpath + work_dir)
        os.system(
            "ln -s " + binaryloc + "/" + binary + " " + parentpath + work_dir + "/"
        )


# -------------------------------------------------------------------------
def changeCellSize(origin, source, CellX, CellY, CellZ):
    inp_files = [f for f in os.listdir(origin) if f.endswith(".inp")]
    if len(inp_files) != 1:
        raise ValueError(
            "InputError: There should be only one inp file in the current directory"
        )
    filename = inp_files[0]
    with open(filename, "r") as f:
        with open(source, "w") as g:
            for line in f.readlines():
                if len(line.split()) >= 1:
                    if line.split()[0] == "ABC":
                        line = (
                            "\tABC "
                            + str(CellX)
                            + " "
                            + str(CellY)
                            + " "
                            + str(CellZ)
                            + "\n"
                        )
                g.write(line)


# -------------------------------------------------------------------------
def CellSizeTest_inputs(
    Cell_Dims, parentpath="./", binaryloc=pathtobinaries, binary="cp2k.popt"
):
    # get xyzfile
    xyz_files = [f for f in os.listdir(parentpath) if f.endswith(".xyz")]
    if len(xyz_files) != 1:
        raise ValueError(
            "InputError: There should be only one inp file in the current directory"
        )
    xyzfilename = xyz_files[0]

    for celldim in Cell_Dims:
        work_dir = (
            "CellDim_"
            + str(celldim[0])
            + "x"
            + str(celldim[1])
            + "x"
            + str(celldim[2])
            + "A"
        )
        if not os.path.isdir(parentpath + work_dir):
            os.mkdir(parentpath + work_dir)
        else:
            filelist = [f for f in os.listdir(parentpath + work_dir)]
            for f in filelist:
                os.remove(parentpath + work_dir + "/" + f)
        Restart_files = [
            f for f in os.listdir(parentpath + "/") if f.endswith("-RESTART.wfn")
        ]
        if len(Restart_files) != 1:
            raise ValueError(
                "InputError: There should be exactly one Restart file in the current directory"
            )
        Restart_filename = Restart_files[0]
        changeCellSize(
            parentpath,
            parentpath + work_dir + "/input_file.inp",
            celldim[0],
            celldim[1],
            celldim[2],
        )
        os.system("cp " + parentpath + xyzfilename + " " + parentpath + work_dir)
        os.system("cp " + parentpath + Restart_filename + " " + parentpath + work_dir)
        Geometry.centerMolecule(parentpath + work_dir)
        os.system(
            "ln -s " + binaryloc + "/" + binary + " " + parentpath + work_dir + "/"
        )


def CheckConvergence(quantity, path="./"):
    ##Function to check the convergence of DFT total energie w.r.t. different
    ##numerical input parameters
    ## input:   Quantity                                                    (string,'PW_Cutoff','rel_Cutoff','Cutoff_Rad',Cell_Size)
    ## (opt.)   path   path to the folder of the cooresponding calculation  (string)
    ## output:  -               (void)

    if quantity == "PW_Cutoff":
        Cutoff_dirs = [f for f in os.listdir("./") if f.endswith("Ry")]
        Cutoffs = []
        Energies = []
        xyzfile = [f for f in os.listdir(path) if f.endswith(".xyz")]
        if len(xyzfile) != 1:
            raise ValueError(
                "InputError: There should be only one inp file in the current directory"
            )
        with open(path + "/" + xyzfile[0]) as g:
            lines = g.readlines()
            numofatoms = int(lines[0])
        for dirs in Cutoff_dirs:
            currentpath = path + "/" + dirs
            inpfile = [f for f in os.listdir(currentpath) if f.endswith(".inp")]
            if len(inpfile) != 1:
                raise ValueError(
                    "InputError: There should be only one inp file in the current directory"
                )
            with open(path + "/" + dirs + "/" + inpfile[0]) as g:
                lines = g.readlines()
                for line in lines:
                    if len(line.split()) > 0:
                        if line.split()[0] == "CUTOFF" or line.split()[0] == "Cutoff":
                            Cutoffs.append(float(line.split()[1]))
            g.close()

            outfile = [f for f in os.listdir(currentpath) if f.endswith(".out")]
            if len(outfile) != 1:
                raise ValueError(
                    "InputError: There should be only one out file in the current directory"
                )
            with open(path + "/" + dirs + "/" + outfile[0]) as g:
                lines = g.readlines()
                for line in lines:
                    if len(line.split()) > 0:
                        if (
                            line.split()[0] == "ENERGY|"
                            and line.split()[1] == "Total"
                            and line.split()[2] == "FORCE_EVAL"
                            and line.split()[3] == "("
                        ):
                            Energies.append(float(line.split()[8]))
                            print(line.split())
            g.close()
        sorted_indices = np.argsort(Cutoffs)
        Cutoffs = np.array(Cutoffs)[sorted_indices]
        Energies = np.array(Energies)[sorted_indices]
        diffofE = (
            np.array(
                [
                    np.abs(Energies[it] - (Energies[-1])) / numofatoms
                    for it in range(len(Energies) - 1)
                ]
            )
            * 2.72113838565563e04
        )
        plt.scatter(Cutoffs[0:-1], diffofE, s=70)
        plt.yscale("log")
        plt.ylabel(
            r"$\vert E_{\text{tot}}(\Delta)-E_{ \text{tot}}($"
            + str(Cutoffs[-1])
            + r"$\text{Ry})\vert $ $[\text{meV}]$",
            fontsize=30,
        )
        plt.xlabel(r"Plane-Wave-Cutoff $\Delta$ [\text{Ry}]", fontsize=30)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.grid()
        plt.show()
    elif quantity == "Rel_Cutoff":
        Cutoff_dirs = [f for f in os.listdir("./") if f.endswith("Ry")]
        Cutoffs = []
        Energies = []
        xyzfile = [f for f in os.listdir(path) if f.endswith(".xyz")]
        if len(xyzfile) != 1:
            raise ValueError(
                "InputError: There should be only one inp file in the current directory"
            )
        with open(path + "/" + xyzfile[0]) as g:
            lines = g.readlines()
            numofatoms = int(lines[0])
        for dirs in Cutoff_dirs:
            currentpath = path + "/" + dirs
            inpfile = [f for f in os.listdir(currentpath) if f.endswith(".inp")]
            if len(inpfile) != 1:
                raise ValueError(
                    "InputError: There should be only one inp file in the current directory"
                )
            with open(path + "/" + dirs + "/" + inpfile[0]) as g:
                lines = g.readlines()
                for line in lines:
                    if len(line.split()) > 0:
                        if (
                            line.split()[0] == "REL_CUTOFF"
                            or line.split()[0] == "Rel_Cutoff"
                        ):
                            Cutoffs.append(float(line.split()[1]))
            g.close()

            outfile = [f for f in os.listdir(currentpath) if f.endswith(".out")]
            if len(outfile) != 1:
                raise ValueError(
                    "InputError: There should be only one out file in the current directory"
                )
            with open(path + "/" + dirs + "/" + outfile[0]) as g:
                lines = g.readlines()
                for line in lines:
                    if len(line.split()) > 0:
                        if (
                            line.split()[0] == "ENERGY|"
                            and line.split()[1] == "Total"
                            and line.split()[2] == "FORCE_EVAL"
                            and line.split()[3] == "("
                        ):
                            Energies.append(float(line.split()[8]))
                            print(line.split())
            g.close()
        sorted_indices = np.argsort(Cutoffs)
        Cutoffs = np.array(Cutoffs)[sorted_indices]
        Energies = np.array(Energies)[sorted_indices]
        diffofE = (
            np.array(
                [
                    np.abs(Energies[it] - (Energies[-1])) / numofatoms
                    for it in range(len(Energies) - 1)
                ]
            )
            * 2.72113838565563e04
        )
        plt.scatter(Cutoffs[0:-1], diffofE, s=70)
        plt.yscale("log")
        plt.ylabel(
            r"$\vert E_{\text{tot}}(\Delta)-E_{ \text{tot}}($"
            + str(Cutoffs[-1])
            + r"$\text{Ry})\vert $ $[\text{meV}]$",
            fontsize=30,
        )
        plt.xlabel(r"Relative Cutoff $\Delta$ [\text{Ry}]", fontsize=30)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.grid()
        plt.show()
    elif quantity == "Cutoff_Radius":
        Cutoff_dirs = [f for f in os.listdir("./") if f.endswith("A")]
        Cutoffs = []
        Energies = []
        xyzfile = [f for f in os.listdir(path) if f.endswith(".xyz")]
        if len(xyzfile) != 1:
            raise ValueError(
                "InputError: There should be only one .xyz file in the current directory"
            )
        with open(path + "/" + xyzfile[0]) as g:
            lines = g.readlines()
            numofatoms = int(lines[0])
        for dirs in Cutoff_dirs:
            currentpath = path + "/" + dirs
            inpfile = [f for f in os.listdir(currentpath) if f.endswith(".inp")]
            if len(inpfile) != 1:
                raise ValueError(
                    "InputError: There should be only one inp file in the current directory"
                )
            with open(path + "/" + dirs + "/" + inpfile[0]) as g:
                lines = g.readlines()
                for line in lines:
                    if len(line.split()) > 0:
                        if line.split()[0] == "CUTOFF_RADIUS":
                            Cutoffs.append(float(line.split()[1]))
            g.close()

            outfile = [f for f in os.listdir(currentpath) if f.endswith(".out")]
            if len(outfile) != 1:
                raise ValueError(
                    "InputError: There should be only one .out file in the current directory"
                )
            with open(path + "/" + dirs + "/" + outfile[0]) as g:
                lines = g.readlines()
                for line in lines:
                    if len(line.split()) > 0:
                        if (
                            line.split()[0] == "ENERGY|"
                            and line.split()[1] == "Total"
                            and line.split()[2] == "FORCE_EVAL"
                            and line.split()[3] == "("
                        ):
                            Energies.append(float(line.split()[8]))
                            print(line.split())
            g.close()
        sorted_indices = np.argsort(Cutoffs)
        Cutoffs = np.array(Cutoffs)[sorted_indices]
        Energies = np.array(Energies)[sorted_indices]
        diffofE = (
            np.array(
                [
                    np.abs(Energies[it] - (Energies[-1])) / numofatoms
                    for it in range(len(Energies) - 1)
                ]
            )
            * 2.72113838565563e04
        )
        plt.scatter(Cutoffs[0:-1], diffofE, s=70)
        plt.yscale("log")
        plt.ylabel(
            r"$\vert E_{\text{tot}}(R_c)-E_{ \text{tot}}($"
            + str(Cutoffs[-1])
            + r"$\textup{~\AA})\vert $ $[\text{meV}]$",
            fontsize=30,
        )
        plt.xlabel(r"$R_c$ $[\textup{~\AA}]$", fontsize=30)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.grid()
        plt.show()
    elif quantity == "Cell_Size":
        folders = [f for f in os.listdir("./") if f.endswith("A")]
        cellsizes = []
        absre = []
        SameSizeFlag = False
        OverlapMatrixFlag = False
        for folder in folders:
            print(folder)
            strcut = folder[8:]
            if "x" not in strcut[:-1]:
                SameSizeFlag = True
                cellsize = float(strcut[:-1])
                cellsizes.append(cellsize)
                _, _, OLM = Read.read_ks_matrices(path + folder)
                print("Reading in Overlapmatrix -> Done")
                Atoms = Read.read_atomic_coordinates(path + folder)
                print("Reading in Atomic Coordinates -> Done")
                Basis = AtomicBasis.getBasis(path + folder)
                print(
                    "Construct carthesian Basis and spherical to cartesian Transformations -> Done"
                )
                if OverlapMatrixFlag == False:
                    Overlapmatrix = AtomicBasis.getTransformationmatrix(
                        Atoms, Atoms, Basis
                    )
                    OverlapMatrixFlag = True
                diff = np.abs(Overlapmatrix - OLM)
                absre.append(np.max(np.max(diff)))
            else:
                cellsize = (float(strcut[0:2]), float(strcut[3:5]), float(strcut[6:-1]))
                cellsizes.append(cellsize)
                _, _, OLM = Read.read_ks_matrices(path + folder)
                print("Reading in Overlapmatrix -> Done")
                Atoms = Read.read_atomic_coordinates(path + folder)
                print("Reading in Atomic Coordinates -> Done")
                Basis = AtomicBasis.getBasis(path + folder)
                print(
                    "Construct carthesian Basis and spherical to cartesian Transformations -> Done"
                )
                if OverlapMatrixFlag == False:
                    Overlapmatrix = AtomicBasis.getTransformationmatrix(
                        Atoms, Atoms, Basis
                    )
                    OverlapMatrixFlag = True
                diff = np.abs(Overlapmatrix - OLM)
                absre.append(np.max(np.max(diff)))
        if SameSizeFlag:
            plt.scatter(cellsizes, absre, marker="x", s=125)
            plt.yscale("log")
            plt.ylabel(
                r"$\displaystyle\text{max}_{i,j}\{\vert S_{i,j}-S_{i,j}^{\text{cp2k}}\vert\} $ ",
                fontsize=50,
            )
            plt.xlabel(r"cell dimension $a$ [$\mathring{\text{A}}$]", fontsize=50)
            plt.xticks(fontsize=35)
            plt.yticks(fontsize=35)
            plt.grid()
            plt.show()
        else:
            fig, ax = plt.subplots()
            ax.scatter(range(len(absre)), absre)
            ax.set_xticks(range(len(absre)))
            ax.set_xticklabels([str(item) for item in cellsizes], fontsize=25)
            ax.set_yscale("log")
            ax.set_ylabel(
                r"$\displaystyle\text{max}_{i,j}\{\vert S_{i,j}-S_{i,j}^{\text{cp2k}}\vert\} $ ",
                fontsize=50,
            )
            plt.grid()
            plt.show()
    elif quantity == "Geo_Opt":
        # Read in the out file
        Forces = []
        Forceflag = False
        with open(path + "/Forces") as g:
            lines = g.readlines()
            for line in lines:
                if len(line.split()) >= 1:
                    if (
                        line.split()[0] == "SUM"
                        and line.split()[1] == "OF"
                        and line.split()[2] == "ATOMIC"
                    ):
                        Forceflag = False
                        Forces.append(np.array(Forcecontainer))
                    elif Forceflag:
                        Forcecontainer.append(
                            np.array(
                                [
                                    float(line.split()[3]),
                                    float(line.split()[4]),
                                    float(line.split()[5]),
                                ]
                            )
                        )
                    elif (
                        line.split()[0] == "#"
                        and line.split()[1] == "Atom"
                        and line.split()[2] == "Kind"
                    ):
                        Forceflag = True
                        Forcecontainer = []
        Forces = np.array(Forces)
        maxforces = []
        rmsforces = []
        iterations = []
        iter = 0
        for fo in Forces:
            iterations.append(iter)
            maxforces.append(np.max(np.max(fo)))
            rmsforces.append(np.sqrt(np.mean(np.square(fo))))
            iter += 1
        plt.scatter(iterations, maxforces)
        # plt.scatter(iterations,rmsforces)
        plt.yscale("log")
        plt.ylabel(r"$\text{max}_{s,\alpha}\{F_{s,\alpha}(n)\}$ ", fontsize=30)
        plt.xlabel(r"Iteration $n$", fontsize=30)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.grid()
        plt.show()
    else:
        print(
            "Option not recognized! Use  'PW_Cutoff','Rel_Cutoff','Cutoff_Radius','Cell_Size' or 'Geo_Opt' as input."
        )
