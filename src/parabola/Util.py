import os
import sys
from . import Read
import numpy as np
import scipy as sci


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def progressbar(it, prefix="", size=60, out=sys.stdout):  # Python3.3+
    count = len(it)

    def show(j):
        x = int(size * j / count)
        print(
            "{}[{}{}] {}/{}".format(prefix, "█" * x, "." * (size - x), j, count),
            end="\r",
            file=out,
            flush=True,
        )

    show(0)
    for i, item in enumerate(it):
        yield item
        show(i + 1)
    print("\n", flush=True, file=out)


def LoewdinTransformation(S, algorithm="Schur-Pade"):
    ##  Function to compute S^(-0.5) for the Loewdin orthogonalization
    ##   input:   S         (numpy array)            the overlapmatrix
    ##
    ##   output:  Sm12      (numpy array)            the overlapmatrix to the power 1/2
    if algorithm == "Diagonalization":
        e, U = np.linalg.eigh(S)
        Sm12 = np.diag(e ** (-0.5))
        Sm12 = np.dot(U, np.dot(Sm12, np.transpose(np.conjugate(U))))
    elif algorithm == "Schur-Pade":
        Sm12 = sci.linalg.fractional_matrix_power(S, -0.5)
    else:
        ValueError(
            "Algorithm not recognized! Currently available 'Schur-Pade' and 'Diagonalization'"
        )
    return Sm12


def getSm1(S, algorithm="Schur-Pade"):
    ##  Function to compute S^(-0.5) for the Loewdin orthogonalization
    ##   input:   S         (numpy array)            the overlapmatrix
    ##
    ##   output:  Sm12      (numpy array)            the overlapmatrix to the power 1/2
    if algorithm == "Diagonalization":
        e, U = np.linalg.eigh(S)
        Sm1 = np.diag(e ** (-1))
        Sm1 = np.dot(U, np.dot(Sm1, np.transpose(np.conjugate(U))))
    elif algorithm == "Schur-Pade":
        Sm1 = sci.linalg.fractional_matrix_power(S, -1)
    else:
        ValueError(
            "Algorithm not recognized! Currently available 'Schur-Pade' and 'Diagonalization'"
        )
    return Sm1


def getNumberofBasisFunctions(parentfolder="./"):
    _, _, OLM = Read.read_ks_matrices(parentfolder)
    dim = np.shape(OLM)[0]
    return dim


def Diagonalize_KS_Hamiltonian(parentfolder="./"):
    ##  Function to compute the electronic energies from the equilibrium file
    ##   input:   parentfolder:         (string)            absolute/relative path, where the geometry optimized .xyz file lies
    ##                                                      in the subfolders there we find the electronic structure at displaced geometries
    UKS = Read.check_uks(parentfolder)
    if not UKS:
        try:
            E = np.load(parentfolder + "/KS_Eigenvalues.npy")
            a_orth = np.load(parentfolder + "/KS_orth_Eigenstates.npy")
            Sm12 = np.load(parentfolder + "/OLMm12.npy")
        except:
            # Read in the KS Hamiltonian
            KSHamiltonian_alpha, _, OLM = Read.read_ks_matrices(parentfolder)
            Sm12 = LoewdinTransformation(OLM)
            KSHorth_alpha = np.dot(Sm12, np.dot(KSHamiltonian_alpha, Sm12))
            E_alpha, a_orth_alpha = np.linalg.eigh(KSHorth_alpha)
            E = np.zeros((np.shape(E_alpha)[0], 1))
            E[:, 0] = E_alpha
            a_orth = np.zeros((np.shape(a_orth_alpha)[0], np.shape(a_orth_alpha)[1], 1))
            a_orth[:, :, 0] = a_orth_alpha
            np.save(parentfolder + "/OLMm12.npy", Sm12)
            np.save(parentfolder + "/KS_Eigenvalues.npy", E)
            np.save(parentfolder + "/KS_orth_Eigenstates.npy", a_orth)
    else:
        try:
            E = np.load(parentfolder + "/KS_Eigenvalues_alpha.npy")
            a_orth = np.load(parentfolder + "/KS_orth_Eigenstates.npy")
            Sm12 = np.load(parentfolder + "/OLMm12.npy")
        except:
            # Read in the KS Hamiltonian
            KSHamiltonian_alpha, KSHamiltonian_beta, OLM = Read.read_ks_matrices(
                parentfolder
            )
            Sm12 = LoewdinTransformation(OLM)
            KSHorth_alpha = np.dot(Sm12, np.dot(KSHamiltonian_alpha, Sm12))
            KSHorth_beta = np.dot(Sm12, np.dot(KSHamiltonian_beta, Sm12))
            E_alpha, a_orth_alpha = np.linalg.eigh(KSHorth_alpha)
            E_beta, a_orth_beta = np.linalg.eigh(KSHorth_beta)
            E = np.zeros((np.shape(E_alpha)[0], 2))
            E[:, 0] = E_alpha
            E[:, 1] = E_beta
            a_orth = np.zeros((np.shape(a_orth_alpha)[0], np.shape(a_orth_alpha)[1], 2))
            a_orth[:, :, 0] = a_orth_alpha
            a_orth[:, :, 1] = a_orth_beta
            np.save(parentfolder + "/OLMm12.npy", Sm12)
            np.save(parentfolder + "/KS_Eigenvalues.npy", E)
            np.save(parentfolder + "/KS_orth_Eigenstates.npy", a_orth)
    return E, a_orth, Sm12


def compressKSfile(parentfolder="./"):
    _, _, _ = Read.read_ks_matrices(parentfolder)


def compressCubefile(parentfolder="./"):
    spinmultiplicity = Read.checkforSpinMultiplicity(parentfolder)
    cubefiles = [f for f in os.listdir(parentfolder) if f.endswith(".cube")]
    for file in cubefiles:
        with open(file, "r") as f:
            summand = 0
            for it, line in enumerate(f):
                if it <= 1:
                    if it == 1:
                        print(line)
                        if line.split()[3] == "1":
                            spin = "alpha"
                        elif line.split()[3] == "2":
                            spin = "beta"
                        else:
                            ValueError(
                                "Multiplicities other then 1 or 2 are not yet implemented"
                            )
                        if line.split()[5] == "HOMO":
                            summand = 0
                        elif line.split()[5] == "LUMO":
                            summand = 1
                        else:
                            ValueError("Fourth column should be either HOMO or LUMO!")
                        number = int(line.split()[7])
                        if spinmultiplicity == 1:
                            filename = str(summand + number)
                        if spinmultiplicity == 2:
                            filename = str(summand + number) + "_" + spin
                else:
                    break
        Orbital = Read.readCubeFile(file)
        np.save(parentfolder + "/" + filename, Orbital)
        os.remove(parentfolder + "/" + file)


def CompressFolder(writeexcludelist=False):
    dirs = [x[0] for x in os.walk("./")]
    dirs = dirs[1:]
    excludelist = []
    for it in progressbar(range(len(dirs)), "Compression Progress:", 40):
        try:
            compressKSfile(dirs[it])
            os.remove(dirs[it]+"/KSHamiltonian")
            excludelist.append(dirs[it])
        except:
            pass
    if writeexcludelist:
        with open("rsync_exclude.txt", "w") as f:
            for element in excludelist:
                f.write(element[2:] + "\n")
