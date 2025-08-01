import Modules.Read as Read
import Modules.Write as Write
from Modules.PhysConst import ConversionFactors as ConFactors
from Modules.Molecular_Structure import MolecularStructure
import Modules.Symmetry as Symmetry
import numpy as np
from scipy.linalg import schur
import matplotlib.pyplot as plt
import os
pathtocp2k=os.environ["cp2kpath"]
pathtobinaries=pathtocp2k+"/exe/local/"
class VibrationalStructure(MolecularStructure):
    def __init__(self,name,path="./",disable_symmetry=False):
        # If the object was loaded from a pickle, it will already have its
        # vibrational attributes. If so, we can exit immediately.
        # We check for 'Hessian', an attribute unique to this child class.
        if hasattr(self, 'Hessian'):
            return
        # Initialize MolecularStructure
        super().__init__(name, path)
        Hessian=Read.readinHessian(path)
        self.Hessian=Hessian
        self.inverse_sqrt_MassMatrix=np.sqrt(np.linalg.inv(np.kron(np.diag(self.masses),np.eye(3))))
        self.VibrationalModes={}
        if disable_symmetry:
            print("❗ Symmetry Disabled for Vibrational Analysis.")
            # 1. Get the number of atoms
            num_atoms = len(self.masses)
            
            # 2. Create an identity matrix of size N_atoms x N_atoms
            #    This represents the identity permutation of atoms.
            id_matrix = np.eye(num_atoms)

            # 3. Create a new, blank Symmetry object
            self.Molecular_Symmetry.Symmetry_Generators={"Id": id_matrix}
            
        # Now safe to access parent attributes
        self.Vibrational_Symmetry = Vibrational_Symmetry(self.Molecular_Symmetry)
        self.ImposeTranslationalSymmetry()
        self.getVibrationalModes()
        self.save()
    def getVibrationalModes(self):
        def TransformHessian(Hessian,Axis):
            M = np.array(Axis).T  # M is 3x3
            N = Hessian.shape[0] // 3  # Number of blocks
            block = np.kron(np.eye(N), M)  # Build 3N x 3N block diagonal matrix
            return block.T @ Hessian @ block, block
        
        # --- Setup and Hessian Transformation ---
        Hessian,O=TransformHessian(self.Hessian,self.Geometric_UC_Principle_Axis)
        sqrtMm1 = self.inverse_sqrt_MassMatrix
        MassWeightedHessian = sqrtMm1 @ Hessian @ sqrtMm1
        # Convert units to get frequencies in cm^-1
        MassWeightedHessian *= (10**3 / 1.8228884842645) * (2.19474631370540E+02)**2
        
        # (Optional) Visualize the original mass-weighted Hessian
        plt.imshow(MassWeightedHessian, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.savefig("./Hessian_Original.png")
        plt.close()
        
        # --- CONSISTENT SYMMETRY ORDERING ---
        SymSectors = self.Vibrational_Symmetry.SymSectors
        VIrr = self.Vibrational_Symmetry.IrrepsProjector
        
        # 1. Sort symmetry labels alphabetically to ensure a consistent order.
        sorted_sym_labels = sorted(SymSectors.keys())
        
        # 2. Build the reordering array based on the sorted labels.
        # This groups basis functions by symmetry, in a fixed order.
        reordering = np.concatenate([SymSectors[key] for key in sorted_sym_labels])
        
        # 3. Reorder the projector matrix columns based on the sorted symmetry order.
        VIrr_reordered = VIrr[:, reordering]
        
        # 4. Create the block-diagonal Hessian. The blocks are now in a consistent order.
        MassWeightedHessian_Sectors = VIrr_reordered.T @ MassWeightedHessian @ VIrr_reordered
        
        # (Optional) Visualize the block-diagonalized Hessian
        plt.imshow(MassWeightedHessian_Sectors, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.savefig("./Hessian_Sectors.png")
        plt.close()
        
        # --- BLOCK DIAGONALIZATION (IN CONSISTENT ORDER) ---
        print(f"ℹ️ : Symmetry Sectors and Vibrational Frequencies")
        current_index = 0
        # 5. Iterate through the sorted labels to process each block.
        for sym in sorted_sym_labels:
            block_size = len(SymSectors[sym])
            
            # Extract the symmetry block
            Block_Hessian = MassWeightedHessian_Sectors[current_index : current_index + block_size,
                                                        current_index : current_index + block_size]
            
            # Diagonalize the block to get eigenvalues and eigenvectors for this symmetry
            eigenvalues, eigenvectors = np.linalg.eigh(Block_Hessian)
            
            # Frequencies (cm^-1) are the signed square roots of the eigenvalues
            frequencies_cm_inv = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues))
            print(f"Symmetry Sector: {sym}, Frequencies (cm⁻¹): \n {frequencies_cm_inv}")
            
            # Transform eigenvectors from the symmetry basis back to the mass-weighted basis
            V_mwh = O@VIrr_reordered[:, current_index : current_index + block_size] @ eigenvectors
            
            # Store the calculated vibrational modes
            if sym not in self.VibrationalModes:
                self.VibrationalModes[sym] = []
            
            for i in range(block_size):
                mode = VibrationalMode(frequencies_cm_inv[i], V_mwh[:, i], sym, self)
                self.VibrationalModes[sym].append(mode)
                
            # Advance the index to the start of the next block
            current_index += block_size
        self.WriteMolFile()
    def ImposeTranslationalSymmetry(self):
        # Imposes exact relation on the Hessian, that has to be fullfilled for the Translational D.O.F. to decouple from the rest
        #input: 
        #Hessian:    (numpy.array)  Hessian in carthesian coordinates 
        #Hessian:    (numpy.array)  Hessian in carthesian coordinates, which has been cleaned from contamination of the Translations
        def getJ(masses):
            # Computes the J matrix, the linear, invertible transformation, that transforms into the Jacobi coordinates, in which the center of mass
            # motion explicitly decouples, x_rel=J*x
            #input: 
            #Hessian:    (numpy.array)  Hessian in carthesian coordinates 
            #Hessian:    (numpy.array)  Hessian in carthesian coordinates, which has been cleaned from contamination of the Translations
            Rcm=np.zeros(len(masses))
            Rcm[0]=1.0
            Mj=masses[0]
            Jupdown=np.zeros(((len(masses),len(masses))))
            for j in range(1,len(masses)):
                Rj=np.zeros(len(masses))
                Rj[j]=1.0
                Deltaj=Rj-Rcm
                Jupdown[j-1][:]=Deltaj
                Rcm=(Mj*Rcm+masses[j]*Rj)/(Mj+masses[j])
                Mj+=masses[j]
            Jupdown[-1][:]=Rcm
            J=np.zeros((3*len(masses),3*len(masses)))
            for it1 in range(len(masses)):
                for it2 in range(len(masses)):
                    J[3*it1][3*it2]=Jupdown[it1][it2]
                    J[3*it1+1][3*it2+1]=Jupdown[it1][it2]
                    J[3*it1+2][3*it2+2]=Jupdown[it1][it2]
            return J
        
        J=getJ(self.masses)
        Hessiantilde=np.transpose(np.linalg.inv(J))@self.Hessian@np.linalg.inv(J)
        for it1 in range(3*len(self.masses)):
            for it2 in range(3*len(self.masses)):
                if it1>=3*len(self.masses)-3 or it2>=3*len(self.masses)-3:
                    Hessiantilde[it1][it2]=0.0
        Hessian=np.transpose(J)@Hessiantilde@J
        self.Hessian=0.5*(Hessian+np.transpose(Hessian))
    def CheckSymmetry(self):
        for sym in self.Vibrational_Symmetry.Symmetry_Generators:
            print(sym)
            comm1=self.Hessian@self.Vibrational_Symmetry.Symmetry_Generators[sym]-self.Vibrational_Symmetry.Symmetry_Generators[sym]@self.Hessian
            comm2=self.MassMatrix@self.Vibrational_Symmetry.Symmetry_Generators[sym]-self.Vibrational_Symmetry.Symmetry_Generators[sym]@self.MassMatrix
            print(np.linalg.norm(comm1))
            print(np.linalg.norm(comm2))
            print(np.linalg.norm(self.Hessian))
    def WriteMolFile(self):
        '''
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
        '''
        ConFactor=ConFactors()
        ##########################################
        #Get the number of atoms from the xyz file
        ##########################################
        numofatoms=len(self.masses)
        coordinates=self.coordinates
        atoms=self.atoms
        SymSectors=[sym for sym in self.VibrationalModes]
        normalmodeEnergies=[]
        Normalized_Displacement=[]
        normfactors=[]
        for sym in SymSectors:
            for Mode in self.VibrationalModes[sym]:
                if not Mode.istranslation and not Mode.isrotation:
                    normalmodeEnergies.append(Mode.frequency)
                    Normalized_Displacement.append(Mode.Normalized_Displacement)
                    normfactors.append(Mode.NormFactor)
        with open(self.path+"/Vibrations.mol",'w') as f:
            f.write('[Molden Format]\n')
            f.write('[FREQ]\n')
            for Frequency in normalmodeEnergies:
                f.write('   '+str(Frequency)+'\n')
            f.write('[FR-COORD]\n')
            for it,coord in enumerate(coordinates):
                f.write(atoms[it]+'   '+str(coord[0]*ConFactor["A->a.u."])+'   '+str(coord[1]*ConFactor["A->a.u."])+'   '+str(coord[2]*ConFactor["A->a.u."])+'\n')
            f.write('[SYMMETRY-SECTORS]\n')
            for sym in SymSectors:
                f.write(sym+'\n')
            f.write('[NORM-FACTORS]\n')
            for normfactor in normfactors:
                f.write(str(normfactor)+'\n')
            f.write('[FR-NORM-COORD]\n')
            modeiter=1
            for mode in Normalized_Displacement:
                f.write('vibration      '+str(modeiter)+'\n')
                for s in range(numofatoms):
                    f.write('   '+str(round(mode[3*s], 12))+'   '+str(round(mode[3*s+1], 12))+'   '+str(round(mode[3*s+2],12))+'\n')
                modeiter+=1
class XYZ_Symmetry(Symmetry.Symmetry):
    def __init__(self,Molecular_Symmetry):
        super().__init__()  # Initialize parent class
        self.xyz_Generators(Molecular_Symmetry)
    def xyz_Generators(self,Molecular_Symmetry):
        Molecular_Symmetry_Generators=Molecular_Symmetry.Symmetry_Generators
        xyz_Generators={}
        for symmetrylabel in Molecular_Symmetry_Generators.keys():
            xyz_Generators[symmetrylabel]=getXYZRepresentation(symmetrylabel)
        self.Symmetry_Generators=xyz_Generators
class Vibrational_Symmetry(Symmetry.Symmetry):
    def __init__(self,Molecular_Symmetry):
        super().__init__()  # Initialize parent class
        molecular_symmetry=Molecular_Symmetry
        if "Id" not in Molecular_Symmetry.Symmetry_Generators.keys():
            xyz_symmetry=XYZ_Symmetry(Molecular_Symmetry)
            generators={}
            for sym in molecular_symmetry.Symmetry_Generators:
                generators[sym]=np.kron(molecular_symmetry.Symmetry_Generators[sym],xyz_symmetry.Symmetry_Generators[sym])
            self.Symmetry_Generators=generators
            self._iscommutative()
            if self.commutative:
                self.IrrepsProjector=simultaneous_real_block_diagonalization(list(self.Symmetry_Generators.values()))
            else:
                self._determineCentralizer()
                self._determineIrrepsProjector()
        else:
            self.IrrepsProjector=np.kron(molecular_symmetry.Symmetry_Generators["Id"],np.eye(3))
        self._determineSymmetrySectors()
class VibrationalMode:
    def __init__(self, frequency,NormalMode, symmetry_label,VibrationalStructure):
        self.frequency = frequency
        self.Normal_Mode=NormalMode
        disp=VibrationalStructure.inverse_sqrt_MassMatrix@NormalMode
        normfactor=np.sqrt(np.dot(disp,disp))
        self.Normalized_Displacement = disp/normfactor
        self.NormFactor=normfactor
        self.symmetry_label = symmetry_label
        self.istranslation=False
        self.isrotation=False
        self.isTranslation()
        if VibrationalStructure.periodicity==(0,0,0):
            self.isRotation(VibrationalStructure)
    
    def isTranslation(self,tolerance=0.95):
        NumDisplacements=len(self.Normalized_Displacement)
        projector=np.zeros((NumDisplacements,NumDisplacements))
        for it in range(3):
            TransEigenvector = np.zeros(NumDisplacements)
            TransEigenvector[it::3] = 1.0/np.sqrt(NumDisplacements/3.0)
            projector+=np.outer(TransEigenvector,TransEigenvector)
        if np.abs(np.dot(self.Normalized_Displacement,projector@self.Normalized_Displacement))>tolerance:
            self.istranslation=True

    def isRotation(self,VibrationalStructure,tolerance=0.8):
        centerofmasscoordinates=VibrationalStructure.Mass_UC_Centered_Coordinates
        [v1,v2,v3]=VibrationalStructure.Geometric_UC_Principle_Axis
        [u1,u2,u3]=VibrationalStructure.Mass_UC_Principle_Axis
        V=np.array([v1,v2,v3])
        U=np.array([u1,u2,u3])
        T=V@U.T
        masses=VibrationalStructure.masses
        Roteigenvectors=[]
        for it in [0,1,2]:
            #Define the RotEigenvector
            RotEigenvector=np.zeros(3*len(masses))
            #generate the vector X along the principle axis
            X=np.zeros(3)
            X[it]=1.0
            for s in range(len(masses)):
                rvector=np.cross(X,centerofmasscoordinates[s])
                #Expand rvector in basis [v1,v2,v3]
                rvector=T@rvector
                RotEigenvector[3*s]=rvector[0]
                RotEigenvector[3*s+1]=rvector[1]
                RotEigenvector[3*s+2]=rvector[2]
            #Normalize the generated Eigenvector
            RotEigenvector/=np.linalg.norm(RotEigenvector)
            Roteigenvectors.append(RotEigenvector)
        projector=np.zeros((3*len(masses),3*len(masses)))
        for it in range(3):
            projector+=np.outer(Roteigenvectors[it],Roteigenvectors[it])
        if np.abs(np.dot(self.Normalized_Displacement,projector@self.Normalized_Displacement))>tolerance:
            self.isrotation=True
    
    def getTransEigenvectors(self):
        """
        Computes the translational according to "Vibrational Analysis in Gaussian,"
        Joseph W. Ochterski (1999).
        Reference: https://gaussian.com/wp-content/uploads/dl/vib.pdf

        Version: 01.07.2023

        Parameters:
        - pathtoEquilibriumxyz (string, optional): Path to the folder of the Equilibrium_Geometry calculation.
        - rescale (bool, optional): Flag to rescale the eigenvectors based on atom masses. Default is True.

        Returns:
        - Transeigenvectors (list of np.arrays): Translational eigenvectors in the rescaled Cartesian Basis |\tilde{s,alpha}>.
        - Roteigenvectors (list of np.arrays): Rotational eigenvectors in the rescaled Cartesian Basis |\tilde{s,alpha}>.

        Notes:
        - The function relies on the `readCoordinatesAndMasses`, `getInertiaTensor`, and `ComputeCenterOfMassCoordinates` functions.
        - The rotational eigenvectors are generated based on the principle axis obtained from the inertia tensor.
        - The translational eigenvectors are generated along each Cartesian axis.
        - The generated eigenvectors can be rescaled based on atom masses if the `rescale` flag is set to True.
        - All generated eigenvectors are normalized.
        """

        masses=self.masses
        numofatoms=int(len(masses))
        Transeigenvectors=[]
        for it in [0,1,2]:
            #Define the TransEigenvector
            TransEigenvector=np.zeros(3*numofatoms)
            for s in range(numofatoms):
                TransEigenvector[3*s+it]=1.0
            #Normalize the generated Eigenvector
            TransEigenvector/=np.linalg.norm(TransEigenvector)
            Transeigenvectors.append(TransEigenvector)
        return Transeigenvectors
#### Define Symmetry Class ####
def detect_block_sizes(matrix, tol=1e-8):
        """
        Detects block sizes in a (approximately) block-diagonal square matrix.
        Args:
            matrix: (n x n) NumPy array (assumed square and block-diagonal).
            tol: threshold below which off-diagonal elements are considered zero.

        Returns:
            A list of (start_index, block_size) tuples.
        """
        n = matrix.shape[0]
        assert matrix.shape[0] == matrix.shape[1], "Matrix must be square."
        blocks = []
        i = 0

        while i < n:
            block_found = False
            for size in range(1, n - i + 1):
                # Extract candidate block
                block = matrix[i:i+size, i:i+size]

                # Check if it's isolated: off-block rows/cols should be zero
                off_block = matrix[i:i+size, i+size:]
                off_block_T = matrix[i+size:, i:i+size]

                if np.all(np.abs(off_block) < tol) and np.all(np.abs(off_block_T) < tol):
                    # Check if next row/col introduces new coupling
                    if i + size == n:
                        blocks.append((i, size))
                        i += size
                        block_found = True
                        break
                    next_col = matrix[i:i+size, i+size]
                    next_row = matrix[i+size, i:i+size]
                    if np.all(np.abs(next_col) < tol) and np.all(np.abs(next_row) < tol):
                        blocks.append((i, size))
                        i += size
                        block_found = True
                        break
            if not block_found:
                # Fallback: treat single diagonal element as a block
                blocks.append((i, 1))
                i += 1

        return blocks
def simultaneous_real_block_diagonalization(matrices):
    """
    Block-diagonalize a set of mutually commuting real matrices using a single real orthogonal basis.
    
    The matrices are assumed to commute and be real-valued. This function finds a single
    real orthogonal matrix Q that transforms every matrix A in the input list into a
    block-diagonal matrix Q.T @ A @ Q. The blocks are 1x1 for real eigenvalues
    and 2x2 for complex-conjugate eigenvalue pairs.

    Parameters:
    -----------
    matrices : list of np.ndarray
        A list of real-valued, square, mutually commuting matrices of the same shape (n x n).

    Returns:
    --------
    Q : np.ndarray
        The real orthogonal matrix (n x n) that simultaneously block-diagonalizes all matrices.

    transformed_matrices : list of np.ndarray
        The list of transformed (block-diagonal) matrices.
    """
    if not matrices:
        raise ValueError("Input list of matrices cannot be empty.")

    n = matrices[0].shape[0]
    for A in matrices:
        if A.shape != (n, n):
            raise ValueError("All matrices must be square and have the same shape.")
        # This check is good practice, though scipy.linalg.schur can handle complex inputs.
        if not np.allclose(A, A.real):
            raise ValueError("All matrices must be real-valued.")

    # 1. Create a random linear combination of the matrices.
    # Since all matrices commute, any linear combination of them also commutes with them.
    # A random combination is very likely to have distinct eigenvalues, which simplifies
    # the identification of the common eigenspaces.
    C = np.zeros((n, n))
    for A in matrices:
        C += np.random.randn() * A

    # 2. Compute the real Schur decomposition of the combined matrix C.
    # The schur function returns an orthogonal matrix Q and a block-upper-triangular
    # matrix T (the "real Schur form") such that C = Q @ T @ Q.T.
    # This matrix Q is the transformation we need.
    _, Q = schur(C, output='real')
    return Q
    
    
def getXYZRepresentation(symmetrylabel):
    """
    Return the 3 × 3 Cartesian (XYZ) matrix representation of a basic
    point‑symmetry operation.

    The function currently recognises **inversion, proper rotations,
    mirror reflections** and a trivial identity operation, using the
    following label grammar:

    ──────────────────────────────────────────────────────────────────────
    Label       Meaning                               Returned matrix
    ──────────────────────────────────────────────────────────────────────
    "i"         Inversion through the origin          −I₃
    "C<axis>_<n>"
                n‑fold proper rotation about          R_axis(2π / n)
                the given axis (x, y or z)            (right‑hand rule)
    "S<axis>"   Mirror (σ) plane normal to <axis>     diag(±1, ±1, ±1)
    "t"         Identity (useful placeholder)         I₃
    ──────────────────────────────────────────────────────────────────────

    Parameters
    ----------
    symmetrylabel : str
        A string encoded as described above.
        Examples: "i", "Cx_2", "Cy_4", "Sz", "t".

    Returns
    -------
    numpy.ndarray
        A 3 × 3 `float64` NumPy array representing the operation in
        Cartesian coordinates.

    Raises
    ------
    ValueError
        If *symmetrylabel* does not conform to any of the supported
        patterns.

    Notes
    -----
    * **Right‑hand convention** – Positive rotation angles follow the
      right‑hand rule about the specified axis.
    * Rotations are constructed with ``theta = 2π / n`` (radians), so
      ``Cx_2`` is a 180 ° (π) rotation, ``Cz_4`` is a 90 ° (π/2) rotation,
      etc.
    * The mirror (“S”) labels here implement *simple* reflections, not
      roto‑reflections; extend as needed for improper rotations **Sₙ**.

    Examples
    --------
    >>> getXYZRepresentation("i")
    array([[-1.,  0.,  0.],
           [ 0., -1.,  0.],
           [ 0.,  0., -1.]])

    >>> getXYZRepresentation("Cy_4")        # 90° rotation about y
    array([[ 0. ,  0. ,  1. ],
           [ 0. ,  1. ,  0. ],
           [-1. ,  0. ,  0. ]])

    >>> getXYZRepresentation("Sz")          # mirror in xy‑plane
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0., -1.]])
    """
    if symmetrylabel=="i":
        return (-1.0)*np.eye(3)
    elif symmetrylabel[0]=="C":
        theta=2*np.pi/int(symmetrylabel.split("_")[1])
        if symmetrylabel[1]=="x":
            return np.array([[1, 0, 0],[0, np.cos(theta), -np.sin(theta)],[0, np.sin(theta),  np.cos(theta)]]).T
        elif symmetrylabel[1]=="y":
            return np.array([[np.cos(theta), 0, np.sin(theta)],[0, 1, 0],[-np.sin(theta), 0, np.cos(theta)]]).T
        elif symmetrylabel[1]=="z":
            return np.array([[np.cos(theta), -np.sin(theta), 0],[np.sin(theta),  np.cos(theta), 0],[0, 0, 1]]).T
    elif symmetrylabel[0]=="S":
        if symmetrylabel[1]=="x":
            return np.array([[-1, 0, 0],[0, 1, 0],[0, 0, 1]])
        elif symmetrylabel[1]=="y":
            return  np.array([[1, 0, 0],[0, -1, 0],[0, 0, 1]])
        elif symmetrylabel[1]=="z":
            return np.array([[1, 0, 0],[0, 1, 0],[0, 0, -1]])
    elif symmetrylabel[0]=="t":
        return np.eye(3)







def test_IrrepsProjector(name):
    v=VibrationalStructure(name)
    SymSectors = v.Vibrational_Symmetry.SymSectors
    VIrr = v.Vibrational_Symmetry.IrrepsProjector
    
    # 1. Sort symmetry labels alphabetically to ensure a consistent order.
    sorted_sym_labels = sorted(SymSectors.keys())
    
    # 2. Build the reordering array based on the sorted labels.
    # This groups basis functions by symmetry, in a fixed order.
    reordering = np.concatenate([SymSectors[key] for key in sorted_sym_labels])
    
    # 3. Reorder the projector matrix columns based on the sorted symmetry order.
    VIrr_reordered = VIrr[:, reordering]
    for sym in v.Vibrational_Symmetry.Symmetry_Generators:
        symm=VIrr_reordered.T@v.Vibrational_Symmetry.Symmetry_Generators[sym]@VIrr_reordered
        plt.imshow(symm, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.title(sym)
        # Save the figure
        plt.savefig("./{}.png".format(sym))
        plt.close()
