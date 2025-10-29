import unittest
import parabola as p
import numpy as np


class Test_Wfn_on_XYZ_Grid(unittest.TestCase):
    def test_OBC(self):
        print("\nTesting parabola's", "\033[1m", "WFNonxyzGrid", "\033[0m")
        print("Started OBC Test")
        regression_data = np.load("data/OBC/wfn_on_xyz_grid.npy")
        homo_id = p.Read.read_homo_index("data/OBC/")
        _, A, Sm12 = p.Util.Diagonalize_KS_Hamiltonian("data/OBC/")
        coefficients = (Sm12 @ A[:, homo_id]) * p.TDDFT.getPhaseOfMO(A[:, homo_id])
        Atoms = p.Read.read_atomic_coordinates("data/OBC/Benzene_opt.xyz")
        Basis = p.atomic_basis.getBasis("data/OBC/")
        parabola_data = p.atomic_basis.WFNonxyzGrid(
            grid1=np.linspace(0, 20, 20, endpoint=True),
            grid2=np.linspace(0, 20, 20, endpoint=True),
            grid3=np.linspace(0, 20, 20, endpoint=True),
            Coefficients=coefficients,
            Atoms=Atoms,
            Basis=Basis,
        )
        np.testing.assert_array_almost_equal(parabola_data, regression_data, 12, "OBC Test failed!")

    def test_PBC(self):
        print("\nStarted PBC Test")
        regression_data = np.load("data/PBC/wfn_on_xyz_grid.npy")
        homo_id = p.Read.read_homo_index("data/PBC/")
        _, A, Sm12 = p.Util.Diagonalize_KS_Hamiltonian("data/PBC/")
        coefficients = ( Sm12 @ A[:, homo_id]) * p.TDDFT.getPhaseOfMO(A[:, homo_id])
        Atoms = p.Read.read_atomic_coordinates("data/PBC/Benzene.xyz")
        Basis = p.atomic_basis.getBasis("data/PBC/")
        cell = p.Read.read_cell_vectors(path="data/PBC/", verbose=False)
        cellvectors = p.Geometry.getNeibouringCellVectors(cell, 2, 2, 2)
        parabola_data = p.atomic_basis.WFNonxyzGrid(
            grid1=np.linspace(0, np.linalg.norm(cell[0]), 20, endpoint=True),
            grid2=np.linspace(0, np.linalg.norm(cell[1]), 20, endpoint=True),
            grid3=np.linspace(0, np.linalg.norm(cell[2]), 20, endpoint=True),
            Coefficients=coefficients,
            Atoms=Atoms,
            Basis=Basis,
            cell_vectors=cellvectors,
        )
        np.testing.assert_array_almost_equal(parabola_data, regression_data, 12, "PBC Test failed!")


if __name__ == "__main__":
    unittest.main()
