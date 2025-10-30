import unittest
import parabola as p
import numpy as np


class Test_Local_Potential_on_XYZ_Grid(unittest.TestCase):
    def test_OBC(self):
        print("\nTesting parabola's", "\033[1m", "LocalPotentialonxyzGrid", "\033[0m")
        print("Started OBC Test")
        regression_data = np.load("data/OBC/local_potential_on_xyz_grid.npy")
        matrixelements = p.Read.read_ks_matrices("data/OBC/")[0]
        Atoms = p.Read.read_atomic_coordinates("data/OBC/Benzene_opt.xyz")
        Basis = p.atomic_basis.getBasis("data/OBC/")
        grid_x, grid_y, grid_z = (
            np.linspace(0, 20, 10, endpoint=True),
            np.linspace(0, 20, 10, endpoint=True),
            np.linspace(0, 20, 10, endpoint=True),
        )
        gridpoints = np.array(np.meshgrid(grid_x, grid_y, grid_z, indexing="ij")).flatten("F").tolist()
        parabola_data = p.atomic_basis.LocalPotentialonxyzGrid(gridpoints=gridpoints,
            MatrixElements=matrixelements,
            Atoms=Atoms,
            Basis=Basis,
        )
        np.testing.assert_array_almost_equal(parabola_data, regression_data, 13, "OBC Test failed!")

    def test_PBC(self):
        print("\nStarted PBC Test")
        regression_data = np.load("data/PBC/local_potential_on_xyz_grid.npy")
        matrixelements = p.Read.read_ks_matrices("data/PBC/")[0]
        Atoms = p.Read.read_atomic_coordinates("data/PBC/Benzene.xyz")
        Basis = p.atomic_basis.getBasis("data/PBC/")
        cell = p.Read.read_cell_vectors(path="data/PBC/", verbose=False)
        cellvectors = p.Geometry.getNeibouringCellVectors(cell, 2, 2, 2)
        grid_x, grid_y, grid_z = (
            np.linspace(0, np.linalg.norm(cell[0]), 5, endpoint=True),
            np.linspace(0, np.linalg.norm(cell[1]), 5, endpoint=True),
            np.linspace(0, np.linalg.norm(cell[2]), 5, endpoint=True),
        )
        gridpoints = np.array(np.meshgrid(grid_x, grid_y, grid_z, indexing="ij")).flatten("F").tolist()
        parabola_data = p.atomic_basis.LocalPotentialonxyzGrid(gridpoints=gridpoints,
            MatrixElements=matrixelements,
            Atoms=Atoms,
            Basis=Basis,
            cell_vectors=cellvectors,
        )
        np.testing.assert_array_almost_equal(parabola_data, regression_data, 13, "PBC Test failed!")


if __name__ == "__main__":
    unittest.main()
