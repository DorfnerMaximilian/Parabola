import unittest
import parabola as p
import numpy as np


class Test_Phase_Operators(unittest.TestCase):
    def test_OBC(self):
        print("\nTesting parabola's", "\033[1m", "get_phase_operators", "\033[0m")
        print("Started OBC Test")
        regression_data = np.load("data/OBC/phase_operators.npy")
        Atoms = p.Read.read_atomic_coordinates("data/OBC/Benzene_opt.xyz")
        Basis = p.atomic_basis.getBasis("data/OBC/")
        parabola_data = p.atomic_basis.get_phase_operators(Atoms, Basis)
        np.testing.assert_array_almost_equal(parabola_data, regression_data, 13, "OBC Test failed!")
    
    def test_PBC(self):
        print("\nStarted PBC Test")
        regression_data = np.load("data/PBC/phase_operators.npy")
        Atoms = p.Read.read_atomic_coordinates("data/PBC/Benzene.xyz")
        Basis = p.atomic_basis.getBasis("data/PBC/")
        cell = p.Read.read_cell_vectors(path="data/PBC/", verbose=False)
        cellvectors = p.Geometry.getNeibouringCellVectors(cell, 2, 2, 2)
        parabola_data = p.atomic_basis.get_phase_operators(Atoms, Basis, cell_vectors=cellvectors)
        np.testing.assert_array_almost_equal(parabola_data, regression_data, 13, "PBC Test failed!")


if __name__ == "__main__":
    unittest.main()
