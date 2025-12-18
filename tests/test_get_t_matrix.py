import unittest
import parabola as p
import numpy as np
import time

class Test_T_Matrix(unittest.TestCase):
    def test_OBC(self):
        print("\nTesting parabola's", "\033[1m", "getTransformationmatrix", "\033[0m")
        print("Started OBC Test")
        OLM_CP2K = np.load("data/OBC/OLM_OBC.npy")
        Atoms = p.Read.read_atomic_coordinates("data/OBC/Benzene_opt.xyz")
        Basis = p.atomic_basis.getBasis("data/OBC/")
        start_time = time.perf_counter()
        OLM_Parabola = p.atomic_basis.getTransformationmatrix(Atoms, Atoms, Basis)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        print(f"Time taken: {elapsed:.6f} seconds")
        np.testing.assert_array_almost_equal(OLM_Parabola, OLM_CP2K, 13, "OBC Test failed!")

    def test_PBC(self):
        print("\nStarted PBC Test")
        OLM_CP2K = np.load("data/PBC/OLM_PBC.npy")
        Atoms = p.Read.read_atomic_coordinates("data/PBC/Benzene.xyz")
        Basis = p.atomic_basis.getBasis("data/PBC/")
        cell = p.Read.read_cell_vectors(path="data/PBC/", verbose=False)
        cellvectors = p.Geometry.getNeibouringCellVectors(cell, 2, 2, 2)
        start_time = time.perf_counter()
        OLM_Parabola = p.atomic_basis.getTransformationmatrix(Atoms, Atoms, Basis, cellvectors)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        print(f"Time taken: {elapsed:.6f} seconds")
        np.testing.assert_array_almost_equal(OLM_Parabola, OLM_CP2K, 13, "PBC Test failed!")


if __name__ == "__main__":
    unittest.main()
