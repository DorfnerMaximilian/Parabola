import unittest

import numpy as np

import parabola as p


class Test_Sublattice(unittest.TestCase):
    def test_graphene(self):
        print("\nTesting parabola's", "\033[1m", "sublattice", "\033[0m", "detection for graphene")
        graphene_supercell331 = p.structure.Molecular_Structure(name="graphene_supercell331", path="data/sublattice/")
        self.assertEqual(graphene_supercell331.periodicity, (6, 3, 1))

        # check if coordinate transofrmation is correct
        coordinates_reconstructed_hexagonal_supercell = np.load(
            "data/sublattice/reconstructed_hexagonal_supercell_coordinates.npy"
        )
        np.testing.assert_array_almost_equal(
            graphene_supercell331.coordinates,
            coordinates_reconstructed_hexagonal_supercell,
            decimal=6,
            err_msg="Graphene sublattice test failed!",
        )



if __name__ == "__main__":
    unittest.main()
