import unittest
import Parabola as p
import numpy as np

class Test_Vib_Ana(unittest.TestCase):
    def test_getHessian(self):
        print("Started Test - getHessian:")
        Modes_Reference=np.load("./Mode_Energies_Reference.npy")
        p.VibAna.getHessian(False,"Y","./Pyrazine_Test/",False)
        Modes_Test=np.load("Normal-Mode-Energies.npy")
        np.testing.assert_array_almost_equal(Modes_Test, Modes_Reference,10,"Vibration Analysis failed!")

if __name__ == '__main__':
    unittest.main()

