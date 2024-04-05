import unittest
import Parabola as p
import numpy as np
import os
class Test_Vib_Ana(unittest.TestCase):
    def test_getHessian(self):
        print("Started Test - getHessian:")
        parentfolder="./Pyrazine_Test/"
        Modes_Reference=np.load(parentfolder+"/Mode_Energies_Reference.npy")
        p.VibAna.getHessian(False,"Y","./Pyrazine_Test/",False)
        Modes_Test=np.load(parentfolder+"/Normal-Mode-Energies.npy")
        np.testing.assert_array_almost_equal(Modes_Test, Modes_Reference,2,"Vibration Analysis failed!")
        os.system("rm "+parentfolder+"/normalized-Carthesian-Displacements.npy")
        os.system("rm "+parentfolder+"/Normal-Mode-Energies.npy")
        os.system("rm "+parentfolder+"/Norm-Factors.npy")

if __name__ == '__main__':
    unittest.main()

